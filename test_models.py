import argparse
import time
import datetime as dt
from tqdm import tqdm

import numpy as np
import torch.nn.parallel
import torch.optim
from sklearn.metrics import confusion_matrix

from dataset import TSNDataSet
from models import TSN
from transforms import *
from ops import ConsensusModule

import itertools
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt


start_time = dt.datetime.now()
print('Start running at {}'.format(str(start_time)))

# options
parser = argparse.ArgumentParser(
    description="Standard video-level testing")
parser.add_argument('dataset', type=str, choices=['ucf101', 'hmdb51', 'kinetics', 'kaist', 'ma'])
parser.add_argument('num_class', type=int, default=0)
parser.add_argument('modality', type=str, choices=['RGB', 'Flow', 'RGBDiff'])
parser.add_argument('root_path', type=str)
parser.add_argument('test_list', type=str)
parser.add_argument('weights', type=str)
parser.add_argument('--arch', type=str, default="resnet101")
parser.add_argument('--save_scores', type=str, default=None)
parser.add_argument('--test_segments', type=int, default=25)
parser.add_argument('--num_spacial_segments', type=int, default=1)
parser.add_argument('--max_num', type=int, default=-1)
parser.add_argument('--test_crops', type=int, default=10)
parser.add_argument('--input_size', type=int, default=224)
parser.add_argument('--crop_fusion_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk'])
parser.add_argument('--k', type=int, default=3)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--gpus', nargs='+', type=int, default=None)
parser.add_argument('--flow_prefix', type=str, default='')

args = parser.parse_args()


#if args.dataset == 'ucf101':
#    num_class = 101
#elif args.dataset == 'hmdb51':
#    num_class = 51
#elif args.dataset == 'kinetics':
#    num_class = 400
#elif args.dataset == 'kaist':
#    num_class = 3
#elif args.dataset == 'ma':
#    num_class = 5
#else:
#    raise ValueError('Unknown dataset '+args.dataset)
num_class = args.num_class

net = TSN(num_class, 1, args.modality,
          base_model=args.arch,
          consensus_type=args.crop_fusion_type,
          dropout=args.dropout)

checkpoint = torch.load(args.weights)
print("model epoch {} best prec@1: {}".format(checkpoint['epoch'], checkpoint['best_prec1']))

base_dict = {'.'.join(k.split('.')[1:]): v for k,v in list(checkpoint['state_dict'].items())}
net.load_state_dict(base_dict)

if args.test_crops == 1:
    cropping = torchvision.transforms.Compose([
        GroupScale(net.scale_size),
        GroupCenterCrop(net.input_size),
    ])
elif args.test_crops == 10:
    cropping = torchvision.transforms.Compose([
        GroupOverSample(net.input_size, net.scale_size)
    ])
#else:
#    raise ValueError("Only 1 and 10 crops are supported while we got {}".format(args.test_crops))

data_loader = torch.utils.data.DataLoader(
        TSNDataSet(args.root_path, args.test_list, train_val_test='test', num_segments=args.test_segments,
                   new_length=1 if args.modality == "RGB" else 5,
                   modality=args.modality,
                   image_tmpl="-{:04d}.jpg" if args.modality in ['RGB', 'RGBDiff'] else args.flow_prefix+"_{}-{:04d}.jpg",
                   test_mode=True,
                   transform=torchvision.transforms.Compose([
#                       GroupCrop((576,1024),(0,0)),#size:sequence like (h, w) offset:coordinate of top left pixel (x,y)
#                       GroupCrop((720*0.8, 1280*0.8),(0,0)), # robust test
#                       GroupCenterCrop((720*0.4, 1280*0.4)), # robust test
#                       GroupCrop(224,(250,50)),# tree
#                       GroupCrop(224,(1000,0)),# sky
#                       GroupRandomCrop(256),
#                       GroupNRandomCrop(224, args.test_crops),
#                       GroupNRandomCrop(224, 4), # r4TSSN
                       GroupMbyNCrop(2, 2, 720, 1280), # 2x2TSSN
#                       GroupMbyNCrop(4, 4, 720, 1280), # 4x4TSSN
#                       GroupMbyNRandomCrop(4, 4, 4, 720, 1280), # one out of 2x2TSSN
#                       GroupScale((224,224)),
#                       ImgNRandomCrop(224, args.test_crops),
                       cropping,
                       Stack(roll=args.arch == 'BNInception'),
                       ToTorchFormatTensor(div=args.arch != 'BNInception'),
                       GroupNormalize(net.input_mean, net.input_std),
                   ])),
        batch_size=1, shuffle=False,
        num_workers=args.workers * 2, pin_memory=True)

if args.gpus is not None:
    devices = [args.gpus[i] for i in range(args.workers)]
else:
#    devices = list(range(args.workers))
    devices = list(range(1))


net = torch.nn.DataParallel(net.cuda(devices[0]), device_ids=devices)
net.eval()

data_gen = enumerate(data_loader)

total_num = len(data_loader.dataset)
output = []


def eval_video(video_data):
    i, data, label = video_data
    num_crop = args.test_crops * args.num_spacial_segments

    if args.modality == 'RGB':
        length = 3
    elif args.modality == 'Flow':
        length = 10
    elif args.modality == 'RGBDiff':
        length = 18
    else:
        raise ValueError("Unknown modality "+args.modality)

    input_var = torch.autograd.Variable(data.view(-1, length, data.size(2), data.size(3)),
                                        volatile=True)
    rst = net(input_var).data.cpu().numpy().copy()
    return i, rst.reshape((num_crop, args.test_segments, num_class)).mean(axis=0).reshape( # avg of crops
        (args.test_segments, 1, num_class) #lz: 1 seems useless -> to use default_aggregation_func from pyActionRecog.utils.video_funcs 
    ), label[0]


proc_start_time = time.time()
max_num = args.max_num if args.max_num > 0 else len(data_loader.dataset)

pbar = tqdm(total=total_num)

for i, (data, label) in data_gen:
    if i >= max_num:
        break
    rst = eval_video((i, data, label))
    output.append(rst[1:])
    cnt_time = time.time() - proc_start_time
    pbar.update(1)
#    print('video {} done, total {}/{}, average {} sec/video'.format(i, i+1,
#                                                                    total_num,
#                                                                    float(cnt_time) / (i+1)))
pbar.close()

video_pred = [np.argmax(np.mean(x[0], axis=0)) for x in output] # avg of segments

video_labels = [x[1] for x in output]

cf = confusion_matrix(video_labels, video_pred, labels=np.arange(num_class)).astype(float)
print(cf)

cls_cnt = cf.sum(axis=1)
cls_hit = np.diag(cf)

cls_acc = cls_hit / cls_cnt

print(cls_acc)

print('Accuracy {:.02f}%'.format(np.nanmean(cls_acc) * 100))

if args.save_scores is not None:

    # reorder before saving
#    name_list = [x.strip().split()[0] for x in open(args.test_list)]
    name_list = [x.path + str(time.time()) for x in data_loader.dataset.video_list] # use timestamp to distinguish repeated sample

    order_dict = {e:i for i, e in enumerate(sorted(name_list))}

    reorder_output = [None] * len(output)
    reorder_label = [None] * len(output)

    for i in range(len(output)):
        idx = order_dict[name_list[i]]
        reorder_output[idx] = output[i]
        reorder_label[idx] = video_labels[i]

    np.savez(args.save_scores, scores=reorder_output, labels=reorder_label)


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix',saveas='cm', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    
    print(cm)

    plt.figure()    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    foo_fig = plt.gcf() # 'get current figure'
#    foo_fig.savefig('confusion_matrix.eps', format='eps', dpi=1000) 
    foo_fig.savefig(saveas, dpi=1000, bbox_inches='tight')
    plt.show()


plot_confusion_matrix(
        cf, 
        classes=np.arange(num_class)+1,#data.classes, 
        normalize=True, 
        title='Confusion matrix',
        saveas='Confusion_matrix_normalized_%s_%d'%(args.modality, num_class))



end_time = dt.datetime.now()
print('Stop running at {}'.format(str(end_time)))
elapsed_time = end_time - start_time
print("Total running time is {}.".format(str(elapsed_time)))
