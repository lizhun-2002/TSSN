import argparse
import sys
import numpy as np
sys.path.append('.')

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt

from pyActionRecog.utils.video_funcs import default_aggregation_func
from pyActionRecog.utils.metrics import mean_class_accuracy

parser = argparse.ArgumentParser()
parser.add_argument('score_files', nargs='+', type=str)
parser.add_argument('--score_weights', nargs='+', type=float, default=None)
parser.add_argument('--crop_agg', type=str, choices=['max', 'mean'], default='mean')
parser.add_argument('--grid', type=bool, default=False)
parser.add_argument('--target_score', type=str, default=None)
args = parser.parse_args()

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def eval_scores(score_files, score_weights, agg_method):
    """Fuse the score files of different models
    
    Args:
        list(str) score_files: file names of score files
        list(float) score_weights: weights of score files
        str agg_method: the name of method for aggregating the segment level scores
            This is because the current used scores are segment level scores. See test_models.py.
    Returns:
        int: the fused accuracy.
    """
    score_npz_files = [np.load(x) for x in score_files]
    
    if score_weights is None:
        score_weights = [1] * len(score_npz_files)
    else:
        if len(score_weights) != len(score_npz_files):
            raise ValueError("Only {} weight specifed for a total of {} score files"
                             .format(len(score_weights), len(score_npz_files)))
    
    score_list = [x['scores'][:, 0] for x in score_npz_files] # x['scores'] has two columns [segment level score, label]
    label_list = [x['labels'] for x in score_npz_files]
    
    # label verification
    
    # score_aggregation
    agg_score_list = []
    for score_vec in score_list:
        agg_score_vec = [default_aggregation_func(x, normalization=False, crop_agg=getattr(np, agg_method)) for x in score_vec]#lz:video level scores
        agg_score_list.append(np.array(agg_score_vec))
    
    final_scores = np.zeros_like(agg_score_list[0])
    for i, agg_score in enumerate(agg_score_list):
        final_scores += agg_score * score_weights[i]
    
    # accuracy
    acc = mean_class_accuracy(final_scores, label_list[0])
    
    # softmax score
    softmax_scores = [softmax(vec) for vec in final_scores]
    return acc, softmax_scores

acc, soft_scores = eval_scores(score_files=args.score_files, score_weights=args.score_weights, agg_method=args.crop_agg)
print('Final accuracy {:02f}%'.format(acc * 100))



# transpose the class labels to continuous value (soft_scores weighted sum of labels)
if args.target_score is not None:

    target_labels = np.load(args.target_score)['labels']
#    print(target_labels)
    
    if len(soft_scores[0])==3:
        class_labels = np.array([1, 3, 5])
    elif len(soft_scores[0])==5:
        class_labels = np.array([1, 2, 3, 4, 5])
    elif len(soft_scores[0])==4:
        class_labels = np.array([0, 2, 4, 6])
        
    target_labels = [class_labels[x] for x in target_labels]
#    print(target_labels)
    pred_continuous_labels = [np.sum(class_labels * p) for p in soft_scores]
#    print(pred_continuous_labels)
    
    mae = mean_absolute_error(target_labels, pred_continuous_labels)
    rmse = sqrt(mean_squared_error(target_labels, pred_continuous_labels))
    print('MAE {:02f}'.format(mae))
    print('RMSE {:02f}'.format(rmse))



# grid search a proper score weights. This should be conducted on validation data.
if args.grid:
    num_files = len(args.score_files)
    if num_files == 2:
        w1 = [1]
    elif num_files == 3:
        w1 = np.arange(1,11,1)
    else:
        raise
    w2 = np.arange(1,11,1)
    w3 = np.arange(1,11,1)
    res = np.zeros((len(w2), len(w3)))
    
    original = np.get_printoptions()
    np.set_printoptions(precision=3)
    
    for k, w1_ in enumerate(w1):
        for i, w2_ in enumerate(w2):
            for j, w3_ in enumerate(w3):
                if num_files == 2:
                    weights = [w2_, w3_]
                elif num_files == 3:
                    weights = [w1_, w2_, w3_]
                acc, _ = eval_scores(score_files=args.score_files, score_weights=weights, agg_method=args.crop_agg)
                res[i,j] = acc
        print(res)
        print('The max acc is {} at {}'.format(np.max(res), np.argmax(res)+1))
    
    np.set_printoptions(**original)





#score_npz_files = [np.load(x) for x in args.score_files]
#
#if args.score_weights is None:
#    score_weights = [1] * len(score_npz_files)
#else:
#    score_weights = args.score_weights
#    if len(score_weights) != len(score_npz_files):
#        raise ValueError("Only {} weight specifed for a total of {} score files"
#                         .format(len(score_weights), len(score_npz_files)))
#
#score_list = [x['scores'][:, 0] for x in score_npz_files] # x['scores'] has two columns [segment level score, label]
#label_list = [x['labels'] for x in score_npz_files]
#
## label verification
#
## score_aggregation
#agg_score_list = []
#for score_vec in score_list:
#    agg_score_vec = [default_aggregation_func(x, normalization=False, crop_agg=getattr(np, args.crop_agg)) for x in score_vec]#lz:video level scores
#    agg_score_list.append(np.array(agg_score_vec))
#
#final_scores = np.zeros_like(agg_score_list[0])
#for i, agg_score in enumerate(agg_score_list):
#    final_scores += agg_score * score_weights[i]
#
## accuracy
#acc = mean_class_accuracy(final_scores, label_list[0])
#print('Final accuracy {:02f}%'.format(acc * 100))