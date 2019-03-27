"""
After moving all the videos, we run this script to extract the images from the 
videos and also create a data file we can use for training and testing later.

Note: 
    1. run 1_move_videos.py first; 
    2. copy full frame image first to speed up
       e.g.: cp -rl ../data/train ./
"""
import argparse
import csv
import glob
import os
import datetime as dt
from subprocess import call
import cv2
from tqdm import tqdm


def extract_images(sam_interval, sam_num, class_sam_num, num_con_frame, 
                   diff=False, flow=False, cutup=None):
    """Extract images and generate a csv file
    
    Extract the images from the videos and also create a 'data_file.csv' which 
    can be used as reference for training and testing later.
    
    The content in 'data_file.csv' is formated as:
        [train|test], class, filename_no_ext_i, nb frames
        
    Args:
        float sam_interval: Time interval of sampling (seconds). 
            Sample 1 frame every 'sam_interval' seconds.
        int sam_num: The number of samples from each video. 
            Sample with time lag equals to 'sam_interval/sam_num' seconds.
        int class_sam_num: The number of samples from each class. 
            This arg means balancing the number of samples in each class, 
            which will generate class_sam_num samples from each class.
        int num_con_frame: The number of continous frames.
    """

    data_file = []
    folders = ['./train/', './test/'] # Add './validation/' if necessary
    ext = args.videotype
    
    if class_sam_num > 0:
        print('Generate class_sam_num = {} samples from each class.'.format(class_sam_num))
        pbar = tqdm(total=class_sam_num*len(glob.glob('./train/*'))) # not accurate because of rounding
    elif sam_num > 0:
        print('Generate sam_num = {} samples from each video.'.format(sam_num))
        pbar = tqdm(total=sam_num*len(glob.glob('./*/*/*.'+ext))) 
    else:
        print('Need positive interger for sam_num or class_sam_num.')
        return

    for folder in folders:
        class_folders = glob.glob(folder + '*')

        for vid_class in class_folders:
            class_files = glob.glob(vid_class + '/*.'+ext)
            
            if class_sam_num > 0:
                # calculate the average sample number for each video
                sam_num = int(round(class_sam_num/len(class_files),0))

            for video_path in class_files:
                # Get the parts of the file.
                video_parts = get_video_parts(video_path)
                train_or_test, classname, filename_no_ext, filename = video_parts

                # Generate sam_num samples
                for i in range(sam_num):
                    # Only extract if we haven't done it yet. Otherwise, just get
                    # the info.
                    filename_no_ext_i = filename_no_ext + '-' + '%03d'%(i)
                    if not os.path.exists(train_or_test + '/' + classname + 
                            '/' + filename_no_ext_i + '-0001.jpg'):
                        # Now extract images
                        src = video_path
                        dest = train_or_test + '/' + classname + '/' + \
                            filename_no_ext_i + '-%04d.jpg'
                        time_lag = dt.timedelta(seconds=sam_interval/sam_num * i)
                        time_lag = str(time_lag)
                        # lz:'-q:v'(qscale:v) is used to set quality of jpg, 1 is highest, 31 is lowest. we use '10'
                        # '-q:v', '10' should be put before dest
                        call(['ffmpeg', '-v', 'quiet', '-ss', time_lag, '-i', src, '-y',
                              '-f', 'image2', '-q:v', '1', '-vf', 
                              # In list form, don't need to pass double quate for select parameter.
                              'select=lt(mod(n\\,30 * %f)\\,%d)' % (sam_interval, num_con_frame),
                              '-vsync', 'vfr', dest])
#                        call(['ffmpeg', '-ss', time_lag, '-i', src, '-y',
#                              '-f', 'image2', '-q:v', '1', '-vf', 
#                              '\"select=not(mod(n\\,30 * %d))\"' % sam_interval, # This line failed
#                              '-vsync', 'vfr', dest])
                    
                    # Preprocessing: differential frames    
                    if diff==True:
                        generate_diff_image(video_parts, i, num_con_frame)
                        # Now get how many frames it is.
                        nb_frames = get_nb_frames(train_or_test + '/' + classname + '/' + \
                                                  filename_no_ext_i + '_diff')
                        data_file.append([train_or_test, classname, 
                                          filename_no_ext_i + '_diff', nb_frames])
                    # Preprocessing: optical flow    
                    elif flow==True:
                        generate_flow_image(video_parts, i, num_con_frame)
                        # Now get how many frames it is.
                        nb_frames = get_nb_frames(train_or_test + '/' + classname + '/' + \
                                                  filename_no_ext_i + '_flow_x')
                        data_file.append([train_or_test, classname, 
                                          filename_no_ext_i + '_flow', nb_frames])
                    # Preprocessing: cut up frames    
                    elif cutup != None:
                        cutup_image(video_parts, i, cutup)
                        num_row, num_col = cutup
                        for k in range(num_row * num_col):
                            # Now get how many frames it is.
                            nb_frames = get_nb_frames(train_or_test + '/' + classname + '/' + \
                                                      filename_no_ext_i + '_cut-%03d'%k)
                            data_file.append([train_or_test, classname, 
                                              filename_no_ext_i + '_cut-%03d'%(k), nb_frames])
                            
                    else:
                        # Now get how many frames it is.
                        nb_frames = get_nb_frames(train_or_test + '/' + classname + '/' + \
                                                  filename_no_ext_i)
                        data_file.append([train_or_test, classname, 
                                          filename_no_ext_i, nb_frames])
    
#                    print("Generated %d frames for %s" % (nb_frames, filename_no_ext_i))
                    
                    pbar.update(1)
    
    with open('data_file.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)

    print("Extracted and wrote %d video files." % (len(data_file)))

    pbar.close()

def get_nb_frames(video_path):
    """Given video path of an (assumed) already extracted video, return
    the number of frames that were extracted.
    
    Args:
        str video_path: Str of a full path. 
    """
    generated_files = glob.glob(video_path + '*.jpg')
    return len(generated_files)

def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    video_path = video_path.replace('\\','/') # Adapt for win
    parts = video_path.split('/')
    filename = parts[3]
    filename_no_ext = filename.split('.')[0]
    classname = parts[2]
    train_or_test = parts[1]

    return train_or_test, classname, filename_no_ext, filename

def generate_diff_image(video_parts, sam_index, num_con_frame):
    """Generate differencial images and delete original images.
    
    Args:
        list video_parts: Parts of a full path. Return of get_video_parts().
        int sam_index: The index of current sample.
        int num_con_frame: The number of continuous frames.
    """
    assert num_con_frame > 1
    
    train_or_test, classname, filename_no_ext, filename = video_parts
    filename_no_ext_i = filename_no_ext + '-' + '%03d'%(sam_index)
    images = sorted(glob.glob(train_or_test + '/' + classname + '/' + filename_no_ext_i + '*jpg'))

    # Loop of sample interval
    for i in range(len(images)//num_con_frame):
        # Loop of continuous frames
        for j in range(num_con_frame - 1):
            img1 = cv2.imread(images[i * num_con_frame + j])
            img2 = cv2.imread(images[i * num_con_frame + j + 1])
            # lz:OpenCV uses BGR as its default colour order for images, matplotlib uses RGB. 
            # The easiest way of fixing this is to use OpenCV to explicitly convert it back to RGB
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            # lz: absolute difference                
            img = cv2.absdiff(img1,img2)
            # lz: save diff image
            diff_filename=train_or_test + '/' + classname + '/' + \
                filename_no_ext_i + '_diff-%04d.jpg' %(i * (num_con_frame - 1) + j + 1)
            cv2.imwrite(diff_filename, img)
    
    # delete original frames
    for im in images:
        os.remove(im)

def generate_flow_image(video_parts, sam_index, num_con_frame):
    """Generate optical flow images and delete original images.
    
    Args:
        list video_parts: Parts of a full path. Return of get_video_parts().
        int sam_index: The index of current sample.
        int num_con_frame: The number of continuous frames.
    """
    assert num_con_frame > 1
    
    train_or_test, classname, filename_no_ext, filename = video_parts
    filename_no_ext_i = filename_no_ext + '-' + '%03d'%(sam_index)
    images = sorted(glob.glob(train_or_test + '/' + classname + '/' + filename_no_ext_i + '*jpg'))
    
    op_method = args.op_method
    bound = args.bound

    # Loop of sample interval
    for i in range(len(images)//num_con_frame):
        # Loop of continuous frames
        for j in range(num_con_frame - 1):
            img1 = cv2.imread(images[i * num_con_frame + j])
            img2 = cv2.imread(images[i * num_con_frame + j + 1])
            
            if args.crop != None:
                img1 = img1[args.crop[0]:args.crop[1], args.crop[2]:args.crop[3]]
                img2 = img2[args.crop[0]:args.crop[1], args.crop[2]:args.crop[3]]
            
            # OpenCV uses BGR as its default colour order for images, matplotlib uses RGB. 
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            if op_method == 'tvl1': # Dual tvl1 algorithm
#                dtvl1 = cv2.createOptFlow_DualTVL1()
                dtvl1 = cv2.DualTVL1OpticalFlow_create() # the same with createOptFlow_DualTVL1
                flows = dtvl1.calc(img1,img2,None)
            elif op_method == 'fb': # Farneback
                flows = cv2.calcOpticalFlowFarneback(img1,img2, None, 0.6, 3, 25, 7, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)       

            # save flow image
            flow_x = ToImg(flows[...,0],bound)
            flow_y = ToImg(flows[...,1],bound)
            flow_x_filename = train_or_test + '/' + classname + '/' + \
                filename_no_ext_i + '_flow_x-%04d.jpg' %(i * (num_con_frame - 1) + j + 1)
            flow_y_filename = train_or_test + '/' + classname + '/' + \
                filename_no_ext_i + '_flow_y-%04d.jpg' %(i * (num_con_frame - 1) + j + 1)
            cv2.imwrite(flow_x_filename, flow_x)
            cv2.imwrite(flow_y_filename, flow_y)
    
    # delete original frames
    for im in images:
        os.remove(im)

def ToImg(raw_flow,bound):
    """Scale the input pixels to 0-255 with bi-bound
    
    Args:
        param raw_flow: input raw pixel value (not in 0-255)
        param bound: upper and lower bound (-bound, bound)
    Returns:
        pixel value scale from 0 to 255
    """
    flow=raw_flow
    flow[flow>bound]=bound
    flow[flow<-bound]=-bound
    flow-=-bound
    flow*=(255/float(2*bound))
    return flow

def cutup_image(video_parts, sam_index, num_cutup):
    """Cutup images and delete original images.
    
    Args:
        list video_parts: Parts of a full path. Return of get_video_parts().
        int sam_index: The index of current sample.
        tube num_cutup: (num_row, num_col). e.g.: (4,4) means 4x4 cut up
    """
    train_or_test, classname, filename_no_ext, filename = video_parts
    filename_no_ext_i = filename_no_ext + '-' + '%03d'%(sam_index)
    images = sorted(glob.glob(train_or_test + '/' + classname + '/' + filename_no_ext_i + '*jpg'))

    num_row, num_col = num_cutup
    # the video resolution
    v_height = 720
    v_width = 1280
    # resolution of video piece
    out_w = v_width//num_col
    out_h = v_height//num_row
    
    for cut_idx, image in enumerate(images):
        src = image
        dest = train_or_test + '/' + classname + '/' + \
            filename_no_ext_i + '_cut-%03d-%04d.jpg'

        # Only cut up if we haven't cut it yet.
        if not os.path.exists(dest%(0,cut_idx+1)):
            for h in range(num_row):
                for w in range(num_col):
                    x = w*out_w
                    y = h*out_h
                    #ffmpeg -i in.mp4 -filter:v "crop=out_w:out_h:x:y" out.mp4
                    crop_arg = "crop=" + str(out_w) + ":" + str(out_h) + ":" + str(x) + ":" + str(y)
                    call(["ffmpeg", '-v', 'quiet', "-i", src, "-filter:v", crop_arg,
                          dest%(h*num_col+w, cut_idx+1)])
            
        # delete original video
        os.remove(image)


    
def main():

    global args
    
    start_time = dt.datetime.now()
    print('Start running at {}'.format(str(start_time)))

    # options
    parser = argparse.ArgumentParser(
        description="Extract images/differential images/optical flow")
    parser.add_argument('--sam_interval', type=int, default=5)
    parser.add_argument('--sam_num', type=int, default=0) 
    parser.add_argument('--class_sam_num', type=int, default=0) 
    parser.add_argument('--num_con_frame', type=int, default=1)
    parser.add_argument('--diff', type=bool, default=False)
    parser.add_argument('--flow', type=bool, default=False)
    parser.add_argument('--cutup', nargs='+', type=int) # This is equivalent to 'default=None'
    parser.add_argument('--videotype', type=str, default='mp4',
                        choices=['mp4', 'avi'])

    # options for flow mode
    parser.add_argument('--op_method', type=str, default='tvl1',
                        choices=['tvl1', 'fb'])
    parser.add_argument('--bound', type=int, default=10)
    parser.add_argument('--crop', nargs='+', type=int, default=None) # nargs="+" (meaning one or more); index of array img[50:306, 250:506]
    
    args = parser.parse_args()

    extract_images(sam_interval=args.sam_interval, 
                   sam_num=args.sam_num, 
                   class_sam_num=args.class_sam_num,
                   num_con_frame=args.num_con_frame, 
                   diff=args.diff, 
                   flow=args.flow, 
                   cutup=args.cutup
                   )
    
    end_time = dt.datetime.now()
    print('Stop running at {}'.format(str(end_time)))
    elapsed_time = end_time - start_time
    print("Total running time is {}.".format(str(elapsed_time)))

if __name__ == '__main__':
    main()

# command: 
"""
# full frame
    # balance
        python 2_extract_images.py --class_sam_num 100 --num_con_frame 6
    # fix
        python 2_extract_images.py --sam_num 10 --num_con_frame 6 --videotype avi
# cutup
    python 2_extract_images.py --class_sam_num 100 --cutup (4,4)
# diff
    python 2_extract_images.py --class_sam_num 100 --num_con_frame 6 --diff True
# flow
    # tvl1
    python 2_extract_images.py --class_sam_num 100 --num_con_frame 6 --flow True --bound 20 --crop 50 306 250 506
    # fb
    python 2_extract_images.py --class_sam_num 100 --num_con_frame 6 --flow True --op_method fb --bound 20 --crop 50 306 250 506
"""


"""
TODO: multi process. 

However, above program performs similar with/without multiprocessing.
Maybe, the C++ code behind cv2.DualTVL1OpticalFlow_create() use some paralel technique.

from multiprocessing import Pool, cpu_count
            with Pool(cpu_count()) as pool:
                pool.starmap(generate_flow_image, zip([video_parts]*sam_num, range(sam_num), [num_con_frame]*sam_num))

"""