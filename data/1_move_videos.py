"""
After decide the original data, we run this script to move all the files into
the appropriate train/test/validation folders.

Prerequisites:
Should rename orginal videos as "classname_XXXX_date_XXXX.avi"
    e.g. 00_threeway_20180507_091400-091900.mp4
"""

import numpy as np
from pathlib import Path


def move_videos():
    """Move videos into train/test folders
    
    Move all the videos in specified folder into train or test folders.
    """
#    vid_folder = '/media/usb_storage/Data/MA_5min/'
#    vid_files = list(Path(vid_folder).glob('*.avi'))
#    vid_folder = '../threeway/'
    vid_folder = '../eastgate/'
    vid_files = list(Path(vid_folder).glob('*.mp4'))
    
    for video_path in vid_files:
        filename = video_path.name
        filename_no_ext = video_path.stem
        
        parts = filename_no_ext.split('_')
        classname = parts[0]
        
        # Mark set
        group = mark_train_test(parts)
        if group == 'unknown':
            continue
        
        # Check if this class folder exists.
        outpath = Path('./' + group + '/' + classname)
        if not outpath.exists():
            print("Creating folder for %s/%s" % (group, classname))
            outpath.mkdir(parents=True) #If parents is true, any missing parents of this path are created as needed

        # Check if we have already moved this file.
        output_name = outpath / filename
        if output_name.exists():
            print("Already moved %s. Skipping." % (output_name))
            continue

        # Move file.
        print("Moving %s to %s. (Softlink)" % (video_path, output_name))
        output_name.symlink_to(video_path.resolve()) # Need full path to link

    print('Done!')

def mark_train_test(filename_parts):
    """Mark video with train/test
    
    This script contains the rules with which we create train and test set.
    The rules are based on the filename_parts string list
    """
    assert len(filename_parts)>=4
    classname, location, date, time_interval = filename_parts[:4]
    
    # choose class
#    if not classname in ['00', '02', '04']:
#        return 'unknown'
#    if classname in ['00', '01', '02', '03', '04']:
    if int(classname) <= 15:
        return 'train'
    else:
        return 'unknown'
    
#    # group by random. 20% test.
#    ra = np.random.rand()
#    if ra < 0.2:
#        return 'test'
#    else:
#        return 'train'
    
#    # group by date and time
#    if date > '20180501' and date < '20180512':
#        return 'train'
#    elif date == '20180512' and time_interval < '132000-132500':
#        return 'train'
#    elif date == '20180512':
#        return 'test'
#    else:
#        return 'unknown'

def main():

    move_videos()

if __name__== '__main__':
    main()