import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import csv

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file, train_val_test,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='-{:04d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path # absolute path of the data 
        self.list_file = os.path.join(self.root_path, list_file)
        self.train_val_test = train_val_test
        self.classes = self.get_classes() # Get the classes.
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self.video_list = self._parse_list()

    def _load_image(self, directory, idx):
        """Load one image (set).
        
        Args:
            str directory: Image path excluding the self.root_path.
            int idx: The index of image to be loaded.
        Returns:
            A list of images
        """
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(directory + self.image_tmpl.format(idx)).convert('RGB')] # cannot use os.path.join, as we don't need '/'
        elif self.modality == 'Flow':
            x_img = Image.open(directory + self.image_tmpl.format('x', idx)).convert('L')
            y_img = Image.open(directory + self.image_tmpl.format('y', idx)).convert('L')

            return [x_img, y_img]

    def get_classes(self):
        """Extract the classes from our data. If we want to limit them,
        only return the classes we need."""
        classes = []
        # load datafile.csv
        with open(self.list_file) as datafile:
            reader = csv.reader(datafile)
            data = list(reader)
        for item in data:
            if item[1] not in classes:
                classes.append(item[1])
        # Sort them.
        classes = sorted(classes)
        # Return.
        return classes

    def _parse_list(self):
        """Read self.list_file to parse train/test/val lists.
        
        Args:
#            list val_list: Video name w/o extension (stem).
#            list test_list: Video name w/o extension (stem).
#            list train_list: Video name w/o extension (stem).
            None. 
        Returns:
            A list of VideoRecord objects
        """
###############################################################################        
        # data 191/235 mix
        # threeway: 9 class ['00', '01', '02', '03', '04', ..., '8']
        val_list = ['00_eastgate_20180507_095400-095900',
                '00_eastgate_20180507_104400-104900',
                '00_eastgate_20180512_133000-133500',
                '00_eastgate_20180626_163000-163500',
                '00_eastgate_20180630_173000-173500',
                '00_eastgate_20180701_090500-091000',
                '00_eastgate_20180701_162500-163000',
                '01_eastgate_20180507_074900-075400',
                '01_eastgate_20180512_074400-074900',
                '01_eastgate_20180626_073200-073700',
                '01_eastgate_20180626_131600-132100',
                '01_eastgate_20180702_151700-152200',
                '01_eastgate_20180706_073800-074300',
                '01_eastgate_20180706_115100-115600',
                '02_eastgate_20180512_072300-072800',
                '02_eastgate_20180512_102600-103100',
                '02_eastgate_20180626_123800-124300',
                '02_eastgate_20180701_082800-083300',
                '02_eastgate_20180702_163500-164000',
                '02_eastgate_20180706_065000-065500',
                '03_eastgate_20180512_065500-070000',
                '03_eastgate_20180512_092000-092500',
                '03_eastgate_20180512_105200-105700',
                '03_eastgate_20180701_153900-154400',
                '03_eastgate_20180702_173200-173700',
                '04_eastgate_20180512_082000-082500',
                '04_eastgate_20180630_163500-164000',
                '04_eastgate_20180702_160000-160500',
                '05_eastgate_20180701_132300-132800',
                '05_eastgate_20180709_113100-113600',
                '06_eastgate_20180701_063200-063700',
                '07_eastgate_20180701_155800-160300',
                '00_threeway_20180507_095400-095900',
                '00_threeway_20180507_104400-104900',
                '00_threeway_20180512_133000-133500',
                '00_threeway_20180626_163000-163500',
                '00_threeway_20180630_173000-173500',
                '00_threeway_20180701_090500-091000',
                '00_threeway_20180701_162500-163000',
                '01_threeway_20180507_074900-075400',
                '01_threeway_20180512_074400-074900',
                '01_threeway_20180626_073200-073700',
                '01_threeway_20180626_131600-132100',
                '01_threeway_20180702_151700-152200',
                '01_threeway_20180706_073800-074300',
                '01_threeway_20180706_115100-115600',
                '02_threeway_20180512_072300-072800',
                '02_threeway_20180512_102600-103100',
                '02_threeway_20180626_123800-124300',
                '02_threeway_20180701_082800-083300',
                '02_threeway_20180702_163500-164000',
                '02_threeway_20180706_065000-065500',
                '03_threeway_20180512_065500-070000',
                '03_threeway_20180512_092000-092500',
                '03_threeway_20180512_105200-105700',
                '03_threeway_20180701_153900-154400',
                '03_threeway_20180702_173200-173700',
                '04_threeway_20180512_082000-082500',
                '04_threeway_20180630_163500-164000',
                '04_threeway_20180702_160000-160500',
                '05_threeway_20180701_132300-132800',
                '05_threeway_20180709_113100-113600',
                '06_threeway_20180701_063200-063700',
                '07_threeway_20180701_155800-160300'
                ]
        test_list = ['00_eastgate_20180507_093400-093900',
                '00_eastgate_20180507_102400-102900',
                '00_eastgate_20180512_131000-131500',
                '00_eastgate_20180626_162000-162500',
                '00_eastgate_20180626_164500-165000',
                '00_eastgate_20180701_085500-090000',
                '00_eastgate_20180701_092000-092500',
                '00_eastgate_20180706_094000-094500',
                '01_eastgate_20180507_061200-061700',
                '01_eastgate_20180512_071600-072100',
                '01_eastgate_20180512_160500-161000',
                '01_eastgate_20180626_082700-083200',
                '01_eastgate_20180702_150700-151200',
                '01_eastgate_20180706_063100-063600',
                '01_eastgate_20180706_083600-084100',
                '01_eastgate_20180709_105900-110400',
                '02_eastgate_20180512_065200-065700',
                '02_eastgate_20180512_094100-094600',
                '02_eastgate_20180512_173200-173700',
                '02_eastgate_20180630_165000-165500',
                '02_eastgate_20180702_153000-153500',
                '02_eastgate_20180702_171700-172200',
                '02_eastgate_20180706_120100-120600',
                '03_eastgate_20180512_064600-065100',
                '03_eastgate_20180512_082900-083400',
                '03_eastgate_20180512_101700-102200',
                '03_eastgate_20180512_170300-170800',
                '03_eastgate_20180702_155000-155500',
                '03_eastgate_20180706_120500-121000',
                '03_eastgate_20180709_125000-125500',
                '04_eastgate_20180630_151800-152300',
                '04_eastgate_20180701_080800-081300',
                '04_eastgate_20180702_161400-161900',
                '05_eastgate_20180701_064400-064900',
                '05_eastgate_20180701_180600-181100',
                '05_eastgate_20180709_125900-130400',
                '06_eastgate_20180701_152200-152700',
                '07_eastgate_20180709_113700-114200',
                '08_eastgate_20180701_151000-151500',
                '00_threeway_20180507_093400-093900',
                '00_threeway_20180507_102400-102900',
                '00_threeway_20180512_131000-131500',
                '00_threeway_20180626_162000-162500',
                '00_threeway_20180626_164500-165000',
                '00_threeway_20180701_085500-090000',
                '00_threeway_20180701_092000-092500',
                '00_threeway_20180706_094000-094500',
                '01_threeway_20180507_061200-061700',
                '01_threeway_20180512_071600-072100',
                '01_threeway_20180512_160500-161000',
                '01_threeway_20180626_082700-083200',
                '01_threeway_20180702_150700-151200',
                '01_threeway_20180706_063100-063600',
                '01_threeway_20180706_083600-084100',
                '01_threeway_20180709_105900-110400',
                '02_threeway_20180512_065200-065700',
                '02_threeway_20180512_094100-094600',
                '02_threeway_20180512_173200-173700',
                '02_threeway_20180630_165000-165500',
                '02_threeway_20180702_153000-153500',
                '02_threeway_20180702_171700-172200',
                '02_threeway_20180706_120100-120600',
                '03_threeway_20180512_064600-065100',
                '03_threeway_20180512_082900-083400',
                '03_threeway_20180512_101700-102200',
                '03_threeway_20180512_170300-170800',
                '03_threeway_20180702_155000-155500',
                '03_threeway_20180706_120500-121000',
                '03_threeway_20180709_125000-125500',
                '04_threeway_20180630_151800-152300',
                '04_threeway_20180701_080800-081300',
                '04_threeway_20180702_161400-161900',
                '05_threeway_20180701_064400-064900',
                '05_threeway_20180701_180600-181100',
                '05_threeway_20180709_125900-130400',
                '06_threeway_20180701_152200-152700',
                '07_threeway_20180709_113700-114200',
                '08_threeway_20180701_151000-151500'
                ]


#        # data 218/235, use different day for testing
#        # threeway: 16 class ['00', '01', '02', '03', '04', ..., '15']
#        val_list = ['00_threeway_20180507_095400-095900',
#                '00_threeway_20180507_104400-104900',
#                '00_threeway_20180512_133000-133500',
#                '00_threeway_20180626_163000-163500',
#                '00_threeway_20180630_173000-173500',
#                '00_threeway_20180701_090500-091000',
#                '00_threeway_20180701_162500-163000',
#                '01_threeway_20180507_074900-075400',
#                '01_threeway_20180512_074400-074900',
#                '01_threeway_20180626_073200-073700',
#                '01_threeway_20180626_131600-132100',
#                '01_threeway_20180702_151700-152200',
#                '02_threeway_20180512_072300-072800',
#                '02_threeway_20180512_102600-103100',
#                '02_threeway_20180626_123800-124300',
#                '02_threeway_20180701_082800-083300',
#                '02_threeway_20180702_163500-164000',
#                '03_threeway_20180512_065500-070000',
#                '03_threeway_20180512_092000-092500',
#                '03_threeway_20180512_105200-105700',
#                '03_threeway_20180701_153900-154400',
#                '03_threeway_20180702_173200-173700',
#                '04_threeway_20180512_082000-082500',
#                '04_threeway_20180630_163500-164000',
#                '04_threeway_20180702_160000-160500',
#                '05_threeway_20180701_132300-132800',
#                '06_threeway_20180701_063200-063700',
#                '07_threeway_20180701_155800-160300',
#                '11_threeway_20180701_070500-071000',
#                '12_threeway_20180701_151700-152200',
#                '14_threeway_20180701_183100-183600',
#                '15_threeway_20180701_172700-173200'
#                ]
#        test_list = ['00_threeway_20180706_093000-093500',
#                '00_threeway_20180706_094000-094500',
#                '00_threeway_20180706_095000-095500',
#                '00_threeway_20180706_100000-100500',
#                '01_threeway_20180706_063100-063600',
#                '01_threeway_20180706_065800-070300',
#                '01_threeway_20180706_073800-074300',
#                '01_threeway_20180706_074900-075400',
#                '01_threeway_20180706_081000-081500',
#                '01_threeway_20180706_083600-084100',
#                '01_threeway_20180706_085100-085600',
#                '01_threeway_20180706_115100-115600',
#                '01_threeway_20180706_125100-125600',
#                '01_threeway_20180709_091500-092000',
#                '01_threeway_20180709_105900-110400',
#                '02_threeway_20180706_063700-064200',
#                '02_threeway_20180706_065000-065500',
#                '02_threeway_20180706_081900-082400',
#                '02_threeway_20180706_083100-083600',
#                '02_threeway_20180706_120100-120600',
#                '02_threeway_20180706_124000-124500',
#                '03_threeway_20180706_064100-064600',
#                '03_threeway_20180706_120500-121000',
#                '03_threeway_20180706_121900-122400',
#                '03_threeway_20180706_122500-123000',
#                '03_threeway_20180709_091900-092400',
#                '03_threeway_20180709_125000-125500',
#                '05_threeway_20180709_092900-093400',
#                '05_threeway_20180709_113100-113600',
#                '05_threeway_20180709_121300-121800',
#                '05_threeway_20180709_125900-130400',
#                '06_threeway_20180709_115400-115900',
#                '07_threeway_20180709_113700-114200',
#                '07_threeway_20180709_114900-115400',
#                '07_threeway_20180709_131100-131600',
#                '08_threeway_20180709_114300-114800',
#                '10_threeway_20180709_100500-101000',
#                '10_threeway_20180709_104000-104500',
#                '10_threeway_20180709_104400-104900',
#                '11_threeway_20180709_120200-120700',
#                '13_threeway_20180709_095000-095500',
#                '13_threeway_20180709_102500-103000',
#                '13_threeway_20180709_103200-103700',
#                '15_threeway_20180709_095500-100000'
#                ]



###############################################################################        
#        # Version 3
#        # threeway: 5 class ['00', '01', '02', '03', '04']
#        val_list = ['00_threeway_20180507_095400-095900',
#                '00_threeway_20180507_104400-104900',
#                '01_threeway_20180507_073600-074100',
#                '01_threeway_20180512_073100-073600',
#                '02_threeway_20180512_065200-065700',
#                '02_threeway_20180512_094100-094600',
#                '03_threeway_20180512_065000-065500',
#                '03_threeway_20180512_091500-092000',
#                '03_threeway_20180512_102100-102600',
#                ]
#        test_list = ['00_threeway_20180507_093400-093900',
#                '00_threeway_20180507_102400-102900',
#                '00_threeway_20180512_131000-131500',
#                '01_threeway_20180507_060200-060700',
#                '01_threeway_20180512_071100-071600',
#                '01_threeway_20180512_100100-100600',
#                '02_threeway_20180507_072300-072800',
#                '02_threeway_20180512_072600-073100',
#                '02_threeway_20180512_144800-145300',
#                '03_threeway_20180512_063900-064400',
#                '03_threeway_20180512_081600-082100',
#                '03_threeway_20180512_101200-101700',
#                '03_threeway_20180512_145300-145800',
##                '04_threeway_20180512_075600-080100'
#                ]

###############################################################################  
#        # Version 2
#        # threeway: 5 class ['00', '01', '02', '03', '04']
#        val_list = ['00_threeway_20180507_091000-091500',
#                '00_threeway_20180507_093500-094000',
#                '00_threeway_20180512_120500-121000',
#                '01_threeway_20180507_072900-073400',
#                '01_threeway_20180512_073100-073600',
#                '02_threeway_20180512_065300-065800',
#                '02_threeway_20180512_093400-093900',
#                '02_threeway_20180512_110100-110600',
#                '03_threeway_20180512_082900-083400',
#                '03_threeway_20180512_104900-105400'
#                ]
#        test_list = ['00_threeway_20180507_090000-090500',
#                '00_threeway_20180507_092500-093000',
#                '00_threeway_20180507_095000-095500',
#                '00_threeway_20180512_133000-133500',
#                '01_threeway_20180507_061200-061700',
#                '01_threeway_20180512_070600-071100',
#                '01_threeway_20180512_151900-152400',
#                '02_threeway_20180507_071500-072000',
#                '02_threeway_20180512_090900-091400',
#                '02_threeway_20180512_102200-102700',
#                '02_threeway_20180512_161600-162100',
#                '03_threeway_20180512_064600-065100',
#                '03_threeway_20180512_100700-101200',
#                '03_threeway_20180512_144500-145000',
##                '04_threeway_20180512_075600-080100'
#                ]
    
###############################################################################        
###############################################################################        
#        # multi-location: 3 class ['00', '02', '04']
#        val_list = ['00_gh1g_20170102_092000-092500',
#                '00_ss_20170418_172500-173000',
#                '00_ss2g_20170331_152000-152500',
#                '02_gh1g_20170417_131000-131500',
#                '02_gh1g_20170417_151500-152000',
#                '02_gh1g_20170417_154000-154500',
#                '02_glhgs_20170417_151500-152000',
#                '02_mm1_20170417_153600-154100',
#                '02_ss_20170417_134800-135300',
#                '02_ssl_20170331_170500-171000',
#                '02_ssl_20170405_173200-173700',
#                '02_ssl_20170417_131500-132000',
#                '02_ssl_20170417_140800-141300',
#                '02_ssl_20170417_154500-155000',
#                '04_ss_20170417_142000-142500',
#                '04_ss_20170417_151000-151500',
#                '04_ss2g_20170417_150500-151000',
#                '04_ssl_20170417_143500-144000'
#                ]
#        test_list = ['00_gh1g_20170102_091000-091500',
#                '00_gh1g_20170405_182500-183000',
#                '00_ss2g_20170331_151000-151500',
#                '02_gh1g_20170331_161900-162400',
#                '02_gh1g_20170417_134200-134700',
#                '02_gh1g_20170417_153000-153500',
#                '02_gh1g_20170417_155500-160000',
#                '02_mm1_20170417_131500-132000',
#                '02_ss_20170417_131500-132000',
#                '02_ss_20170418_170900-171400',
#                '02_ssl_20170331_172000-172500',
#                '02_ssl_20170417_130500-131000',
#                '02_ssl_20170417_134200-134700',
#                '02_ssl_20170417_153500-154000',
#                '02_wz_20170331_173000-173500',
#                '04_glhgs_20170417_153000-153500',
#                '04_ss_20170417_144900-145400',
#                '04_ss2g_20170417_144600-145100',
#                '04_ssl_20170417_135600-140100',
#                '04_wz_20170417_144500-145000'
#                ]        


#        # multi-location: 6 class ['00', '01', '02', '03', '04', '05']
#        val_list = ['00_gh1g_20170102_092000-092500',
#                '00_ss_20170418_172500-173000',
#                '00_ss2g_20170331_152000-152500',
#                '01_gh1g_20170417_140500-141000',
#                '01_glhgs_20170417_131000-131500',
#                '01_glhgs_20170417_133500-134000',
#                '01_mm1_20170417_130000-130500',
#                '01_mm1_20170417_150000-150500',
#                '01_ss_20170418_171400-171900',
#                '01_wz_20161226_132200-132700',
#                '02_gh1g_20170417_131000-131500',
#                '02_gh1g_20170417_151500-152000',
#                '02_gh1g_20170417_154000-154500',
#                '02_glhgs_20170417_151500-152000',
#                '02_mm1_20170417_153600-154100',
#                '02_ss_20170417_134800-135300',
#                '02_ssl_20170331_170500-171000',
#                '02_ssl_20170405_173200-173700',
#                '02_ssl_20170417_131500-132000',
#                '02_ssl_20170417_140800-141300',
#                '02_ssl_20170417_154500-155000',
#                '03_gh1g_20170417_142700-143200',
#                '03_glhgs_20170417_143500-144000',
#                '03_glhgs_20170417_153600-154100',
#                '03_ss_20170417_140000-140500',
#                '03_ss_20170417_151500-152000',
#                '03_ss2g_20170417_134600-135100',
#                '03_ss2g_20170417_143500-144000',
#                '03_ssl_20170405_175500-180000',
#                '03_ssl_20170417_152700-153200',
#                '04_ss_20170417_142000-142500',
#                '04_ss_20170417_151000-151500',
#                '04_ss2g_20170417_150500-151000',
#                '04_ssl_20170417_143500-144000',
#                '05_ssl_20170417_143800-144300'
#                ]
#        test_list = ['00_gh1g_20170102_091000-091500',
#                '00_gh1g_20170405_182500-183000',
#                '00_ss2g_20170331_151000-151500',
#                '01_dgl4_20170331_173700-174200',
#                '01_gh1g_20170417_141500-142000',
#                '01_glhgs_20170417_132000-132500',
#                '01_mm1_20170411_065000-065500',
#                '01_mm1_20170417_133400-133900',
#                '01_ss_20170417_131000-131500',
#                '01_ss2g_20170417_132500-133000',
#                '01_wz_20170331_174200-174700',
#                '02_gh1g_20170331_161900-162400',
#                '02_gh1g_20170417_134200-134700',
#                '02_gh1g_20170417_153000-153500',
#                '02_gh1g_20170417_155500-160000',
#                '02_mm1_20170417_131500-132000',
#                '02_ss_20170417_131500-132000',
#                '02_ss_20170418_170900-171400',
#                '02_ssl_20170331_172000-172500',
#                '02_ssl_20170417_130500-131000',
#                '02_ssl_20170417_134200-134700',
#                '02_ssl_20170417_153500-154000',
#                '02_wz_20170331_173000-173500',
#                '03_gh1g_20170417_133000-133500',
#                '03_glhgs_20170417_142500-143000',
#                '03_glhgs_20170417_150000-150500',
#                '03_mm1_20170417_143100-143600',
#                '03_ss_20170417_143000-143500',
#                '03_ss_20170417_153500-154000',
#                '03_ss2g_20170417_142500-143000',
#                '03_ssl_20170405_174600-175100',
#                '03_ssl_20170417_135000-135500',
#                '03_wz_20170417_145000-145500',
#                '04_glhgs_20170417_153000-153500',
#                '04_ss_20170417_144900-145400',
#                '04_ss2g_20170417_144600-145100',
#                '04_ssl_20170417_135600-140100',
#                '04_wz_20170417_144500-145000',
##                '05_ss_20170417_145900-150400',
##                '05_ssl_20170417_145200-145700'
#                '05_gh1g_20170417_144500-145000',
#                '05_ssl_20170417_145000-145500'
#                ]                

#        # use different day for testing
#        # multi-location: 5 class ['00', '01', '02', '03', '04']
#        val_list = ['00_gh1g_20170102_092000-092500',
#                '00_ss2g_20170331_152000-152500',
#                '01_gh1g_20170417_140500-141000',
#                '01_glhgs_20170417_131000-131500',
#                '01_glhgs_20170417_133500-134000',
#                '01_mm1_20170417_130000-130500',
#                '01_mm1_20170417_150000-150500',
#                '01_wz_20161226_132200-132700',
#                '02_gh1g_20170417_131000-131500',
#                '02_gh1g_20170417_151500-152000',
#                '02_gh1g_20170417_154000-154500',
#                '02_glhgs_20170417_151500-152000',
#                '02_mm1_20170417_153600-154100',
#                '02_ss_20170417_134800-135300',
#                '02_ssl_20170417_131500-132000',
#                '02_ssl_20170417_140800-141300',
#                '02_ssl_20170417_154500-155000',
#                '03_gh1g_20170417_142700-143200',
#                '03_glhgs_20170417_143500-144000',
#                '03_glhgs_20170417_153600-154100',
#                '03_ss_20170417_140000-140500',
#                '03_ss_20170417_151500-152000',
#                '03_ss2g_20170417_134600-135100',
#                '03_ss2g_20170417_143500-144000',
#                '03_ssl_20170417_152700-153200',
#                '04_ss_20170417_142000-142500',
#                '04_ss_20170417_151000-151500',
#                '04_ss2g_20170417_150500-151000',
#                '04_ssl_20170417_143500-144000'
#                ]
#        test_list = ['00_gh1g_20170405_182000-182500',
#                '00_gh1g_20170405_182500-183000',
#                '00_ss_20170418_172000-172500',
#                '00_ss_20170418_172500-173000',
#                '01_dgl4_20170331_173100-173600',
#                '01_dgl4_20170331_173700-174200',
#                '01_dgl4_20170331_174800-175300',
#                '01_ss_20170418_171400-171900',
#                '01_wz_20170331_174200-174700',
#                '01_wz_20170418_164300-164800',
#                '02_gh1g_20170331_161000-161500',
#                '02_gh1g_20170331_161900-162400',
#                '02_gh1g_20170405_180100-180600',
#                '02_mm1_20170405_174000-174500',
#                '02_ss_20170418_170900-171400',
#                '02_ssl_20170331_170000-170500',
#                '02_ssl_20170331_170500-171000',
#                '02_ssl_20170331_171000-171500',
#                '02_ssl_20170331_171500-172000',
#                '02_ssl_20170331_172000-172500',
#                '02_ssl_20170331_172500-173000',
#                '02_ssl_20170405_173200-173700',
#                '02_ssl_20170405_173700-174200',
#                '02_ssl_20170405_174100-174600',
#                '02_wz_20170331_173000-173500',
#                '03_mm1_20170405_174300-174800',
#                '03_mm1_20170405_174700-175200',
#                '03_ssl_20170405_174600-175100',
#                '03_ssl_20170405_175000-175500',
#                '03_ssl_20170405_175500-180000',
#                '03_wz_20170331_173500-174000',
#                '03_wz_20170418_162100-162600',
#                '03_wz_20170418_163700-164200',
#                '04_wz_20170418_162600-163100',
#                '04_wz_20170418_163100-163600'
#                ]                

        
        train = []
        val = []
        test = []
        
        with open(self.list_file) as datafile:
            for row in datafile:
                item = row.strip().split(',')
                path = os.path.join(self.root_path, item[0],item[1],item[2])
                num_frames = item[3]
                label = self.classes.index(item[1])
                
#                ##################################rainfall depth-Shifted Grouping
#                # change 01,02,03,04,05 to 01,03,05 (0,1,2,3,4 to 0,1,2)
#                #label = np.array([0,1,2,3,4])
#                #np.floor((label+shift)/group_size).astype(int)
#                group_size = 2
#                shift = 1 # 0 to group_size-1
#                label = int(np.floor((label+shift)/group_size))
#                
#                #TODO: balance the marginal class
#                ##################################
                
                video = VideoRecord([path, num_frames, label])
                
                # split train val and test set
                parts = item[2].split('-')
                video_name = parts[0]+'-'+parts[1]
                if video_name in test_list:
                    test.append(video)
                elif video_name in val_list:
                    val.append(video)
                else:
                    train.append(video)
                
#                ##################################double the set of "05" or "01"
#                if (shift == 0 and label == 2) or (shift == 1 and label == 0):
#                    if video_name in test_list:
#                        pass # do nothing
#                    elif video_name in val_list:
#                        val.append(video)
#                    else:
#                        train.append(video)
#                ##################################
        if self.train_val_test == 'test':
            video_list = test
        elif self.train_val_test == 'val':
            video_list = val
        else:
            video_list = train
        return video_list

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        # the number of continous frames
        num_con_frame = self.new_length # 'RGB':1, 'RGBDiff':6, 'flow':5
#        num_con_frame = 6 if self.modality in ['RGB', 'RGBDiff'] else 5 # 'RGB':6, 'RGBDiff':6, 'flow':5
        
        average_duration = (record.num_frames / num_con_frame) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames / num_con_frame, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets * num_con_frame + 1 # image index starts from 1

#        [original]
#        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
#        if average_duration > 0:
#            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
#        elif record.num_frames > self.num_segments:
#            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
#        else:
#            offsets = np.zeros((self.num_segments,))
#        return offsets + 1

    def _get_val_indices(self, record):
        """Select proper frames index. Try to select middle frames, not endpoints. 
        e.g.: choose 3 frames from 9 frames, [1,4,7] is better than [0,3,6] and [2,5,8].
        """
        # the number of continous frames
        num_con_frame = self.new_length

        if record.num_frames / num_con_frame > self.num_segments:
            tick = (record.num_frames / num_con_frame) / float(self.num_segments) # average_duration
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets * num_con_frame + 1

#        [original]
#        if record.num_frames > self.num_segments + self.new_length - 1:
#            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments) # average_duration
#            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
#        else:
#            offsets = np.zeros((self.num_segments,))
#        return offsets + 1

    def _get_test_indices(self, record):
        """The same with _get_val_indices()
        """
        # the number of continous frames
        num_con_frame = self.new_length
        
        tick = (record.num_frames / num_con_frame) / float(self.num_segments)
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        return offsets * num_con_frame + 1

#        [original]
#        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
#        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
#        return offsets + 1

    def __getitem__(self, index):
        """Override method in torch.utils.data.Dataset
        """
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
        return process_data, record.label

    def __len__(self):
        """Override method in torch.utils.data.Dataset
        """
        return len(self.video_list)
