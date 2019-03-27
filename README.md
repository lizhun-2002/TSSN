
# [Temporal and Spatial Segment Networks](https://github.com/lizhun-2002/TSSN)

This is an implementation of Temporal and Spatial Segment Networks(TSSN) on Python 3 and PyTorch. 
TSSN is proposed in my thesis entitled Rainfall Depth Recognition from Road Surveillance Videos Using Deep Learning.
This architecture is based on Temporal Segment Networks(TSN) and the code is also based on the PyTorch implementation of [TSN](https://github.com/yjxiong/tsn-pytorch).

## Requirements
Python 3.6 and PyTorch 0.4.0

## Data
We collected new video data sets and proposed an estimation procedure to calculate refined rainfall depth from the original meteorological data. The data set is too large to upload. We will consider sharing the data set in other ways.

## Train
```bash
python main.py kaist 3 RGB ./data_6frames/ data_file_1-3-5.csv \
   --arch BNInception --num_segments 3 --num_spacial_segments 4 \
   --gd 20 --lr 0.001 --lr_steps 190 300 --epochs 340 \
   -b 12 -j 8 --dropout 0.8 \
   --snapshot_pref threeway_1-3-5_mix_2x2tssn
```

## Test
```bash
python test_models.py kaist 3 RGB ./data_6frames/ data_file_1-3-5.csv \
    threeway_1-3-5_mix_2x2tssn_rgb_model_best.pth.tar --num_spacial_segments 4 \
   --arch BNInception --save_scores score_file_threeway_1-3-5_mix_2x2tssn_rgb_seg3_epoch340_tseg4 --test_segments 4
```

## Fuse the scores
```bash
python eval_scores.py \
    score_file_threeway_1-3-5_mix_2x2tssn_rgb_seg3_epoch340_tseg4.npz \
    score_file_threeway_1-3-5_mix_2x2tssn_rgbdiff_seg3_epoch340_tseg4.npz \
    --score_weights 1 1 \
    --grid True
```
