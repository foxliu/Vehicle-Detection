#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by LiuSonghui on 2018/1/20
from feature import Features

color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
hist_range = (0, 256)  # histogram bins range
y_start_stop_scale = [
    [350, 500, 1.0],
    [350, 500, 1.3],
    [360, 556, 1.4],
    [370, 556, 1.6],
    [380, 556, 1.8],
    [380, 556, 2.0],
    [390, 556, 1.9],
    [350, 556, 1.3],
    [350, 556, 2.2],
    [450, 656, 3.0]
]  # Min and max in y to search and the image scale
svc_pickle = 'svc_pickle.p'  # Store the svc

is_train = False  # If now running start with

window = 64
cells_per_step = 2  # define how many cells to step
threshhold = 3  # Define the heatmap thershhold

Features = Features(color_space=color_space,
                    orient=orient,
                    pix_per_cell=pix_per_cell,
                    cell_per_block=cell_per_block,
                    hog_channel=hog_channel,
                    spatial_size=spatial_size,
                    hist_bins=hist_bins,
                    spatial_feat=spatial_feat,
                    hist_feat=hist_feat,
                    hog_feat=hog_feat,
                    hist_range=hist_range)
