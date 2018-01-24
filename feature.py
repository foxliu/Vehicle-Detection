#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by LiuSonghui on 2018/1/20
import numpy as np
import cv2
from skimage.feature import hog
from joblib import Parallel, delayed


class Features(object):
    def __init__(self,
                 color_space='RGB',
                 spatial_size=(32, 32),
                 hist_bins=32,
                 orient=9,
                 pix_per_cell=8,
                 cell_per_block=2,
                 hog_channel=0,
                 spatial_feat=True,
                 hist_feat=True,
                 hog_feat=True,
                 hist_range=(0, 256)):
        self.color_space = color_space
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.orient = orient
        self.pix_per_cell = (pix_per_cell, pix_per_cell)
        self.cell_per_block = (cell_per_block, cell_per_block)
        self.hog_channel = hog_channel
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat
        self.hist_range = hist_range

    @staticmethod
    def get_feature_image(img, color_space='RGB'):
        if color_space != 'RGB':
            return cv2.cvtColor(img, eval('cv2.COLOR_RGB2{}'.format(color_space)))
        else:
            return np.copy(img)

    def get_single_hog_features(self, img, vis=False, feature_vec=True):
        # Cell with two outputs if vis == True
        if vis == True:
            features, hog_image = hog(img,
                                      orientations=self.orient,
                                      pixels_per_cell=self.pix_per_cell,
                                      cells_per_block=self.cell_per_block,
                                      transform_sqrt=True,
                                      visualise=True,
                                      feature_vector=feature_vec,
                                      block_norm='L2')
            return features, hog_image
        # Otherwise call with one output
        else:
            features = hog(img,
                           orientations=self.orient,
                           pixels_per_cell=self.pix_per_cell,
                           cells_per_block=self.cell_per_block,
                           transform_sqrt=True,
                           visualise=False,
                           feature_vector=feature_vec,
                           block_norm='L2')
        return features

    def get_hog_features(self, img):
        if self.hog_channel == 'ALL':
            hog_features = []
            for channel in range(img.shape[2]):
                hog_features.append(self.get_single_hog_features(img[:, :, channel], vis=False))
            # hog_features = Parallel(n_jobs=-1)(delayed(self.get_single_hog_features)(img[:,:,x], vis=False)
            #                                    for x in range(img.shape[2]))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = self.get_single_hog_features(img[:, :, self.hog_channel], vis=False)
        return hog_features

    # Define a function to compute binned color features
    def bin_spatial(self, img):
        c1 = cv2.resize(img[:, :, 0], self.spatial_size).ravel()
        c2 = cv2.resize(img[:, :, 1], self.spatial_size).ravel()
        c3 = cv2.resize(img[:, :, 2], self.spatial_size).ravel()
        return np.hstack((c1, c2, c3))

    # Define a function to compute color histogram features
    # Need to change bins_range if reading .png files whit mpimg
    @staticmethod
    def color_hist(img, nbins=32, bins_range=(0, 256)):
        # Compute the histogram of the color channels sparately
        c1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
        c2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
        c3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
        # hists = Parallel(n_jobs=-1)(delayed(np.histogram)(x, bins=nbins, range=bins_range)
        #                                     for x in [img[:,:,0], img[:,:,1], img[:,:2]])
        # Concatenate the histograms into a single feature vector
        hist_features = np.concatenate((c1_hist[0], c2_hist[0], c3_hist[0]))
        # hist_features = np.concatenate([i[0] for i in hists])
        # Return the individual histograms, bin_centers and feature vector
        return hist_features

    # Define a function to extarct features
    def extract_feature(self, img):
        file_features = []
        feature_image = self.get_feature_image(img, self.color_space)
        if self.spatial_feat == True:
            file_features.append(self.bin_spatial(feature_image))
        if self.hist_feat == True:
            file_features.append(self.color_hist(feature_image, bins_range=self.hist_range))
        if self.hog_feat == True:
            file_features.append(self.get_hog_features(feature_image))

        return np.concatenate(file_features)
