#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by LiuSonghui on 2018/1/20
import pickle
import numpy as np
import cv2
import config

from scipy.ndimage.measurements import label
from collections import deque
from joblib import Parallel, delayed


class Prepocess(object):
    def __init__(self):
        self.dist_pickle = pickle.load(open(config.svc_pickle, 'rb'))
        self.svc = self.dist_pickle['svc']
        self.X_scale = self.dist_pickle['scaler']
        self.Features = config.Features
        self.y_start_stop_scale = config.y_start_stop_scale
        self.color_space = config.color_space
        self.pix_per_cell = config.pix_per_cell
        self.orient = config.orient
        self.cell_per_block = config.cell_per_block
        self.window = config.window
        self.cell_per_step = config.cell_per_block
        self.threshhold = config.threshhold
        self.history = deque(maxlen=8)
        self.hog_channel = config.hog_channel

    def find_car_boxs(self, img, y_start, y_stop, scale):
        img = np.copy(img)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img_tosearch = img[y_start:y_stop, :, :]
        # ctrans_tosearch = self.Features.get_feature_image(img_tosearch, self.color_space)
        ctrans_tosearch = self.Features.get_feature_image(img_tosearch, "YCrCb")
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                         (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        h, w = ctrans_tosearch.shape[0], ctrans_tosearch.shape[1]

        # Define blocks and steps
        nxblocks = (w // self.pix_per_cell) - self.cell_per_block + 1
        nyblocks = (h // self.pix_per_cell) - self.cell_per_block + 1
        nfeat_per_block = self.orient * self.cell_per_block ** 2

        nblocks_per_window = (self.window // self.pix_per_cell) - self.cell_per_block + 1
        nxsteps = (nxblocks - nblocks_per_window) // self.cell_per_step + 1
        nysteps = (nyblocks - nblocks_per_window) // self.cell_per_step + 1

        # Compute individual channel Hog features for the entire image
        if self.hog_channel == 'ALL':
            ch1 = ctrans_tosearch[:, :, 0]
            ch2 = ctrans_tosearch[:, :, 1]
            ch3 = ctrans_tosearch[:, :, 2]
            hog1 = self.Features.get_single_hog_features(ch1, feature_vec=False)
            hog2 = self.Features.get_single_hog_features(ch2, feature_vec=False)
            hog3 = self.Features.get_single_hog_features(ch3, feature_vec=False)
            # hogs = Parallel(n_jobs=-1)(delayed(self.Features.get_single_hog_features)(x, feature_vec=False)
            #                                    for x in [ch1, ch2, ch3])
        else:
            ch = ctrans_tosearch[:, :, self.hog_channel]
            hog = self.Features.get_single_hog_features(ch, feature_vec=False)

        # Define the car boxs
        car_bboxs = []
        for xb in range(nxsteps):
            xpos = xb * self.cell_per_step
            xleft = xpos * self.pix_per_cell
            for yb in range(nysteps):
                ypos = yb * self.cell_per_step
                ytop = ypos * self.pix_per_cell
                # Extract Hog for this path
                if self.hog_channel == 'ALL':
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    # hog_features = [_hog[ypos:ypos+nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    #                 for _hog in hogs]
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                else:
                    hog_features = hog[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    # hog_features = np.hstack((hog_feat, hog_feat, hog_feat))
                    # hog_features = np.hstack((hog_features, hog_features, hog_features))

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop+self.window, xleft:xleft+self.window], (64, 64))

                # Get color features
                spatial_features = self.Features.bin_spatial(subimg)
                hist_features = self.Features.color_hist(subimg)

                # Scale features and make a prediction
                test_features = self.X_scale.transform(
                    np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
                )
                test_prediction = self.svc.predict(test_features)

                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(self.window * scale)

                    # Append the box to car_bboxs
                    car_bboxs.append(((xbox_left, ytop_draw + y_start),
                                      (xbox_left + win_draw, ytop_draw + win_draw + y_start)))

        return car_bboxs

    def apply_sliding_window(self, image):
        bboxes = []
        all_boxes = Parallel(n_jobs=-1, pre_dispatch='10*n_jobs')(delayed(self.find_car_boxs, check_pickle=False)(
            image, y_start, y_stop, scale) for y_start, y_stop, scale in self.y_start_stop_scale)
        for box in all_boxes:
            bboxes.extend(box)
        return bboxes

    def get_heatmap(self, image):
        """
        Create the heatmap
        """
        heatmap = np.zeros_like(image[:, :, 0]).astype(np.float)
        for box in self.apply_sliding_window(image):
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" taks the form ((x1, y1), (x2, y2))
            heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Zero out pixels below the threshhold
        heatmap[heatmap <= self.threshhold] = 0
        current_heatmap = np.clip(heatmap, 0, 255)
        self.history.append(current_heatmap)
        heat = np.zeros_like(current_heatmap).astype(np.float32)
        for _ in self.history:
            heat += _
        return heat

    def draw_labeled_bboxs(self, image):
        """
        Draw the car box from the heatmap labels
        """
        lables = label(self.get_heatmap(image))
        for car_number in range(1, lables[1] + 1):
            # Find pixels with eatch car_number label value
            nonzero = (lables[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box base on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(image, bbox[0], bbox[1], (0, 0, 255), 6)

        return image

    def draw_heatmap(self, image):
        """
        Use cv2.applyColorMap create heatmap image
        """
        heatmap = self.get_heatmap(image)
        return cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)


if __name__ == '__main__':
    pass
