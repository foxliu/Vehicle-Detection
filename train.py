#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by LiuSonghui on 2018/1/20
import pickle
import cv2
import numpy as np

import config

from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Train(object):
    def __init__(self, cars, notcars):
        self.cars = cars
        self.notcars = notcars
        self.svc_pickle = config.svc_pickle
        self.Features = config.Features

    def get_features(self, file_list):
        file_features = []
        for img_file in file_list:
            # Image read and convert to RGB
            image = cv2.imread(img_file)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Append the extract features to the file features
            file_features.append(self.Features.extract_feature(image))
        return file_features

    def train(self):
        car_features = self.get_features(self.cars)
        notcar_features = self.get_features(self.notcars)
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        X_scaler = StandardScaler().fit(X)
        scaled_x = X_scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test set
        X_train, X_test, y_train, y_test = train_test_split(scaled_x, y, test_size=.2,
                                                            random_state=np.random.randint(0, 100))
        svc = LinearSVC()
        svc.fit(X_train, y_train)
        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

        dist_pickle = {}
        dist_pickle["svc"] = svc
        dist_pickle["scaler"] = X_scaler
        pickle.dump(dist_pickle, open(self.svc_pickle, 'wb'))


if __name__ == '__main__':
    pass
