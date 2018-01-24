#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by LiuSonghui on 2018/1/20
import glob
import config
from train import Train
from preprocess import Prepocess
from moviepy.editor import VideoFileClip


def train_classify():
    car_image_path = 'vehicles/*/*.png'
    notcar_image_path = 'non-vehicles/*/*.png'
    cars = glob.glob(car_image_path)
    notcars = glob.glob(notcar_image_path)

    feature_train = Train(cars, notcars)
    feature_train.train()


def main():
    if config.is_train == True:
        train_classify()

    preprocess = Prepocess().draw_labeled_bboxs
    project_output = 'test_video_output.mp4'
    # project_output = 'project_video_output.mp4'
    clip = VideoFileClip('test_video.mp4')
    # clip = VideoFileClip('project_video.mp4')
    project_clip = clip.fl_image(preprocess)
    project_clip.write_videofile(project_output, audio=False)


if __name__ == '__main__':
    main()
