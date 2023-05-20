#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import os
import random
import math
import datetime as dt
from collections import deque
import pandas as pd

import matplotlib.pyplot as plt


from numpy.lib.scimath import sqrt # used for hoF
from numpy import arctan2 # used for hoF

from scipy import pi, cos, sin # used for HoF
from scipy.ndimage import uniform_filter # used for hoF

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint,CSVLogger
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
print("Tensorflow version: ", tf.__version__)
print(tf.test.gpu_device_name())

import torch
# for repnet
# tensowflow==1.15
# python=3.5,this is problem.i can't success in 3.9
import sys
#print(sys.path)
#sys.path.insert(1, 'C:\\Users\\Grace\\MTech Jupyter\\Intelligent Sensing Systems\\PracticeMod\\FineDiving\\itsspm_fenglei\\models')
#print(sys.path)
from models.repnet import get_repnet_model
from models.repnet import video_transform_image
from models.repnet import slow_video_generate
from models.repnet import get_counts
from models.repnet import read_video
tf.compat.v1.enable_v2_behavior()


# In[ ]:


#Global Variables
dive_action_labels  = ['Entry', 'Flight', 'Takeoff']
temp_segment_model = None
autoscore_model = None
ss_twist_classifier_model = None
somersault_model = None
twist_model = None
angle_of_entry_model = None
splash_model = None
linear_regression_model = None
folderpath      = 'modelcheckpoints/'


#set parameter
# FPS while recording video from webcam.
WEBCAM_FPS = 16#@param {type:"integer"}

# Time in seconds to record video on webcam. 
RECORDING_TIME_IN_SECONDS = 8. #@param {type:"number"}

# Threshold to consider periodicity in entire video.
THRESHOLD = 0.001#@param {type:"number"}

# Threshold to consider periodicity for individual frames in video.
WITHIN_PERIOD_THRESHOLD = 0.005#@param {type:"number"}

# Use this setting for better results when it is 
# known action is repeating at constant speed.
CONSTANT_SPEED = False#@param {type:"boolean"}

# Use median filtering in time to ignore noisy frames.
MEDIAN_FILTER = True#@param {type:"boolean"}

# Use this setting for better results when it is 
# known the entire video is periodic/reapeating and
# has no aperiodic frames.
FULLY_PERIODIC = True#@param {type:"boolean"}

# Plot score in visualization video.
PLOT_SCORE = False#@param {type:"boolean"}

# Visualization video's FPS.
VIZ_FPS = 30#@param {type:"integer"}


def load_somersault_model():
    print('loading somersault model')
    global somersault_model
    PATH_TO_CKPT = folderpath+'repnet_ckpt'
    model = get_repnet_model(PATH_TO_CKPT)
    somersault_model=model

def load_twist_model():
    print('loading twist model')
    global twist_model, somersault_model
    twist_model=somersault_model
   
def ensureDirectoryClean(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    else:
        for f in os.listdir(dirpath):
            os.remove(os.path.join(dirpath, f))

def extractFolderAndFileNameFromAbsPath(absFilePath):
    filename_sep = absFilePath.rindex('\\')
    extension_sep = absFilePath.rindex(".")
    folder = absFilePath[0: filename_sep]
    shortfilename = absFilePath[filename_sep+1:extension_sep]
    ext = absFilePath[extension_sep+1:len(absFilePath)]
    return folder, shortfilename, ext

def extractEventNoAndDiveNo(folderPath):
    tokens = folderPath.split("\\")
    diveno = tokens[len(tokens)-1]
    eventno = tokens[len(tokens)-2]
    return eventno, diveno


def createVideo(image_folder, video_folder, divephase, vidname, resizeFrame=False, resizeFrameDim=[64,64]):
    images = []
    #folders = [image_folder+"\\Ntakeoff", image_folder+"\\Nflight", image_folder+"\\Nentry"]
    subfolder_images = sorted(os.listdir(image_folder))
    for subfolder_image in subfolder_images:
        if subfolder_image.endswith(".jpg"):
            images.append(image_folder+"\\"+subfolder_image)
    if (len(images)==0):
        return
    
    frame = cv2.imread(images[0])

    height, width, layers = frame.shape
    if (resizeFrame == True):
        height = resizeFrameDim[1]
        width = resizeFrameDim[0]
    vidFullName = video_folder+'\\'+vidname+"_"+divephase+".mp4"
    print('writing video to ', vidFullName , ' framewidth ', width, ' frameheight ', height)
    fps = 25
    video = cv2.VideoWriter(vidFullName, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), fps, (width,height))

    
    for image in images:
        frame = cv2.imread(image)
        if (resizeFrame == True):
            frame = cv2.resize(frame, (height, width))
        video.write(frame)

    cv2.destroyAllWindows()
    video.release()
    return vidFullName

def predict_num_somersaults(ssVidPath, out_dir):
    ensureDirectoryClean(out_dir)
    print('predict num somersaults ', ssVidPath, out_dir)
    video_transform_image(ssVidPath, out_dir) # when merge, must use imgfolder format
    print('predict num somersaults video transform image done')
    output_video=slow_video_generate(ssVidPath, out_dir)
    print('predict num somersaults slow_video_generate done')
    imgs, vid_fps = read_video(output_video)
    print('Running RepNet...') 
    (pred_period, pred_score, within_period,
    per_frame_counts, chosen_stride) = get_counts(
        somersault_model,
        imgs,
        strides=[1,2,3,4],
        batch_size=20,
        threshold=THRESHOLD,
        within_period_threshold=WITHIN_PERIOD_THRESHOLD,
        constant_speed=CONSTANT_SPEED,
        median_filter=MEDIAN_FILTER,
        fully_periodic=FULLY_PERIODIC)
    frames=imgs
    count=per_frame_counts
    if isinstance(count, list):
        counts = len(frames) * [count/len(frames)]
    else:
        counts = count
    sum_counts = np.cumsum(counts)
    num= round(float(sum_counts[-1]),2)
    unique_somes = [1.5, 2.0, 2.5, 3.5, 4.5, 3.0]
    

    modify_num= min(unique_somes, key=lambda x: abs(x - num))
   
    return num,modify_num

def predict_num_twists(twVidPath, out_dir):
    ensureDirectoryClean(out_dir)
    video_transform_image(twVidPath, out_dir) # when merge, must use imgfolder format
    output_video=slow_video_generate(twVidPath, out_dir)
    imgs, vid_fps = read_video(output_video)
    print('Running RepNet...') 
    (pred_period, pred_score, within_period,
    per_frame_counts, chosen_stride) = get_counts(
        twist_model,
        imgs,
        strides=[1,2,3,4],
        batch_size=20,
        threshold=THRESHOLD,
        within_period_threshold=WITHIN_PERIOD_THRESHOLD,
        constant_speed=CONSTANT_SPEED,
        median_filter=MEDIAN_FILTER,
        fully_periodic=FULLY_PERIODIC)
    frames=imgs
    count=per_frame_counts
    if isinstance(count, list):
        counts = len(frames) * [count/len(frames)]
    else:
        counts = count
    sum_counts = np.cumsum(counts)
    num= round(float(sum_counts[-1]),2)
    unique_twist = [0.5, 1.5, 2.0, 3.0, 1.0, 2.5, 3.5]
    

    modify_num= min(unique_twist, key=lambda x: abs(x - num))
    return num,modify_num


def processVideo(vidpath):
    print('processing ', vidpath)
    folder, shortfilename, _ = extractFolderAndFileNameFromAbsPath(vidpath)
    numSomersaults = predict_num_somersaults(".\\images\\"+shortfilename+"\\Flight"+shortfilename+"_Flight_ss.mp4",
                                            ".\\images\\"+shortfilename+"\\repnet_tmp")
    numTwists = predict_num_twists(".\\images\\"+shortfilename+"\\Flight"+shortfilename+"_Flight_tw.mp4",
                                   ".\\images\\"+shortfilename+"\\repnet_tmp")
    print(
          'numSomersaults: ', numSomersaults, 
          ', numTwists: ', numTwists)
    return numSomersaults, numTwists


def main():
    load_somersault_model()
    load_twist_model()

main()


import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from werkzeug.wrappers import Request, Response
import json

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = '.\\uploads\\'

@app.route('/sstwist', methods=['POST'])
def predict_ss_twist():
    if 'videoname' not in request.form:
        return {"error" : "no videoname in request"}
    file = request.form['videoname']
    print('file : ', file)
    numSomersaults, numTwists = processVideo(app.config['UPLOAD_FOLDER']+file)
    
    result = {
        "file" : file,
        "numSomersaults" : str(numSomersaults),
        "numTwists" : str(numTwists)
    }
    result_json = json.dumps(result)
    print('result', result_json)
    return result_json

if __name__ == "__main__":
    from werkzeug.serving import run_simple
    run_simple('localhost', 5001, app)
    