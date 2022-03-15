import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
from PIL import Image

# rm -rf affectnet && mkdir affectnet

path = '/Users/ashley/Desktop/Research/affectnet/train_set/annotations/'

# annotations key: arousal: aro, valence: val, facial landmarks:lnd, expression:exp

aro=[]
for i in range(5):
    try:
        value = np.load(path+str(i)+'_aro.npy')
        aro.append(value)
        i+=1

    except FileNotFoundError:
        pass

val=[]
for i in range(5):
    try:
        value = np.load(path+str(i)+'_val.npy')
        val.append(value)
        i+=1

    except FileNotFoundError:
        pass

fl=[]
pixels = []
for i in range(5):
    try:
        value = np.load(path+str(i)+'_lnd.npy')
        fl.append(value)
        i+=1

        im = Image.open(i+'.jpg')
        pix_val = list(im.getdata())
        pix_val_flat = [x for sets in pix_val for x in sets]
        pixels.append(pix_val_flat)

    except FileNotFoundError:
        pass

file_num = []
exp=[]
for i in range(5):
    try:
        value = np.load(path+str(i)+'_exp.npy')
        exp.append(value)
        file_num.append(i)
        i+=1

    except FileNotFoundError:
        pass

print(file_num)