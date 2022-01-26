import pandas as pd
import numpy as np
import csv
from PIL import Image
from os import mkdir

df = pd.read_csv('/fer-master/datasets/fer2013.csv')
df.head()

emotions = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'sad', 5:'surprise', 6:'neutral'}

#mkdir train test-public test-private
for root in ("train", "test-public", "test-private"):
  for emotion in emotions:
    mkdir(f'{root}/' + f'{emotion} ' + f'{emotions[emotion]}')

count = 0
for emotion, image_pixels, usage in zip(df['emotion'], df['pixels'], df['Usage']):
    image_string = image_pixels.split(' ')  # pixels are separated by spaces
    image_data = np.asarray(image_string, dtype=np.uint8).reshape(48, 48)
    img = Image.fromarray(image_data)  # final image\
    count_string = str(count).zfill(6)

    path = ''
    if usage == 'Training':
        path = 'train/'
    elif usage == 'PublicTest':
        path = 'test-public/'
    elif usage == 'PrivateTest':
        path = 'test-private/'
    else:
        print("Exception!")

    # train/2 fear/fear-000001.png
    img.save(path + f'{emotion} ' + f'{emotions[emotion]}/' + f'{emotions[emotion]}-{count_string}.png')
    count += 1


#cp {test-private,test-public,train}/*.zip '/fer-master//datasets'