import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from os import mkdir

# git clone https://github.com/ifp-uiuc/do-neural-networks-learn-faus-iccvw-2015.git
# rm -rf ck+ && mkdir ck+
# unzip -q '/content/drive/My Drive/cs230 project/dataset/ck+/extended-cohn-kanade-images.zip' -d /content/ck+
# unzip -q '/content/drive/My Drive/cs230 project/dataset/ck+/FACS_labels.zip' -d /content/ck+
# unzip -q '/content/drive/My Drive/cs230 project/dataset/ck+/Emotion_labels.zip' -d /content/ck+
# mv '/content/ck+/Emotion'{,_labels}

# These folders give some processing issue, so we delete them
# rm -rf ck+/cohn-kanade-images/S5*

# sed -i 's/96/48/g' do-neural-networks-learn-faus-iccvw-2015/data_scripts/make_ck_plus_dataset.py
# rm -rf ck-output && mkdir ck-output
# python2 do-neural-networks-learn-faus-iccvw-2015/data_scripts/make_ck_plus_dataset.py --input_path /content/ck+/ --save_path /content/ck-output

def reindex_labels(y8):
    y = np.zeros(y8.shape, dtype=np.int8)
    label_mapping = {0:6, 2:-1, 1:0, 3:1, 4:2, 5:3, 6:4, 7:5}
    for i in range(0,len(y8)):
        y[i] = label_mapping[y8[i]]

    return y

y8[y8==2]

X = np.load('/content/ck-output/npy_files/X.npy')
y8 = np.load('/content/ck-output/npy_files/y.npy')
y = reindex_labels(y8)
y[y==6]

emotions = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'sad', 5:'surprise', 6:'neutral'}

for i in range(0,10):
  plt.xlabel(emotions[y[i]])
  plt.imshow(X[i].reshape(48,48),cmap='gray')
  plt.figure()
  break

# rm -rf ck-images && mkdir ck-images
for emotion in emotions:
    mkdir(f'/content/ck-images/' + f'{emotion} ' + f'{emotions[emotion]}')

count = 0
for i in range(0,X.shape[0]):
    if y[i] == -1:
      continue
    count_string = str(count).zfill(7)
    fname = '/content/ck-images/' + f'{y[i]} ' + f'{emotions[y[i]]}/' + f'{emotions[y[i]]}-{count_string}.png'
    img = Image.fromarray(X[i].reshape(48,48))
    img.save(fname)
    count += 1

# cd ck-images && zip -r ck-plus.zip *
# mv ck-images/ck-plus.zip '/content/drive/My Drive/cs230 project/dataset/'