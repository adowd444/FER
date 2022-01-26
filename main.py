import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from skimage import io, transform

from tensorflow.keras.optimizers import SGD
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.python.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.lib.io import file_io
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

#MUST PREPROCESS DATASET IMAGES BEFORE RUNNING CODE
EPOCHS = 50
BS = 128
DROPOUT_RATE = 0.35
SGD_LEARNING_RATE = 0.01
SGD_DECAY = 0.0001


def get_datagen(dataset):
    datagen = ImageDataGenerator(rescale=1./255,
                                 featurewise_center=False,
                                 featurewise_std_normalization=False,
                                 rotation_range=10,
                                 width_shift_range=0.1,
                                 height_shift_range=0.1,
                                 zoom_range=0.1,
                                 horizontal_flip=True)

    return datagen.flow_from_directory(dataset,
                                       target_size=(48,48),
                                       color_mode='grayscale',
                                       shuffle=True,
                                       class_mode='categorical',
                                       batch_size=BS)


X_train_gen = get_datagen('/train')
X_dev_gen = get_datagen('/dev')

X_dev = np.zeros((len(X_dev_gen.filepaths),48,48,1))
Y_dev = np.zeros((len(X_dev_gen.filepaths), 7))
for i in range(0,len(X_dev_gen.filepaths)):
    x = io.imread(X_dev_gen.filepaths[i], as_gray=True)
    X_dev[i,:] = transform.resize(x, (48,48,1))
    Y_dev[i,X_dev_gen.classes[i]] = 1

model = load_model('/fer-master/models/webcam-SGD_LR_0.01000-EPOCHS_100-BS_128-DROPOUT_0.35test_acc_0.698.h5')

history = model.fit_generator(generator=X_train_gen,
                              validation_data=(X_dev,Y_dev),
                              shuffle=True,
                              callbacks=[rlrop],
                              epochs=EPOCHS)



# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'dev'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'dev'], loc='upper left')
plt.show()


results_webcam = model.evaluate(X_dev, Y_dev, batch_size=BS)

lr_str = '-SGD_LR_%.5f' % SGD_LEARNING_RATE
epoch_str = '-EPOCHS_' + str(EPOCHS)
bs_str = '-BS_' + str(BS)
dropout_str = '-DROPOUT_' + str(DROPOUT_RATE)
test_acc = 'test_acc_%.3f' % results_webcam[1]
filename = '/fer-master/webapp/public/models' + lr_str + epoch_str + bs_str + dropout_str + test_acc + '.h5'
print(f'Saving model to {filename}...')
model.save(filename)