from moviepy.editor import VideoFileClip
import numpy as np
import pickle
import cv2
import time
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from keras.models import load_model, Model
from keras.callbacks import  EarlyStopping, ModelCheckpoint, LambdaCallback
import keras.optimizers
import glob
import sys, getopt
import h5py
import skimage
from keras.preprocessing.image import ImageDataGenerator
from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
import auth
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image


IMG_SIZE = (227,227)
IMG_SHAPE = IMG_SIZE + (3,)
from skimage import exposure

from tqdm import tqdm

def resize_all(imgs):
  new_shape = (imgs.shape[0],) + IMG_SIZE[::-1] + (imgs.shape[3],)
  if new_shape == imgs.shape:
    return imgs
  ret_imgs = np.zeros(new_shape)
  for i in range(len(imgs)):
    ret_imgs[i] = cv2.resize(imgs[i], IMG_SIZE[::-1])
  return ret_imgs

def preprocess(img):
  # img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  # img = img[:, :, 2]
  img = cv2.resize(img, IMG_SIZE[::-1])

  img = np.expand_dims(img, axis=0)
  return preprocess_input(img.astype(np.float32))

def crop_leftbottom(img, left, bottom, size_x, size_y, adjx=0, adjy=0):
  startx = left + adjx
  starty = img.shape[0] - (bottom + adjy)
  startx = min(max(0, startx), img.shape[1])
  starty = min(max(0, starty), img.shape[0])

  endy = min(max(0, starty - size_y), img.shape[0])

  endx = min(max(0, startx + size_x), img.shape[1])
  return img[endy : starty, startx: endx]

def crop_center(img, cropx, cropy, adjx=0, adjy=0):
  y, x, _ = img.shape
  startx = x // 2 - (cropx // 2) + adjx
  starty = y // 2 - (cropy // 2) + adjy
  startx = min(max(0, startx), img.shape[1])
  starty = min(max(0, starty), img.shape[0])
  return img[starty:starty + cropy, startx:startx + cropx]


# Old data processor now unused.
# class Processor:
#   current_label = ''
#   curr_count = 0
#   x_dataset = y_dataset = None
#   h5f = None
#   total_count = 0
#
#   def __init__(self, max=300000):
#     self.h5f = h5py.File('train.h5', 'w', libver='latest')
#     size = (max,) + IMG_SIZE + (3,)
#
#     self.x_dataset = self.h5f.create_dataset('X', size)
#     self.y_dataset = self.h5f.create_dataset('Y', (max,), dtype='S10')
#
#   def set_next_label(self, label):
#     print('{} count: {}'.format(self.current_label, self.curr_count))
#     self.current_label = label
#     self.curr_count = 0
#
#   def processImage(self, mimg):
#     # Generate shit loads of different data
#     min_crop_x = 400
#     min_crop_y = 300
#
#     ar = min_crop_y / min_crop_x
#     step_x = 200
#     cropx = 800
#     cropy = 800
#     adj = [ (-100,-50), (-100,0), (0,0), (100,0), (100,50)]
#     #while (cropx > min_crop_x  and cropy > min_crop_y):
#     for iadj in adj:
#       img = crop_center(mimg, 1000, 1000, iadj[0], iadj[1])
#       # img = crop_leftbottom(mimg, 0, 0, ,900, iadj[0], iadj[1])
#       img = preprocess(img)
#       self.curr_count += 1
#       # plt.imshow(img)
#       # plt.show()
#       self.x_dataset[self.total_count] = img
#       self.y_dataset[self.total_count] = np.string_(self.current_label)
#       self.total_count += 1
#       # cropx -= int(step_x)
#       # cropy -= int(step_x * ar)
#
#     return img
#
#   def write(self):
#     self.x_dataset.attrs['size'] = self.total_count
#     self.y_dataset.attrs['size'] = self.total_count
#     # self.x_dataset.resize(final_size)
#     # self.y_dataset.resize(self.total_count)
#     self.h5f.close()


def detect_pattern(compressed, pattern, threshold = 3):
  for i in range(len(compressed) - len(pattern) + 1):
    match = True
    for j in range(len(pattern)):
      curr_match = compressed[i+j][0] == pattern[j] and compressed[i+j][1] >= threshold
      match = match and curr_match
    if match:
      return match
  return False

def compress(classifications):
  compressed = []
  current_class = ""
  last_counter = 0
  for classif in classifications:
    if classif == current_class:
      last_counter += 1
    else:
      if last_counter != 0:
        compressed.append((current_class, last_counter))
      current_class = classif
      last_counter = 1
  compressed.append((current_class, last_counter))
  return compressed

def gen_preprocess(x):
  x = np.expand_dims(x, axis=0)
  return preprocess_input(x)

def get_model(train=True):

  if Path('model.h5').is_file():
    return load_model('model.h5')

  datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=False,
    preprocessing_function=gen_preprocess,
    fill_mode='nearest')

  data_generator = datagen.flow_from_directory(
    directory='train_data/',
    target_size=IMG_SIZE,
    class_mode='categorical')
  print(data_generator.classes)

  validgen = ImageDataGenerator(preprocessing_function=gen_preprocess)
  valid_generator = validgen.flow_from_directory(
    directory='valid_data/',
    target_size=IMG_SIZE,
    class_mode='categorical',
    shuffle=False
  )

  test_generator = validgen.flow_from_directory(
    directory='test_data/',
    target_size=IMG_SIZE,
    class_mode='categorical',
    shuffle=False
  )

  model = SqueezeNet()
  print(model.summary())
  x = Convolution2D(4, (1, 1), padding='same', name='conv11')(model.layers[-5].output)
  x = Activation('relu', name='relu_conv11')(x)
  x = GlobalAveragePooling2D()(x)
  x = Activation('softmax')(x)
  # x= Dense(4, activation='softmax')(x)
  # x = Dense(4, activation='softmax')(model.layers[-2].output)
  model = Model(model.inputs, x)
  print(model.summary())

  # Following is the original model I was training
  # model = Sequential()
  #
  # model.add(Convolution2D(16, 3, 3,
  #                         border_mode='same',
  #                         input_shape=IMG_SHAPE))
  # model.add(MaxPooling2D(pool_size=(3, 3)))
  # model.add(Activation('relu'))
  # model.add(Dropout(0.2))
  #
  # model.add(Convolution2D(32, 3, 3,
  #                         border_mode='same'))
  # model.add(MaxPooling2D(pool_size=(3, 3)))
  # model.add(Activation('relu'))
  # model.add(Dropout(0.2))
  #
  # model.add(Convolution2D(48, 3, 3,
  #                         border_mode='same'))
  # model.add(MaxPooling2D(pool_size=(2, 2)))
  # model.add(Activation('relu'))
  # model.add(Dropout(0.2))
  # #
  # model.add(Convolution2D(64, 3, 3,
  #                         border_mode='same'))
  # model.add(MaxPooling2D(pool_size=(2, 2)))
  # model.add(Activation('relu'))
  # model.add(Dropout(0.2))
  # #
  # model.add(Convolution2D(64, 3, 3,
  #                         border_mode='same'))
  # model.add(MaxPooling2D(pool_size=(2, 2)))
  # model.add(Activation('relu'))
  # model.add(Dropout(0.2))
  #
  # # 1st Layer - Add a flatten layer
  # model.add(Flatten())
  #
  # model.add(Dense(1164))
  # model.add(Activation('relu'))
  # model.add(Dropout(0.2))
  #
  # model.add(Dense(128))
  # model.add(Activation('tanh'))
  # model.add(Dropout(0.2))
  #
  # # 2nd Layer - Add a fully connected layer
  # model.add(Dense(50))
  # model.add(Activation('relu'))
  # model.add(Dropout(0.2))
  #
  # model.add(Dense(10))
  # model.add(Activation('relu'))
  # model.add(Dropout(0.2))
  #
  # # 4th Layer - Add a fully connected layer
  # model.add(Dense(4))
  # # 5th Layer - Add a ReLU activation layer
  # model.add(Activation('softmax'))
  # TODO: Build a Multi-layer feedforward neural network with Keras here.
  # TODO: Compile and train the model
  filepath = "weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
  callbacks = [
    EarlyStopping(monitor='loss', min_delta=0.01, patience=2, verbose=1),
    LambdaCallback(on_epoch_end=lambda batch,logs: evaluate_model(model, test_generator)),
    ModelCheckpoint(filepath=filepath, monitor='loss', save_best_only=True, verbose=1),
  ]

  model.compile(keras.optimizers.Adam(lr=0.0001), 'categorical_crossentropy', ['accuracy'])
  model.fit_generator(data_generator, steps_per_epoch=400, epochs=30, verbose=1, callbacks=callbacks)
  evaluate_model(model, test_generator)

  model.save('model.h5', True)

  return model

def get_brightness(img):
  hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  return np.average(hsv[:,:,2])

def evaluate_model(model, generator):
  from sklearn.metrics import accuracy_score
  steps = 0
  scores = []
  for i in generator:
    y_pred = model.predict(x=i[0])
    y_pred = y_pred > 0.6
    scores.append(accuracy_score(i[1], y_pred))
    steps += 1
    if (steps > 29):
      break
  print("Model score is: {}".format(sum(scores)/len(scores)))

def Xgen(all_images, all_labels, batch_size, total):
  while True:
    images = np.zeros((batch_size,) + IMG_SHAPE)
    y_one_hot = np.zeros((batch_size,) + (all_labels.shape[1],))
    for i in range(batch_size):
      randindex = np.random.randint(0, total)
      images[i] = all_images[randindex]
      y_one_hot[i] = all_labels[randindex]

    yield (images, y_one_hot)

class Tester:
  model = None
  def __init__(self, model):
    self.model = model
  def sliding_window(self, img):
    window_size = (400, 300)

    j_iter = int(img.shape[0] / window_size[1])
    i_iter = int(img.shape[1] / window_size[0])
    found_colors = []
    for j in range(j_iter):
      for i in range(i_iter):
        new_img = img[j * window_size[1] : j * window_size[1] + window_size[1],
                  i * window_size[0] : i * window_size[0] + window_size[0]]
        new_img = np.array(preprocess(new_img))
        plt.imshow(new_img)
        plt.show()

        color = self.predict_color(new_img)
        if (color != None):
          found_colors.append(color)
    if (len(found_colors) > 0):
      print('found colors {}'.format(found_colors))


  def predict_color(self, imgs):
    if (len(imgs.shape) < 4):
      imgs.reshape((1,) + imgs.shape)


    predictions = self.model.predict(imgs)

    # dict = ['blue','red','yellow']
    classifications = []
    for out in predictions:
      dict = ['blue', 'none', 'red', 'yellow']
      o = np.where(out == max(out))
      if (len(o[0]) > 0 and max(out) > 0.75):
        classifications.append(dict[o[0][0]])
      else:
        print('weak solution {}'.format(out))
    return classifications

  def test_images(self, imgs):
    return self.predict_color(preprocess_input(resize_all(imgs)))

def test_video(model, testfile):
  print('Starting test on {}'.format(testfile))
  clip = VideoFileClip(testfile)
  tester = Tester(model)
  images = []
  for image in clip.iter_frames():
    images.append(image)
  compressed = compress(tester.test_images(np.array(images).astype(np.float32)))
  print(compressed)
  print(detect_pattern(compressed, ['blue','none','blue']))
  print('**********************************')

def picamera_loop(model):
  import time
  import picamera
  import picamera.array
  tester = Tester(model)
  tweeter = Tweeter()
  with picamera.PiCamera(resolution=(227,227)) as camera:
    camera.start_preview()
    camera.hflip=True
    # Camera warm-up time
    time.sleep(2)
    v=0
    while (True):
      images = []
      for i in range(30):
        with picamera.array.PiRGBArray(camera) as stream:
          camera.capture(stream, format='rgb',use_video_port=True)
          # At this point the image is available as stream.array
          image = stream.array
          images.append(image)
          time.sleep(0.02)

      compressed_classifications = compress(tester.test_images(np.array(images).astype(np.float32)))
      print(compressed_classifications)

      if (detect_pattern(compressed_classifications,['blue','none','blue'])):
        print('blue is blinking!')
        tweeter.tweet("I'm full @faraz_r_khan! Kitty poops too much")

      if (detect_pattern(compressed_classifications,['red'], 10)):
        print('red is on')
        tweeter.tweet('haha kitty just pooped', 900)


def main(argv):
  try:
    opts, args = getopt.getopt(argv, "ct")
  except getopt.GetoptError:
    print("Please specify -c or -t")
    sys.exit(2)

  if (len(opts) == 0):
    print('please specify -c or -t')
    sys.exit(2)

  for opt, arg in opts:
    if opt == '-t':
      model = get_model()
      test_video(model, 'testVideos/test_blue_yellow.mp4')
      test_video(model, 'testVideos/blinking_yellow.mp4')
      test_video(model, 'testVideos/blinking_blue_med.mp4')
      test_video(model, 'testVideos/blinking_blue_close.mp4')
      test_video(model, 'testVideos/blinking_blue_far.mp4')
      # test_images(model)
      del model
    elif opt == '-c':
      model = get_model(False)
      if (model == False):
        print('A client with no model... like a boy who wanders in the forest')
        sys.exit(2)
      picamera_loop(model)

import twython
import random,string

def randomword(length):
   return ''.join(random.choice(string.ascii_lowercase) for i in range(length))

class Tweeter:
  twitter = None

  last_tweet_sent = 0

  def __init__(self):

    self.twitter = twython.Twython(
      auth.CONSUMER_KEY,
      auth.CONSUMER_SECRET,
      auth.ACCESS_TOKEN,
      auth.ACCESS_TOKEN_SECRET
    )
    self.tweet('Powering back up')
    self.last_tweet_sent = 0

  def tweet(self, message, grace_period=3600):
    if (time.time() - self.last_tweet_sent < grace_period):
      print('not tweeting {} cuz itll be spammy'.format(message))
      return

    try:
      message = '{0} -{1}'.format(message, randomword(3))
      self.twitter.update_status(status=message)
    except twython.TwythonError as e:
      print(e)
    except:
      print("twitter error:", sys.exc_info()[0])
    self.last_tweet_sent = time.time()

if __name__ == "__main__":
  Tweeter()
  main(sys.argv[1:])
