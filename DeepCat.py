from moviepy.editor import VideoFileClip
import numpy as np
import pickle
import cv2
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
from keras.callbacks import  EarlyStopping, ModelCheckpoint
import keras.optimizers
import glob
import sys, getopt
import h5py
import skimage
from keras.preprocessing.image import ImageDataGenerator
from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image


IMG_SIZE = (227,227)
IMG_SHAPE = IMG_SIZE + (3,)
from skimage import exposure

from tqdm import tqdm

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


class Processor:
  current_label = ''
  curr_count = 0
  x_dataset = y_dataset = None
  h5f = None
  total_count = 0

  def __init__(self, max=300000):
    self.h5f = h5py.File('train.h5', 'w', libver='latest')
    size = (max,) + IMG_SIZE + (3,)

    self.x_dataset = self.h5f.create_dataset('X', size)
    self.y_dataset = self.h5f.create_dataset('Y', (max,), dtype='S10')

  def set_next_label(self, label):
    print('{} count: {}'.format(self.current_label, self.curr_count))
    self.current_label = label
    self.curr_count = 0

  def processImage(self, mimg):
    # Generate shit loads of different data
    min_crop_x = 400
    min_crop_y = 300

    ar = min_crop_y / min_crop_x
    step_x = 200
    cropx = 800
    cropy = 800
    adj = [ (-100,-50), (-100,0), (0,0), (100,0), (100,50)]
    #while (cropx > min_crop_x  and cropy > min_crop_y):
    for iadj in adj:
      img = crop_center(mimg, 1000, 1000, iadj[0], iadj[1])
      # img = crop_leftbottom(mimg, 0, 0, ,900, iadj[0], iadj[1])
      img = preprocess(img)
      self.curr_count += 1
      # plt.imshow(img)
      # plt.show()
      self.x_dataset[self.total_count] = img
      self.y_dataset[self.total_count] = np.string_(self.current_label)
      self.total_count += 1
      # cropx -= int(step_x)
      # cropy -= int(step_x * ar)

    return img

  def write(self):
    self.x_dataset.attrs['size'] = self.total_count
    self.y_dataset.attrs['size'] = self.total_count
    # self.x_dataset.resize(final_size)
    # self.y_dataset.resize(self.total_count)
    self.h5f.close()

def load():
  if Path('train.h5').is_file():
    h5f = h5py.File('train.h5', 'r')
    X = h5f['X']
    Y = h5f['Y']
    total = h5f['X'].attrs['size']
    # total = 40000
    print('loaded {} images'.format(total))
    return X,Y,total
  else:

    clips = [('fullredlong.mp4', 'red'), ('redlong.mp4' , 'red'), ('redcenter.mp4', 'red'), ('fullbluelong.mp4','blue'), ('bluecenter.mp4', 'blue'), ('bluelong.mp4','blue'),
             ('fullyellowlong.mp4', 'yellow'), ('yellowlong.mp4','yellow'), ('yellowcenter.mp4', 'yellow'),
             ('fullnonelong.mp4', 'none'), ('nonelong.mp4', 'none')]

    p = Processor()

    for clip in clips:
      print('Loading data from {} as {}'.format(clip[0], clip[1]))
      video_clip = VideoFileClip(clip[0])
      p.set_next_label(clip[1])
      for frame in tqdm(video_clip.iter_frames()):
        p.processImage(frame)
    p.set_next_label('meow')
    p.write()
    return p.x_dataset, p.y_dataset, p.total_count

def gen_preprocess(x):
  x = np.expand_dims(x, axis=0)
  return preprocess_input(x)

def get_model(train=True):
  datagen = ImageDataGenerator(
    rotation_range=0,
    width_shift_range=0.3,
    height_shift_range=0.4,
    shear_range=0.0,
    zoom_range=0.3,
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
    class_mode='categorical'
  )

  test_generator = validgen.flow_from_directory(
    directory='test_data/',
    target_size=IMG_SIZE,
    class_mode='categorical'
  )


  if Path('model.h5').is_file():
    model = load_model('model.h5')
    print(model.metrics_names)
    # print(model.evaluate_generator(valid_generator, steps=100))
    print(model.evaluate_generator(test_generator, steps=50))
    return model
  else:
    if (not train):
      return False

    X_train, Y_train, total = load()
    # X_train = np.array(X_train)
    Y_train = np.array([x.decode() for x in Y_train[:total]])
    fit = LabelBinarizer().fit(Y_train)
    Y_train = fit.transform(Y_train)
    # X_train = X_train.reshape(X_train.shape + (1,))
    # X_train, Y_train = shuffle(X_train, Y_train)

    # X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train,test_size=0.33, random_state=42)

    model = SqueezeNet()
    print(model.summary())
    x = Convolution2D(4, (1, 1), padding='same', name='conv11')(model.layers[-4].output)
    x = Activation('relu', name='relu_conv10')(x)
    x = GlobalAveragePooling2D()(x)
    x= Dense(4, activation='softmax')(x)
    model = Model(model.inputs, x)
    print(model.summary())
    # # 5th Layer - Add a ReLU activation layer
    # model.add(Activation('softmax'))

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
    callbacks = [
      EarlyStopping(monitor='loss', min_delta=0.005, patience=2, verbose=1),
      # ModelCheckpoint(kfold_weights_path, monitor='val_loss', save_best_only=True, verbose=0),
    ]


    model.compile(keras.optimizers.Adam(lr=0.0005), 'categorical_crossentropy', ['accuracy'])
    # history = model.fit(X_train, y_one_hot, batch_size=256, nb_epoch=20, validation_split=0.2, verbose=1, shuffle=True, callbacks=callbacks)
    model.fit_generator(data_generator, steps_per_epoch=300, epochs=30, verbose=1, callbacks=callbacks, validation_data=test_generator, validation_steps=50)
    model.save('model.h5', True)

    return model

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


  def predict_color(self, img):
    out = self.model.predict(img)[0]
    # dict = ['blue','red','yellow']
    dict = ['blue', 'none', 'red', 'yellow']
    o = np.where(out == max(out))
    if (len(o[0]) > 0 and max(out) > 0.75):
      print('picture is {}, {}'.format(dict[o[0][0]], out))
      return dict[o[0][0]]
    else:
      print('weak solution {}'.format(out))
      return None

  def test_image(self, img):
    # img = crop_center(img, 0, 0, 1000,1000)
    # plt.imshow(img)
    # plt.show()
    img = np.array(preprocess(img))
    return self.predict_color(img)
    return img

def test(model, testfile):
  print('Starting test on {}'.format(testfile))
  clip = VideoFileClip(testfile)
  tester = Tester(model)
  for image in clip.iter_frames():
    tester.test_image(image)
  print('**********************************')

def test_images(model):
  tester = Tester(model)
  images = glob.glob("testImages/*")
  for i in images:
    img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB)
    print('Testing {}'.format(i))
    tester.test_image(img)

def picamera_loop(model):
  import time
  import picamera
  import picamera.array
  tester = Tester(model)
  with picamera.PiCamera() as camera:
    #camera.resolution = (1920, 1080)
    camera.start_preview()
    camera.hflip=True
    # Camera warm-up time
    time.sleep(2)
    while (True):
      images = []
      for i in range(10):
        with picamera.array.PiRGBArray(camera) as stream:
          camera.capture(stream, format='rgb')
          # At this point the image is available as stream.array
          image = stream.array
          images.append(image)
        time.sleep(0.2)

      print('evaluating batch')
      for image in images:
        print(tester.test_image(image))

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
      test(model, 'testVideos/test_blue_yellow.mp4')
      test(model, 'testVideos/blinking_yellow.mp4')
      test(model, 'testVideos/blinking_blue_med.mp4')
      test(model, 'testVideos/blinking_blue_close.mp4')
      test(model, 'testVideos/blinking_blue_far.mp4')
      # test_images(model)
      del model
    elif opt == '-c':
      model = get_model(False)
      if (model == False):
        print('A client with no model... like a boy who wanders in the forest')
        sys.exit(2)
      picamera_loop(model)

if __name__ == "__main__":
  main(sys.argv[1:])
