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
from keras.layers import Convolution2D, MaxPooling2D, Dropout
from keras.models import load_model
import keras.optimizers
import glob

from skimage import exposure

from tqdm import tqdm

def preprocess(img, should_crop = True):
  if (should_crop):
    img = crop_center(img, 400, 300, -100)
  #img = cv2.cvtColor(mimg, cv2.COLOR_RGB2HSV);
  #img = mimg[:, :, 2]
  img = cv2.resize(img, (40, 40))
  # plt.imshow(img)
  # plt.show()
  img = (img / 255.)
  # Apply localized histogram localization
  #img = exposure.equalize_adapthist(img)
  return img

def crop_center(img, cropx, cropy, adjx=0, adjy=0):
  y, x, _ = img.shape
  startx = x // 2 - (cropx // 2) + adjx
  starty = y // 2 - (cropy // 2) + adjy
  startx = min(max(0, startx), img.shape[1])
  starty = min(max(0, starty), img.shape[0])
  return img[starty:starty + cropy, startx:startx + cropx]


class Processor:
  current_label = ''
  X_train = []
  Y_train = []
  curr_count = 0
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
    adj = [ (-50,-50), (-50,0), (0,0), (50,0), (50,50)]
    #while (cropx > min_crop_x  and cropy > min_crop_y):
    for iadj in adj:
      img = crop_center(mimg, cropx, cropy, iadj[0], iadj[1])
      img = preprocess(img, False)
      self.curr_count += 1
      # plt.imshow(img)
      # plt.show()
      self.X_train.append(img)
      self.Y_train.append(self.current_label)
      # cropx -= int(step_x)
      # cropy -= int(step_x * ar)

    return img

  def write(self):
    dist_pickle = {}
    dist_pickle["data"] = self.X_train
    dist_pickle["labels"] = self.Y_train
    pickle.dump(dist_pickle, open( "train.p", "wb"))



def load():
  if Path('train.p').is_file():
    training_data = pickle.load(open('train.p', 'rb'))
    if (len(training_data) > 0):
      return training_data['data'], training_data['labels']
  else:
    red_clip = VideoFileClip('fullredlong.mp4')

    blue_clip = VideoFileClip('fullbluelong.mp4', )
    none_clip = VideoFileClip('fullnonelong.mp4')
    yellow_clip = VideoFileClip('fullyellowlong.mp4')
    #blue_clip = clip.subclip(1, 8.0)
    #yellow_clip = clip.subclip(14.0, 21.0)
    p = Processor()
    write = False
    if (write):
      p.set_next_label('red')
      out = red_clip.fl_image(p.processImage)
      out.write_videofile('redout.mp4', audio=False)
      p.set_next_label('blue')
      out = blue_clip.fl_image(p.processImage)
      out.write_videofile('blueout.mp4', audio=False)
      p.set_next_label('yellow')
      out = yellow_clip.fl_image(p.processImage)
      out.write_videofile('yellowout.mp4', audio=False)
      # p.set_next_label('none')
      # none_clip.fl_image(p.processImage)
      # none_clip.write_videofile('noneout.mp4', audio=False)
    else:
      p.set_next_label('red')
      for frame in red_clip.iter_frames():
        p.processImage(frame)
      p.set_next_label('blue')
      for frame in blue_clip.iter_frames():
        p.processImage(frame)
      p.set_next_label('yellow')
      for frame in yellow_clip.iter_frames():
        p.processImage(frame)

      p.set_next_label('none')
      for frame in none_clip.iter_frames():
        p.processImage(frame)
      p.set_next_label('meow')

    p.write()
    return p.X_train, p.Y_train

def get_model():
  if Path('model.h5').is_file():
    return load_model('model.h5')
  else:
    X_train, Y_train = load()
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    #X_train = X_train.reshape(X_train.shape + (1,))
    X_train, Y_train = shuffle(X_train, Y_train)
    bin = LabelBinarizer().fit(Y_train)
    y_one_hot = bin.transform(Y_train)
    print(bin.classes_)

    model = Sequential()

    model.add(Convolution2D(24, 5, 5,
                            border_mode='same',
                            input_shape=X_train[0].shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(36, 5, 5,
                            border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(48, 5, 5,
                            border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3,
                            border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Convolution2D(64, 3, 3,
                            border_mode='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # 1st Layer - Add a flatten layer
    model.add(Flatten())

    model.add(Dense(1164))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    # 2nd Layer - Add a fully connected layer
    model.add(Dense(50))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('relu'))
    # 4th Layer - Add a fully connected layer
    model.add(Dense(4))
    # 5th Layer - Add a ReLU activation layer
    model.add(Activation('softmax'))
    # TODO: Build a Multi-layer feedforward neural network with Keras here.
    # TODO: Compile and train the model

    model.compile(keras.optimizers.Adam(lr=0.001), 'categorical_crossentropy', ['accuracy'])
    history = model.fit(X_train, y_one_hot, batch_size=256, nb_epoch=20, validation_split=0.2, verbose=1, shuffle=True)
    model.save('model.h5', True)
    return model

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
        new_img = np.array(preprocess(new_img, False))
        plt.imshow(new_img)
        plt.show()

        color = self.predict_color(new_img)
        if (color != None):
          found_colors.append(color)
    if (len(found_colors) > 0):
      print('found colors {}'.format(found_colors))


  def predict_color(self, img):
    img = img.reshape((1,) + img.shape)
    out = model.predict(img)[0]
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
    img = crop_center(img, 700, 700, 00,-100)
    img = np.array(preprocess(img, False))
    self.predict_color(img)
    # plt.imshow(img)
    # plt.show()

    return img

def test(model, testfile):
  clip = VideoFileClip(testfile)
  tester = Tester(model)
  for image in clip.iter_frames():
    tester.test_image(image)

def test_images(model):
  tester = Tester(model)
  images = glob.glob("testImages/*")
  for i in images:
    img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB)
    print('Testing {}'.format(i))
    tester.test_image(img)
if __name__ == "__main__":
  model = get_model()
  test(model,'test_blue_yellow.mp4')
  #test_images(model)
  del model

"""
  input = np.empty((3,150,200,3))
  red = np.array(preprocess(cv2.imread('Red.jpg')))
  red = cv2.cvtColor(red, cv2.COLOR_RGB2BGR)
  #plt.imshow(red)
  #plt.show()
  red = red / 255.0
  input[0,:,:,:] = red

  red = np.array(preprocess(cv2.imread('Yellow.jpg')))https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=3&ved=0ahUKEwjTw_L3mJzVAhUFl1QKHSFWBc4QFgg1MAI&url=https%3A%2F%2Fgithub.com%2Ffchollet%2Fkeras%2Fissues%2F2743&usg=AFQjCNGQ6NuSwACM7kqQEHcaVd1-R7mVnA
  red = cv2.cvtColor(red, cv2.COLOR_RGB2BGR)
  #plt.imshow(red)
  #plt.show()
  red = red / 255.0
  input[1, :, :, :] = red

  red = np.array(preprocess(cv2.imread('Blue.jpg')))
  red = cv2.cvtColor(red, cv2.COLOR_RGB2BGR)
  #plt.imshow(red)
  red = red / 255.0
  input[2, :, :, :] = red

  out = model.predict(input)
  out = out > 0.5
  print('Model predicted {}'.format(out))
  dict = ['blue','none','red','yellow']
  for i in range(3):
    o = np.where(out[i])
    print('{}th picture is {}'.format(i, dict[o[0][0]]))
"""














