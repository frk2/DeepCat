# DeepCat
CNN Based status detector for the Litter Robot III. 

This detects the status lights and currently tweets to the world about blinking blue (litter bin full) or solid red (the robot was recently used!)  as https://twitter.com/kitty_litterbot

Why and How? Checkout https://medium.com/@farazrkhan/deepcat-a-cnn-based-status-light-detector-for-the-litter-robot-iii-6b4746d92785

Also - terribly organized code follows so proceed at your own risk. Its a litter detector after all!

## Usage
`python3 DeepCat.py -c` launches the Raspberry Pi camera loop 'client' version
`python3 DeepCat.py -t` launches training if `model.h5` doesn't exist - otherwise runs through tests.

## Prerequisites
There may be more, but at the least you need (on your computer AND the raspberry Pi):
- Tensorflow / Keras
- numpy, scipi, scikit-learn, h5py and opencv3 for python 







