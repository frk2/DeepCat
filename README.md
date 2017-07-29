# DeepCat
CNN Based status detector for the Litter Robot III. 

I have a litter robot (https://www.litter-robot.com/) which is shoved into a far corner of the house which only Mr. Cat frequents. So much to the cat's disgust, it ends up blinking blue (which signifies full litter bin) for many days without nobody noticing. The idea was to build something that can detect blinking blue (amongst other) lights. 

## Cool so whats the 'NN' part?
I understand that there are atleast 2 much easier ways of solving this problem:
- I could have opened the litter robot control panel and tapped into the LEDs and fed them to a photon/raspberry pi. That would be too easy
- I could have mounted a raspberry pi with a camera pointed in a known way and then used openCV to look for the blue/red/yellow or the absence thereof.

OR

I can use a Neural net to do all of this. I just really wanted to use a CNN in a home project, basically.

In the end I have a classifier uses a fine-tuned SqueezeNet that does _fairly_ well on images of varying brightness and registers ~ 94% test accuracy on images taken from different cameras *if* I stay within some defined boundaries (particular range of distance away, all 3 lights showing in the picture, etc). In here I discuss what I tried and what went wrong and hope it helps someone :)

PLEASE feel free to email me if you have a suggestion on how I can improve this or if Im doing something completely wrong. I'm a noob when it comes to neural nets so any help/advice is welcome!

## Data Collection
This was by far the hardest part. I figured out quite a bit (I think!) on how to collect data and what to use as training data. What I have available as train_data_orig.tar I collected by recording a video from various angles and distances, using my phone camera and then the raspberry pi camera. I moved the cameras around to generate offsets in the image as a method to increasing the number of data points. Images like the following were generated:

| Red | Yellow | Blue | None |
|---|---|---|---|
|![alt text](testImages/red.jpg "Red")|![alt text](testImages/yellow1.jpg "Yellow")|![alt text](testImages/blue1.jpg "Blue")|![alt text](testImages/none.jpg "None")|

So 4 classes, ~ 2000 images per class. A decent amount of data augmentation was done using the keras ImageDataGenerator which happily adds shear, zoom, translate and rotations to your images. Very handy!



## The Model
I ended up using a Keras Squeezenet port which was pretrained on Imagenet. Thanks to this guy https://github.com/rcmalli/keras-squeezenet. 

Ive been toying with two different techniques for this. In the first I removed the last softmax layer of the model and added my own 4-class Dense layer:
```
model = SqueezeNet()
x = Dense(4, activation='softmax')(model.layers[-2].output)
model = Model(model.inputs, x)
```
This gives me ~ 93% test accuracy.

In the second I took out the entire convnet classifier from Squeezenet:

```
model = SqueezeNet()
x = Convolution2D(4, (1, 1), padding='same', name='conv11')(model.layers[-5].output)
x = Activation('relu', name='relu_conv10')(x)
x = GlobalAveragePooling2D()(x)
x = Activation('softmax')(x)
model = Model(model.inputs, x)
```

Give me 95% Test accuracy. Seems to initially work better.

This was inspired by the fact that Squeezenet has the last convnet generally the size of classes. I'm not sure if this is a sensible approach but seems to work. Does it make sense to use weights of a pretrained model and yank an entire layer this way? Please tell me!

The jury is still out which is better. 

## What worked
- Using a pre-trained Squeezenet! I realized that folks who participiated in the Nexar challenger for traffic light detection were doing that - and my problem is basically super similar so why not. Using something pretrained was key to getting real generalization

- Using your own accuracy function! I dont know if its my Keras version but I always end up gettting an accuracy of 100% using evaluate_generator when in reality my accuracy was near 0! I found out that apparently keras calculates accuracy in mysterious ways so the most reliable way was to do it yourself using the following snippet:
```
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
```

So given a model and a test generator, this guy will calculate your prediction and compare to the truth. The classification is chosen if >60% of the softmax probability is in your favor. Saved my life. Before this everything called accuracy was a lie! (Or I dont understand how keras accuracy works)

- Balancing was important since for some I kept producing way more blue and yellow frames than 'none' or red frames. I had to manually get rid of blue/yellow frames to bring it to par. I got rid of 'bad' images like ones where one of the lights was cut off etc.

- Checkpointing. It was super crazy easy to start overfitting, sometimes even after the first epoch! Going back to earlier models that showed high test accuracy was always way superior than a later overfitted model.

## What didn't work
Well a lot of stuff. Here goes:

- Using a different color space: I thought maybe I could extract the S channel from HLS and intuitively that looks like it might work but in reality it makes accuracy worse

- Training your own model. I tried many different models (including the NVidia pipeline from their end to end paper) but they all sucked in that they overfit to the extreme.  They performed well if the parameters didnt change - like if the test dataset was generated using the same camera and in the same situations as the training set but terrible otherwise.

- Using color balacing to negate the effects of illumination. I tried using equalize_adapthist on the training set and all it did was slow the entire thing down. This was to combat training data gathered in sunlight vs testing data gather later in the day in warm LED light.

- Cropping the image to focus always on the lights. This worked, except it super duper overfit the model to expect images in exactly that distance, captured at exactly that resolution.

- Augmenting yourself. Too painful, just use ImageDataGenerator! Its free!

- I quickly realized that lighting had a big role to play so if I trained my classifier in daylight it would perform poorly at night when my yellow warm LED lighting would come on. So I decided to feed it in some data with the yellow light shining as well to help it generalize. I tried using equalize_adapthist to fix the lightning situation but couldn't find any easy way to completely remove it from the picture. If anyone knows how to, please let me know!
