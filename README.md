# DeepCat
CNN Based status detector for the Litter Robot III. 

Basicaly I have a litter robot which is shoved into a far corner of the house which only Mr. Cat frequents. So much to the cat's disgust, it ends up blinking blue (which signifies full litter bin) for many days without nobody noticing. The idea was to build something that can detect blinking blue (amongst other) lights. 

## Cool so whats the 'NN' part?
I understand that there are atleast 2 much easier ways of solving this problem:
- I could have opened the litter robot control panel and tapped into the LEDs and fed them to a photon/raspberry pi. That would be too easy
- I could have mounted a raspberry pi with a camera pointed in a known way and then used openCV to look for the blue/red/yellow or the absence thereof.
- I can use a Neural net to do all of this. I just really wanted to use a CNN in a home project, basically

More to come...

