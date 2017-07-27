import glob
import cv2
from skimage import exposure
from tqdm import tqdm
import matplotlib.pyplot as plt
import skimage
IMG_SIZE = (256,144)

images = glob.glob("test_data/*/*")
for image in tqdm(images):
  img = cv2.imread(image)
  img = exposure.equalize_adapthist(img)
  img = cv2.resize(img, IMG_SIZE)
  img *= 255
  cv2.imwrite(image, img)


