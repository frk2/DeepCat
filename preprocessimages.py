import glob
import cv2
from skimage import exposure
from tqdm import tqdm
import matplotlib.pyplot as plt
import skimage
IMG_SIZE = (256,144)

images = glob.glob("import/*")
for image in tqdm(images):
  img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_RGB2BGR)
  cv2.imwrite(image, img)


