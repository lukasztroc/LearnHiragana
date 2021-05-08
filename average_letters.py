import numpy as np
from PIL import Image
import os
import PIL.ImageOps
import shutil
from PIL import ImageEnhance

shutil.rmtree('files/samples', ignore_errors=True)
os.makedirs('files/samples')

dataset = np.load('dev/k49-train-imgs.npz')['arr_0']
y = np.load('dev/k49-train-labels.npz')['arr_0']

for index in range(49):
    arr = dataset[y == index]
    asd = np.mean(arr, axis=0)
    im = Image.fromarray(asd).convert('RGB')
    enhancer = ImageEnhance.Sharpness(im)
    im = enhancer.enhance(25)
    im = PIL.ImageOps.invert(im)
    im.save(f'files/samples/{index}.png', format='png')

