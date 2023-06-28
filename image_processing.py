'''image processing to make it conform to mnist format'''
#%%
import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image
from mnist_loader import LabeledImage

YES = {'Y','y','YES','yes','Yes'}
NO = {'N','n','NO','no','No'}

def process(filename, label=None):
    '''Image processing procedure'''
    # open image
    img = Image.open(filename)
    img = img.convert('L')
    width, height = img.size


    while True:             # rotate
        new_img = img.copy()
        new_mat = new_img.load()

        new_img.show()
        if input('rotate? (y/n) ') in YES:
            deg = _int_input('how many turns? (clockwise) ')
            new_img = img.rotate(-deg*90)
            if _is_fine(new_img):
                break
        else:
            break

    img = new_img
    while True:             # to BW
        new_img = img.copy()
        new_mat = new_img.load()

        thresh = _int_input('Choose BW sensitivity (0-100): ')
        thresh = thresh/100 * 255
        for _y in range(height):
            for _x in range(width):
                new_mat[_x,_y] = 0 if new_mat[_x,_y] > thresh else 255

        if _is_fine(new_img):
            break

    new_img = crop(new_img)

    # resize
    new_img = new_img.resize((28,28), 1)

    # to labled image
    new_mat = new_img.load()
    img_array = np.array([[new_mat[x,y] for x in range(28)]for y in range(28)])
    result = LabeledImage(img_array, label)

    return result

def crop(img, pad = 0.2):
    ''' Takes a BW image of a digit on a white background and crops it to a square '''
    mat = img.load()
    _w,_h = img.size
    top = left = np.inf
    bot = right = 0

    # find digit bounds
    for _x in range(_w):
        for _y in range (_h):
            if mat[_x,_y] == 255:
                left = min(_x,left)
                right = max(_x,right)
                top = min(_y,top)
                bot = max(_y,bot)

    length = max(right - left, bot - top) # square crop length
    pad = pad * length
    cropped = img.crop((
                left - pad,
                top - pad,
                left + length + pad,
                top + length + pad))
    return cropped




def _int_input(prompt):
    while True:
        try:
            res = int(input(prompt))
            break
        except ValueError:
            print('input is not a number')
            continue
    return res

def _is_fine(img):
    img.show()
    check = input('is the processing fine? (y/n) ')
    return check in YES

def load_image():
    '''load image using a dialogue box'''

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()
    img = process(file_path)
    if img:
        return img

    return load_image()
