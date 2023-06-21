#%%
from PIL import Image
import numpy as np
from mnist_loader import LabeledImage
import tkinter as tk
from tkinter import filedialog

YES = {'Y','y','YES','yes','Yes'}
NO = {'N','n','NO','no','No'}

def process(filename, label=None):
    # open image 
    img = Image.open(filename)
    img = img.convert('L')
    w, h = img.size


    while True:             # rotate
        new_img = img.copy()
        new_mat = new_img.load()
        
        new_img.show()
        if input('rotate? (y/n) ') in YES:
            deg = _int_input('how many turns? (clockwise) ')
            new_img = img.rotate(-deg*90)
            if _is_fine(new_img): break
        else:
            break
    
    img = new_img
    while True:             # to BW
        new_img = img.copy()
        new_mat = new_img.load()
    
        thresh = _int_input('Choose BW sensitivity (0-100): ')
        thresh = thresh/100 * 255
        for y in range(h):
            for x in range(w):
                new_mat[x,y] = 0 if new_mat[x,y] > thresh else 255
        
        if _is_fine(new_img): break

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
    w,h = img.size
    top = left = np.inf
    bot = right = 0
    
    # find digit bounds
    for x in range(w):
        for y in range (h):
            if mat[x,y] == 255:
                left = min(x,left)
                right = max(x,right)
                top = min(y,top)
                bot = max(y,bot)

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

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()
    img = process(file_path)
    if img:
        return img
    else:
        return load_image()
