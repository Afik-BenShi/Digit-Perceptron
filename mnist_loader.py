#%%
import gzip
import numpy as np
import matplotlib.pyplot as plt
from typing import List

class LabeledImage:
    '''
        Labeled Image class
        -------------------


        28x28 pixel image of handwritten number with it's label
    
        Attributes
        ----------
            image    : returns the image matrix.
            label    : returns the label of the image.
            image_1D : returns a 1D reshape of the image matrix.
            
        Functions
        ----------
            show : shows the image and label
    '''
    def __init__(self, image:np.array, label:int=None):
        '''
        Parameters
        ----------
            image : array like - grayscale values of the pixel.
                    if array is 1D, reshapes it to 28x28 2D array.
            label : int - the number in the image.
        '''
        if len(image) > 28: # 1D array
            self.__image = np.reshape(image,(28,28))/255
        else:
            self.__image = image
        
        self.__label = label
    
    # override set function
    def __setattr__(self, __name: str, __value) -> None:
        if __name in {'_LabeledImage__label', '_LabeledImage__image'}:
            object.__setattr__(self, __name, __value)
        else:
            raise AttributeError

    # override get function
    def __getattr__(self, __name: str):
        if __name == 'image_1D':
            return(self.__image.reshape((784, 1)))
        elif __name == 'label':
            return self.__label
        elif __name == 'image':
            return self.__image
        else:
            raise AttributeError

    # override == operator
    def __eq__(self, other) -> bool:
        ''' 
            If other is int compares int and label.
            If other is a labledImage, compares images. 
        '''
        typ = type(other)
        if typ == int:
            return self.label == other
        elif typ == LabeledImage:
            return np.array_equal(self.image, other.image)

    def show(self):
        ''' 
            Uses matplotlib.pyplot.imshow()
            to plot the image
        '''
        plt.imshow(self.__image, cmap='gray')
        plt.title(f'Handwritten {self.__label}')
        plt.show()
        return f'Handwritten {self.__label}'
    
    # representation
    def __repr__(self) -> str:
        return(f'Handwritten {self.__label}')
    

def load_train_data(
    imgs_path:str='mnist_data\\train-images-idx3-ubyte.gz',
    lbl_path:str='mnist_data\\train-labels-idx1-ubyte.gz'
    ) -> List[LabeledImage]:
    '''
        Loads MNIST train dataset of 60,000 handwritten numbers.
        Parameters
        ----------
            imgs_path : str - path to images file
            lbl_path  : str - path to labels file
        
        Returns
        -------
            labeled_imgs : list of LabledImage objects
    '''
    # read images
    with gzip.open(imgs_path, 'r') as gz_file:
        train_img_ds = list(gz_file.read())[15:] # first 15 bits arn't picture data
    # read labels
    with gzip.open(lbl_path, 'r') as gz_file:
        train_lbl_ds = list(gz_file.read())[8:] # first 8 bits arn't labels

    # create dictionary of image : label
    
    labeled_imgs = order_in_array(train_img_ds, 784, 60000)
    return labeled_imgs

def order_in_array(img_ds, img_len, lbl_ds, lbl_len, arrays_len):
    labeled_imgs = [None for i in range(arrays_len)]
    start_img = 0
    end_img = start_img + img_len
    start_lbl = 0
    end_lbl = start_lbl + lbl_len
    for i in range(arrays_len): 
        img = LabeledImage(img_ds[start_img:end_img], lbl_ds[start_lbl:end_lbl])
        labeled_imgs[i] = img
        
        start_img += img_len
        end_img = start_img + img_len
        start_lbl += lbl_len
        end_lbl = start_lbl + lbl_len
    
    return labeled_imgs


def load_test_data(
    imgs_path:str='mnist_data\\t10k-images-idx3-ubyte.gz',
    lbl_path:str='mnist_data\\t10k-labels-idx1-ubyte.gz'
    ) -> List[LabeledImage]:
    '''
        Loads MNIST test dataset of 10,000 handwritten numbers.
        Parameters
        ----------
            imgs_path : str - path to images file
            lbl_path  : str - path to labels file
        
        Returns
        -------
            labeled_imgs : list of LabledImage objects
    '''
    # read images
    with gzip.open(imgs_path, 'r') as gz_file:
        img_ds = list(gz_file.read())[16:] # first 15 bits arn't picture data
    # read labels
    with gzip.open(lbl_path, 'r') as gz_file:
        lbl_ds = list(gz_file.read())[8:] # first 8 bits arn't labels

    # create dictionary of image : label
    labeled_imgs = order_in_array(img_ds, 784, 10000)  
    return labeled_imgs
# %%
