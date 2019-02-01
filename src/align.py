import numpy as np
import cv2
from skimage.io import imread, imsave
from skimage.transform import resize
import matplotlib.pyplot as plt


class Alignment:
    def __init__(self, image):
        self.image = image

        if type(image).__module__ == np.__name__:
            self.image_ = np.copy(image)
        else:
            self.image_ = imread(image)

        self.image_ = (self.image_ - np.min(self.image_) ) / np.max(self.image_)

        self.parameters = {
            'binary_threshold': 0.25,
            'canny_thresh1': 0.5,
            'canny_thresh2': 1.0,
            'transpose': False,
            'debug': True
        }

    def align(self, params):
        for k,v in params.items():
            self.parameters[k] = v

        if self.parameters['transpose']:
            self.image_ = np.transpose(self.image_)

        binary_threshold = self.parameters['binary_threshold']

        canny1 = self.parameters['canny_thresh1']
        canny2 = self.parameters['canny_thresh2']


        binary = np.zeros_like(self.image_)
        binary[self.image_ > binary_threshold] = 1      
        binary = np.uint8(binary)

        edges = cv2.Canny(binary, canny1, canny2, apertureSize = 7)

     
        x, y = np.nonzero(edges > 0)
        print(np.max(x), np.min(x))
        print(np.max(y), np.min(y))

        plt.imshow(edges, cmap='gray')
        plt.show()
        
        if self.parameters['debug']:
            imsave('original.png', self.image_)
            imsave('binary_threshold.png', binary * 255)
            imsave('edges.png', np.uint8(edges))






   