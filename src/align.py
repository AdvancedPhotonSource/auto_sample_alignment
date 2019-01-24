import numpy as np
import cv2
from skimage.io import imread, imsave
from skimage.transform import resize


class Alignment:
    def __init__(self, image):
        self.image = image

        if type(a).__module__ == np.__name__:
            self.image_ = np.copy(image)
        else:
            self.image_ = imread(image)

        self.parameters = {
            'binary_threshold': 0.25,
            'canny_thresh1': 0.5,
            'canny_thresh2': 1.0,
            'debug': True
        }

    def align(self, params):
        for k,v in params.items():
            self.parameters[k] = v

        binary_threshold = self.parameters['binary_threshold']
        canny1 = self.parameters['canny_thresh1']
        canny2 = self.parameters['canny_thresh2']

        binary = np.zeros_like(img2)
        binary[img2 > 0.4] = 1      
        binary = np.uint8(binary)

        if self.parmaeters['debug']:
            imsave('binary_threshold.jpg', binary)






   