import os, sys

import numpy as np

from skimage.io import imread

from algorithm import Algorithm

class Alignment:
    def __init__(self, image):
        self.image = image

        if type(image).__module__ == np.__name__:
            self.image_ = np.copy(image)
        else:
            if not os.path.isfile(image):
                sys.stderr.write("Image file does not exists or we do not have permission to read it\n")
                sys.exit(1);

            self.image_ = imread(image)

        self.image_ = (self.image_ - np.min(self.image_) ) / np.max(self.image_)
        self.original = np.copy(self.image_)

        self.parameters = {
            'binary_threshold': 0.25,
            'canny_thresh_low': 0.5,
            'canny_thresh_high': 0.8,
            'gap': 100,
            'transpose': False,
            'debug': False,
            'quiet': False,
            'mode' : 'pin'
        }

    def compute_center(self, algorithm, params):
        """
        """
        for k,v in params.items():
            self.parameters[k] = v
        
        if algorithm == 'pin':
            return Algorithm.compute_center_pin(self.image_, self.parameters);
        elif algorithm == 'slit':
            return Algorithm.compute_center_slit(self.image_, self.parameters);
        elif algorithm == 'show':
            cv2.imshow("Window-1", self.original)            
            cv2.waitKey(0)
        else:
            sys.stderr.write("Invalid algorithm " + algorithm)
            sys.exit(1);
            


   