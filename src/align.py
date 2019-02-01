import numpy as np
import cv2
import time
from skimage.io import imread, imsave
from skimage.transform import resize

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from utils import smooth, create_debug_image

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

    def compute_center(self, params):
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
        edges = np.transpose(edges)

        winW, winH = 250, 250
        window = [winW, winH]
        step_size = 100
        nonzero_threshold = 100

        for x,y,window in self.detection_windows(edges, window, step_size):
            if window.shape[0] != winH or window.shape[1] != winH:
                continue

            xx, yy = np.nonzero(window)

            if len(yy) < nonzero_threshold:
                continue

            fig = Figure()
            canvas = FigureCanvas(fig)
            ax = fig.gca()

            ysmooth =  -yy 
            xindices = np.linspace(0, len(ysmooth), len(ysmooth))
            plt.scatter(xindices, ysmooth)
            plt.show()
            # ax.axis('off')

            canvas.draw() #draw the canvas, cache the renderer
            s, (width, height) = canvas.print_to_buffer()

            the_plot = np.fromstring(s, dtype='uint8').reshape((height, width, 4))
            # the_pot = np.resize(the_plot, (480, 640))

            clone = np.dstack((edges.copy(), edges.copy(), edges.copy()))
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)

            print(ysmooth)
            max_pt = np.argmax(ysmooth)
            print ("Max pt ", max_pt)
            data = ysmooth[max(0, max_pt-30):max_pt+30]
            d = np.sign(np.diff(data))
            print (sum(d[:max_pt]), sum(d[max_pt:]))

            cv2.imshow("Window-1", the_plot)            
            cv2.imshow("Window-2", window)
            cv2.waitKey(0)

        # if self.parameters['debug']:
        #     imsave('original.png', self.image_)
        #     imsave('binary_threshold.png', binary * 255)
        #     imsave('edges.png', np.uint8(edges))

    def detection_windows(self, image, window_size, step_size=30):
        h, w = image.shape

        for y in range(0, h, step_size):
            for x in range(0, w, step_size):
                yield(x, y, image[y:y+window_size[1], x:x+window_size[0]])







   