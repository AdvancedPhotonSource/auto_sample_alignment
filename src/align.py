import numpy as np
import cv2
import time
import math
import operator
from skimage.io import imread, imsave
from skimage.transform import resize

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from utils import smooth, create_debug_image
from trackpy import bandpass

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
            'canny_thresh2': 0.8,
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
        binary[self.image_ > binary_threshold] = 255      
        binary = np.uint8(binary)

        edges = cv2.Canny(binary, canny1, canny2, apertureSize = 3)

        winW, winH = 200, 200
        window = [winW, winH]
        step_size = 50
        nonzero_threshold = 300

        # patch = edges[750:900, 600:850]
        # cv2.imshow("Window-1", patch)
        # cv2.waitKey(0)

        # y, x = np.nonzero(patch)    

        # indices = np.argsort(x)

        # ysmooth = smooth(-y[indices], 22)
        # xindices = np.linspace(0, len(ysmooth), len(ysmooth))
        # max_pt = np.argmax(ysmooth)
        # data = ysmooth[max(0, max_pt-100):max_pt+100]
        # d = np.sign(np.diff(data))
        # print(np.sum(d[:max_pt]), np.sum(d[max_pt:]))

        # plt.scatter(xindices, ysmooth)
        # plt.show()

        # y_df1 = np.insert(d, 0, 0)
        # y_df2 = np.insert(np.diff(y_df1), 0, 0) 

        # plt.scatter(np.linspace(0, len(y_df1), len(y_df2)), y_df2)

        points = {}
        min_diff = float('Inf')
        ans_X, ans_Y = 0, 0
        for x,y,window in self.detection_windows(edges, window, step_size):
            if window.shape[0] != winH or window.shape[1] != winH:
                continue

            yy, xx = np.nonzero(window)

            if len(yy) < nonzero_threshold:
                continue

            fig = Figure()
            canvas = FigureCanvas(fig)
            ax = fig.gca()

            indx = np.argsort(xx)
            ysmooth =  smooth(-yy[indx], 12)
            xindices = np.linspace(0, len(ysmooth), len(ysmooth))

            ax.scatter(xindices, ysmooth)
            
            canvas.draw()
            s, (width, height) = canvas.print_to_buffer()

            the_plot = np.fromstring(s, dtype='uint8').reshape((height, width, 4))

            clone = np.dstack((edges.copy(), edges.copy(), edges.copy()))
            cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)

            max_pt = np.argmax(ysmooth)
            # data = ysmooth[max(0, max_pt-30):max_pt+30]
            data = ysmooth
            d = np.sign(np.diff(data))
            y_df1 = np.insert(np.diff(ysmooth), 0, 0)
            y_df2 = np.insert(np.diff(d), 0, 0) 

            if len(ysmooth) > 500:
                continue

            print("******")
            print("Points in Y ", len(ysmooth))
            print("Max point ", max_pt)
            left_half = sum(y_df1[:max_pt])
            right_half = sum(y_df1[max_pt:])
            diff = abs(abs(left_half) - abs(right_half))
            print ("Weight distribution", left_half, right_half, diff)
            print("******")

            if diff < min_diff:
                min_diff = diff
                new_y = np.argmax(-yy)
                ans_X, ans_Y = x + xx[new_y], y + yy[new_y]

            # hitX = np.nonzero(y_df2)[0]
            # if True: #hitX.shape[0] == 1 and y_df2[hitX] == -2:
            #     # print (max_pt, hitX[0])

            #     # print (f"X = {x}, Y = {y}")
            #     new_y = np.argmax(-yy)
            #     pt = (x+xx[new_y], y+yy[new_y])
            #     # print (pt)
            #     # print (sum(d[:hitX[0]]), sum(d[hitX[0]:]))
            #     if pt not in points:
            #         points[pt] = 1
            #     else:
            #         points[pt] += 1

            #     cv2.circle(clone,(x+xx[new_y],y+yy[new_y]), 5, (0,0,255), -1)
            #     cv2.imshow("Window-1", the_plot)            
            #     cv2.imshow("Window-2", clone)
            #     cv2.waitKey(0)

        # pt = (max(points.items(), key=operator.itemgetter(1))[0])
        clone = np.dstack((edges.copy(), edges.copy(), edges.copy()))
        cv2.circle(clone,(ans_X, ans_Y), 5, (0,0,255), -1)
        plt.figure(figsize=(20, 20))
        plt.imshow(clone)
        plt.show()
        

        # if self.parameters['debug']:
        #     imsave('original.png', self.image_)
        #     imsave('binary_threshold.png', binary * 255)
        #     imsave('edges.png', np.uint8(edges))

    def detection_windows(self, image, window_size, step_size=30):
        h, w = image.shape

        for y in range(0, h, step_size):
            for x in range(0, w, step_size):
                yield(x, y, image[y:y+window_size[1], x:x+window_size[0]])







   