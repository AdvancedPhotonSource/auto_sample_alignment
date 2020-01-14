import numpy as np
import cv2

from skimage.io import imsave
from skimage.transform import resize

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

from utils import smooth, create_debug_image, order_points
from trackpy import bandpass

class Algorithm:

    @staticmethod
    def compute_center_pin(image, params):
        image_, original = np.copy(image), np.copy(image)

        if params['transpose']:
            image_ = np.transpose(image_)
        
        image_ = (image_ - np.min(image_)) / np.max(image_)
        
        binary_threshold = params['binary_threshold']
        hist, bins = np.histogram(image_.ravel(), 1000, [0.0, 1.0])
        a = np.argmax(hist)
        binary = np.zeros_like(image_)
        print (bins[a])
        binary[image_ < bins[a]] = 255
        # binary[np.logical_and(image_ > bins[a]-0.20, image_ < bins[a]+0.20)] = 255
        binary = np.uint8(binary)

        cannyl = params['canny_thresh_low']
        cannyh = params['canny_thresh_high']

        # binary = np.zeros_like(image_)
        # binary[image_ > binary_threshold] = 255      
        # binary = np.uint8(binary)

        edges = cv2.Canny(binary, cannyl, cannyh, apertureSize = 3)

        winW, winH = params['window_width'], params['window_height']
        window = [winW, winH]
        step_size = 50
        nonzero_threshold = 300

        points = {}
        min_diff = float('Inf')
        ans_X, ans_Y = 0, 0
        for x,y,window in Algorithm.detection_windows(edges, window, step_size):
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
            data = ysmooth
            d = np.sign(np.diff(data))
            y_df1 = np.insert(np.diff(ysmooth), 0, 0)
            y_df2 = np.insert(np.diff(d), 0, 0) 

            if len(ysmooth) > 500:
                continue

            if params['debug']:
                print("******")
                print("Points in Y ", len(ysmooth))
                print("Max point ", max_pt)
            
            left_half = sum(y_df1[:max_pt])
            right_half = sum(y_df1[max_pt:])
            diff = abs(abs(left_half) - abs(right_half))
            
            if params['debug']:
                print ("Weight distribution", left_half, right_half, diff)
                print("******")

            if diff < min_diff:
                min_diff = diff
                new_y = np.argmax(-yy)
                ans_X, ans_Y = x + xx[new_y], y + yy[new_y]

            if params['debug']:
                cv2.circle(clone, (ans_X, ans_Y), 5, (0,0,255), -1)
                ratio = clone.shape[1] / clone.shape[0]
                h = 600
                clone = resize(clone, (h, h*ratio))
                cv2.imshow("Window-1", the_plot)            
                cv2.imshow("Window-2", resize(clone, (800, 600)))
                cv2.waitKey(0)
        try:        
            pt1 = (ans_X-300) + np.nonzero(binary[ans_Y+params['gap'], ans_X-300:ans_X])[0][-1]
            pt2 = ans_X + np.nonzero(binary[ans_Y+params['gap'], ans_X:ans_X+300])[0][0]
            midpoint = int(pt1 + (pt2-pt1)/2.0)
            ans_X = midpoint
        except:
            return -1, -1

        clone = np.dstack((original.copy(), original.copy(), original.copy()))

        if params['transpose']:
            ans_X, ans_Y = ans_Y, ans_X

        if not params['quiet']:
            cv2.circle(clone,(ans_X, ans_Y), 5, (0,0,255), -1)

            plt.figure(figsize=(20, 20))
            plt.imshow(clone)
            plt.show()
        
        return ans_X, ans_Y


    @staticmethod
    def compute_center_slit(image, params):
        image_, original = np.copy(image), np.copy(image)

        binary_threshold = params['binary_threshold']
        cannyl = params['canny_thresh_low']
        cannyh = params['canny_thresh_high']

        binary = np.zeros_like(image_)
        binary[image_ > binary_threshold] = 255      
        binary = np.uint8(binary)

        if params['debug']:
            cv2.imshow("Window-1", binary)            
            cv2.waitKey(0)

        im2, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        try:
            cnt = [ cnt.shape[0] for cnt in contours]
            cnt = contours[np.argmax(cnt)]
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype='int')
            box = order_points(box)
        except Exception as e:
            if params['debug']:
                print(e)
            return (-1, -1)

        if not params['quiet']:
            clone = np.dstack((original.copy(), original.copy(), original.copy()))
            for pt in box:
                cv2.circle(clone, (pt[0], pt[1]), 5, (0,0,255), -1)

            plt.figure(figsize=(20, 20))
            plt.imshow(clone)
            plt.show()
    
        return ','.join([f"({b[0]}, {b[1]})" for b in box])

    @staticmethod
    def detection_windows(image, window_size, step_size=30):
        h, w = image.shape

        for y in range(0, h, step_size):
            for x in range(0, w, step_size):
                yield(x, y, image[y:y+window_size[1], x:x+window_size[0]])







   