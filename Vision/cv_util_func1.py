import math
import sys
import cv2
import numpy as np
np.set_printoptions(threshold=sys.maxsize, linewidth=150)

class libLANE(object):
    def __init__(self, roi_height=5):
        self.height = 0
        self.width = 0
        self.min_y = 0
        self.mid_y_1 = 0
        self.mid_y_2 = 0
        self.max_y = 0
        self.match_mask_color = 255

        self.roi_height = roi_height

    def region_of_interest(self, img, vertices):
        mask = np.zeros_like(img)
        if len(img.shape) > 2:
            # white
            self.match_mask_color = (255,255,255)
        
        # All black except mask
        cv2.fillPoly(mask, vertices, self.match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    # result = α*initial_img + β*img + λ
    def weighted_img(self, img, initial_img, α=1, β=1., λ=0.):
        return cv2.addWeighted(initial_img, α, img, β, λ)
    
    # polar form: rho=length, theta=angle
    def hough_transform(self, img, rho=None, theta=None, threshold=None, mll=None, mlg=None, mode="lineP"):
        if mode == "line":
            return cv2.HoughLines(img.copy(), rho, theta, threshold)
        elif mode == "lineP":
            return cv2.HoughLinesP(img.copy(), rho, theta, threshold, lines=np.array([]),
                                   minLineLength=mll, maxLineGap=mlg)
        elif mode == "circle":
            return cv2.HoughCircles(img.copy(), cv2.HOUGH_GRADIENT, dp=1, minDist=80,
                                    param1=200, param2=10, minRadius=40, maxRadius=100)
    

    def morphology(self, img, kernel_size=(None, None), mode="opening"):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

        if mode == "opening":
            dst = cv2.erode(img.copy(), kernel)
            return cv2.dilate(dst, kernel)
        elif mode == "closing":
            dst = cv2.dilate(img.copy(), kernel)
            return cv2.erode(dst, kernel)
        elif mode == "gradient":
            return cv2.morphologyEx(img.copy(), cv2.MORPH_GRADIENT, kernel)
    def cutoff(self, mask, percentile=0.99, min_num = 1000):
        HEIGHT = mask.shape[0]
        num_target_pixels = int(np.sum(mask) / 255)
        if num_target_pixels < min_num:
            return np.zeros_like(mask), HEIGHT
        target_height_info = np.array([int(np.sum(mask[h,:])/255) for h in range(HEIGHT)])
        target_height_info = np.cumsum(target_height_info)
        thres = int(target_height_info[-1]*(1-percentile))
        thres_mask_height = len(target_height_info[target_height_info < thres])
        mask[:thres_mask_height, :] = 0
        return mask, thres_mask_height

    def preprocess(self, img):
        HEIGHT = img.shape[0]
        hls_image = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
        hsv_image = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        b,g,r = cv2.split(img)
        h,l,s_hls = cv2.split(hls_image)
        h,s_hsv,v = cv2.split(hsv_image)

        mask_road = cv2.inRange(img, np.array([0,0,0]),np.array([100,100,100]))
        mask_road, thres_road_height = self.cutoff(mask_road)
        # road_height_info = np.array([int(np.sum(mask_road[h,:])/255) for h in range(HEIGHT)])
        # road_height_info = np.cumsum(road_height_info)
        # thres = int(road_height_info[-1]*0.01)
        # thres_road_height = len(road_height_info[road_height_info < thres])
        # mask_road[:thres_road_height,:] = 0
        # num_road_pixels = int(np.sum(mask_road)/255)

        # # Considered noise if the number of 'road-gray' pixels is less than 200. It may be tuned
        # if num_road_pixels < 200:
        #     mask_road = np.zeros_like(mask_road)
        #     thres_road_height = HEIGHT
        
        # Assumed the width of lane takes up at most 100 pixels of height. Should be tuned.
        lane_width = 100    ### FIX ME
        thres_road_height = max(0,thres_road_height-lane_width)

        # mask_white: it contains flower leaves and lanes
        mask_white_bgr = cv2.inRange(img,np.array([150,140,140]),np.array([256,256,256]))
        mask_white_hsv = cv2.inRange(hsv_image,np.array([0,0,150]),np.array([170,40,256]))

        # White pixels above lane(thres_height) is considered to be a flower.
        mask = mask_white_bgr & mask_white_hsv
        mask[:thres_road_height,:] = 0

        
        return mask_white_bgr
    
    def draw_lines(self, img, lines=None, color=[0, 0, 255], thickness=7):
        line_img = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
        if lines is None:
            return
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
        return line_img
    
    # Temporarily commented
    # opencv used BGR system
    def draw_poly(self, img, poly_left, poly_right, min, max):
        left_img = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
        right_img = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
        for left_y in np.arange(min, max, 1):
            left_y = int(left_y)
            left_x = int(poly_left(left_y))
            cv2.line(left_img, (left_x, left_y), (left_x, left_y), color=[255, 0, 50], thickness=7)
        for right_y in np.arange(min, max, 1):
            right_y = int(right_y)
            right_x = int(poly_right(right_y))
            cv2.line(right_img, (right_x, right_y), (right_x, right_y), color=[0, 255, 50], thickness=7)
        line_img = self.weighted_img(right_img, left_img)
        return line_img

    # np.polyfit returns coefficients, highest power first
    def get_poly(self, left_line_y, left_line_x, right_line_y, right_line_x, deg):
        if deg == 1:
            poly_left_param = np.polyfit(left_line_y, left_line_x, deg=1)
            poly_right_param = np.polyfit(right_line_y, right_line_x, deg=1)
        else:
            poly_left_param = np.polyfit(left_line_y, left_line_x, deg=2)
            if abs(poly_left_param[0]) > 0.003 : ### FIX ME
                poly_left_param = np.polyfit(left_line_y, left_line_x, deg=1)
            poly_right_param = np.polyfit(right_line_y, right_line_x, deg=2)
            if abs(poly_right_param[0]) > 0.003 : ### FIX ME
                poly_right_param = np.polyfit(right_line_y, right_line_x, deg=1)

        poly_left = np.poly1d(poly_left_param)
        poly_right = np.poly1d(poly_right_param)

        return poly_left, poly_right

    def get_draw_center(self, img, poly_left, poly_right, draw=False):
        center = []
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

        for y in np.arange(self.min_y, self.max_y, 1):
            y = int(y)
            left_x = poly_left(y)
            right_x = poly_right(y)
            cen = int((left_x + right_x) / 2)
            center.extend([cen])
            cv2.line(line_img, (cen, y), (cen, y), color=[0, 0, 255], thickness=10)
        if draw == False:
            line_img = img
        return center, line_img

    # Used after lane detection
    def steering(self, center):
        right = 0
        left = 0
        for cen in center:
            diff = int(self.width/2) - cen
            if diff < 0:
                # print("go right")
                right += 1
            else:
                # print("go left")
                left += 1

            if right>left:
                steer = 'r'
            else:
                steer = 'l'
        return steer
    
    def lane(self, image):
        self.height, self.width = image.shape[:2]
        pre_image = self.preprocess(image)

        # lines = self.hough_transform(pre_image,rho=1,theta=np.pi/180,threshold=10,mll=10,mlg=20,mode="lineP")
        # line_image = self.draw_lines(image, lines, color=[0, 0, 255], thickness=5)
        # result = self.weighted_img(line_image, image, 0.8, 1.0, 0)

        return pre_image
