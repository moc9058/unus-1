import math
import sys
import cv2
import numpy as np
np.set_printoptions(threshold=sys.maxsize, linewidth=150)

class libLANE(object):
    def __init__(self):
        self.height = 0
        self.width = 0
        self.min_y = 0
        self.mid_y_1 = 0
        self.mid_y_2 = 0
        self.max_y = 0
        self.match_mask_color = 255
    def region_of_interest(self, img, vertices):
        mask = np.zeros_like(img)
        if len(img.shape) > 2:
            self.match_mask_color = (255,255,255)
        cv2.fillPoly(mask, vertices, self.match_mask_color)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image
    def weighted_img(self, img, initial_img, α=1, β=1., λ=0.):
        return cv2.addWeighted(initial_img, α, img, β, λ)
    def preprocess(self, img):
        region_of_interest_vertices_1 = np.array(
            [[(50, self.height), (self.width / 2 - 45, self.height / 2 + 60),
              (self.width / 2 + 45, self.height / 2 + 60), (self.width - 50, self.height)]],
            dtype=np.int32) ### FIX ME
        region_of_interest_vertices_2 = [(0, self.height), (self.width / 2, self.height / 2),
                                         (self.width, self.height), ]

        gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        blur_image = cv2.GaussianBlur(gray_image, (3, 3), 0)
        # cannyed_image = cv2.Canny(blur_image, 100, 200)
        canny_image = cv2.Canny(blur_image, 70, 210)

        cropped_image = self.region_of_interest(canny_image, np.array([region_of_interest_vertices_1], np.int32), )

        return cropped_image
    def draw_lines(self, img, lines, color=[0, 0, 255], thickness=2):
        line_img = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)

        if lines is None:
            return
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

        return line_img
    def draw_poly(self, img, poly_left, poly_right, min, max):
        left_img = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
        right_img = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)

        for left_y in np.arange(min, max, 1):
            left_y = int(left_y)
            left_x = int(poly_left(left_y))
            cv2.line(left_img, (left_x, left_y), (left_x, left_y), color=[255, 0, 50], thickness=5)
        for right_y in np.arange(min, max, 1):
            right_y = int(right_y)
            right_x = int(poly_right(right_y))
            cv2.line(right_img, (right_x, right_y), (right_x, right_y), color=[0, 255, 50], thickness=5)

        line_img = self.weighted_img(right_img, left_img)
        return line_img
    def get_poly(self, left_line_y, left_line_x, right_line_y, right_line_x, deg):
        if deg == 1:
            poly_left_param = np.polyfit(
                left_line_y,
                left_line_x,
                deg=1
            )
            poly_right_param = np.polyfit(
                right_line_y,
                right_line_x,
                deg=1
            )
        else:
            poly_left_param = np.polyfit(
                left_line_y,
                left_line_x,
                deg=2
            )
            if abs(poly_left_param[0]) > 0.001 : ### FIX ME
                poly_left_param = np.polyfit(
                    left_line_y,
                    left_line_x,
                    deg=1
                )
            poly_right_param = np.polyfit(
                right_line_y,
                right_line_x,
                deg=2
            )
            if abs(poly_right_param[0]) > 0.001 : ### FIX ME
                poly_right_param = np.polyfit(
                    right_line_y,
                    right_line_x,
                    deg=1
                )

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
            cv2.line(line_img, (cen, y), (cen, y), color=[0, 0, 255], thickness=3)
        if draw == False:
            line_img = img
        return center, line_img
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
        cropped_image = self.preprocess(image)
        self.min_y = int(image.shape[0] * (6 / 10))
        self.mid_y_1 = int(image.shape[0] * (13 / 20))
        self.mid_y_2 = int(image.shape[0] * (17 / 20))
        self.max_y = int(image.shape[0])

        lines_1 = cv2.HoughLinesP(
            cropped_image,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            lines=np.array([]),
            minLineLength=10,
            maxLineGap=20
        )
        lines_2 = cv2.HoughLinesP( ### option2
            cropped_image,
            rho=6,
            theta=np.pi / 60,
            threshold=160,
            lines=np.array([]),
            minLineLength=40,
            maxLineGap=25
        )

        left_line_x_1 = []
        left_line_y_1 = []
        right_line_x_1 = []
        right_line_y_1 = []
        left_line_x_2 = []
        left_line_y_2 = []
        right_line_x_2 = []
        right_line_y_2 = []

        for line in lines_1:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1)
                if np.abs(slope) < 0.5: # stop line
                    continue
                if slope <= 0:
                    if y1 > self.mid_y_1:
                        left_line_x_1.extend([x1, x2])
                        left_line_y_1.extend([y1, y2])
                    if y1 < self.mid_y_2:
                        left_line_x_2.extend([x1, x2])
                        left_line_y_2.extend([y1, y2])
                else:
                    if y1 > self.mid_y_1:
                        right_line_x_1.extend([x1, x2])
                        right_line_y_1.extend([y1, y2])
                    if y1 < self.mid_y_2:
                        right_line_x_2.extend([x1, x2])
                        right_line_y_2.extend([y1, y2])

        # Drawing POLY (deg=2)
        if len(left_line_y_2) != 0 and len(right_line_y_2) != 0:
            poly_left_2, poly_right_2 = self.get_poly(left_line_y_2, left_line_x_2, right_line_y_2, right_line_x_2, 2)
            center2, center2_image = self.get_draw_center(image, poly_left_2, poly_right_2, False)
            poly2_image = self.draw_poly(image, poly_left_2, poly_right_2, self.min_y, self.mid_y_2)
            #line_image = self.weighted_img(center2_image, poly2_image)
            image = self.weighted_img(poly2_image, image, 0.8, 1.0, 0)

            future_steer = self.steering(center2)

        # Drawing LINE (deg=1)
        if len(left_line_y_1) != 0 and len(right_line_y_1) != 0:
            poly_left_1, poly_right_1 = self.get_poly(left_line_y_1, left_line_x_1, right_line_y_1, right_line_x_1, 1)
            center1, center1_image = self.get_draw_center(image, poly_left_1, poly_right_1, True)

            left_x_start = int(poly_left_1(self.max_y))
            left_x_end = int(poly_left_1(self.mid_y_1))
            right_x_start = int(poly_right_1(self.max_y))
            right_x_end = int(poly_right_1(self.mid_y_1))
            left_line_image = self.draw_lines(image,[[[left_x_start, self.max_y, left_x_end, self.mid_y_1],]],color=[255,0,0],thickness=5,)
            right_line_image = self.draw_lines(image,[[[right_x_start, self.max_y, right_x_end, self.mid_y_1],]],color=[0,255,0],thickness=5,)

            poly1_image = self.weighted_img(left_line_image, right_line_image)
            line_image = self.weighted_img(center1_image, poly1_image)
            result = self.weighted_img(line_image, image, 0.8, 1.0, 0)

            current_steer = self.steering(center1)
        else:
            result = image


        return result