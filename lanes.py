import cv2
import numpy as np


def make_coordinates(image, line_params):
    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    try:
        left_line = make_coordinates(image, left_fit_average)
        right_line = make_coordinates(image, right_fit_average)
        return np.array([left_line, right_line])
    except Exception as e:
        print(e, '\n')
        return None


def roi(image):
    height = image.shape[0]
    triangle = np.array([[(200, height), (1100, height), (550, 250)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def canny(image):
    c_gray_road = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    c_blur = cv2.GaussianBlur(c_gray_road, (5, 5), 0)
    c_canny = cv2.Canny(c_blur, 50, 150)
    return c_canny


def display_lines(image, d_lines):
    d_line_image = np.zeros_like(image)
    if d_lines is not None:
        for line in d_lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(d_line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return d_line_image


cap = cv2.VideoCapture('test2.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()
    canny_image = canny(frame)
    road_roi = roi(canny_image)
    lines = cv2.HoughLinesP(road_roi, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap = 5)
    average_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, average_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('line detection', combo_image)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break



cap.release()
cv2.destroyAllWindows()
