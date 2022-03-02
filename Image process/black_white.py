import os
import cv2
import numpy as np

def process_image(image):
    im = cv2.imread(image)
    grayImage = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)    
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage,50, 255, cv2.THRESH_BINARY)
    cv2.imshow('result.png',blackAndWhiteImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image = "/home/krekik/Documents/my-scripts/camera_save/default_camera_2_link_camera_2_sensor(2)-0033.jpg"
    
    process_image(image)