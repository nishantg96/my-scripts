import numpy as np
import cv2

import rospy
from nav_msgs.msg import Odometry

aruco_type = cv2.aruco.DICT_4X4_100
aruco_dict = cv2.aruco.Dictionary_get(aruco_type)
aruco_params = cv2.aruco.DetectorParameters_create()

#============================================================================
import argparse
import os
from pathlib import Path
from tkinter.messagebox import NO
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages,letterbox
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, plot_one_box, strip_optimizer)
from utils.torch_utils import select_device, load_classifier, time_synchronized

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='/home/krekik/RICAIP/best.pt', help='model.pt path(s)')
parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
parser.add_argument('--img-size', type=int, default=1280, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.12, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.1, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
opt = parser.parse_args()

out, source, weights, view_img, save_txt, imgsz = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
device = select_device(opt.device)
half = device.type != 'cpu'
img_size = 1280
imgs = [None]
model = attempt_load(weights, map_location=device)  # load FP32 model
imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
if half:
    model.half()

names = model.module.names if hasattr(model, 'module') else model.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
#================================================================================

def get_aruco_point(frame):

    # grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    # (thresh, frame) = cv2.threshold(grayImage,150, 255, cv2.THRESH_BINARY)
    # detect ArUco markers in the input frame
    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame,aruco_dict, parameters=aruco_params)

    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):

            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv2.line(frame, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(frame, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(frame, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(frame, bottomLeft, topLeft, (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            center = (cX,cY)
            cv2.circle(frame, (cX, cY), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(markerID),(topLeft[0], topLeft[1] - 15),	cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 2)
            
            out = f"ID: {str(markerID)} , X:{str(cX)},  Y:{str(cY)}"
            return frame,[markerID,cX,cY]
    else:
        return frame,[]

def get_pose():
	# rospy.init_node('myNodeName')
	msg = rospy.wait_for_message("/odom", Odometry, 10)
	position = msg.pose.pose.position
	pos_x = position.x
	pos_y = position.y
	pos_z = position.z
	orientation = msg.pose.pose.orientation
	or_x = orientation.x
	or_y = orientation.y
	or_z = orientation.z
	or_w = orientation.w

	pos_list = [pos_x,pos_y,pos_z]
	or_list = [or_x,or_y,or_z,or_w]

	return pos_list, or_list

def detect_object(image):
    img0 = image.copy()
    img = letterbox(img0, new_shape=img_size)[0]
    # print(img.shape)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    # print(img.shape)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    t1 = time_synchronized()
    pred = model(img, augment=opt.augment)[0]

    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t2 = time_synchronized()
    
    for i, det in enumerate(pred):
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
            
            box = det[:, :4][0].cpu().numpy()
            cx = int((box[0]+box[2])/2)
            cy = int((box[1]+box[3])/2)
            cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
            c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            cv2.rectangle(image, c1, c2, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
            return image,[cx,cy]

        else:
            return image,[]
        
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img