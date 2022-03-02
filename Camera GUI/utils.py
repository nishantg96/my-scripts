import numpy as np
import cv2

import rospy
from nav_msgs.msg import Odometry

aruco_type = cv2.aruco.DICT_4X4_100
aruco_dict = cv2.aruco.Dictionary_get(aruco_type)
aruco_params = cv2.aruco.DetectorParameters_create()

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
