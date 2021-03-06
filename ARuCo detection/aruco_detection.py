from imutils.video import VideoStream
from collections import deque
import numpy as np
import argparse
import imutils
import time
import cv2
import sys,os

ap = argparse.ArgumentParser()

ap.add_argument("-t", "--type", type=str,
   default="DICT_4X4_100",
   help="type of ArUCo tag to generate")

ap.add_argument("-b", "--buffer", type=int, default=128,
    help="max buffer size")

args = vars(ap.parse_args())

ARUCO_DICT = {
   "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
   "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
   "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
   "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
   "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
   "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
   "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
   "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
   "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
   "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
   "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
   "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
   "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
   "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
   "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
   "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
   "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
   "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
   "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
   "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
   "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# verify that the supplied ArUCo tag exists
if ARUCO_DICT.get(args["type"], None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported".format(
        args["type"]))
    sys.exit(0)
# load the ArUCo dictionary and grab the ArUCo parameters
print("[INFO] detecting '{}' tags...".format(args["type"]))
arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters_create()
# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")

# pts = deque(maxlen=args["buffer"])
# vs = VideoStream(src=2).start()
# time.sleep(2.0)

# loop over the frames from the video stream
# while True:
    # grab the frame from the threaded video stream and resize it
    
folder = os.path.normpath("/home/krekik/Documents/my-scripts/camera_save")
images = os.listdir(folder)
i = 0

for image in images:
    original = cv2.imread(os.path.join(folder,image))
    frame = original.copy()
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    (thresh, blackAndWhiteImage) = cv2.threshold(grayImage,50, 255, cv2.THRESH_BINARY)
    frame = blackAndWhiteImage

    (corners, ids, rejected) = cv2.aruco.detectMarkers(frame,arucoDict, parameters=arucoParams)

    # verify *at least* one ArUco marker was detected
    if len(corners) > 0:
        # flatten the ArUco IDs list
        ids = ids.flatten()
        # loop over the detected ArUCo corners
        for (markerCorner, markerID) in zip(corners, ids):
            # extract the marker corners (which are always returned
            # in top-left, top-right, bottom-right, and bottom-left
            # order)
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners
            # convert each of the (x, y)-coordinate pairs to integers
            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            # draw the bounding box of the ArUCo detection
            cv2.line(original, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(original, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(original, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(original, bottomLeft, topLeft, (0, 255, 0), 2)
            # compute and draw the center (x, y)-coordinates of the
            # ArUco marker
            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)
            center = (cX,cY)
            cv2.circle(original, (cX, cY), 1, (0, 0, 255), -1)
            # draw the ArUco marker ID on the frame
            out = f"ID: {str(markerID)} , X:{str(cX)},  Y:{str(cY)}"
            position = (10,50)
            cv2.putText(original, out ,position,	cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 2)

            # pts.appendleft(center)
            # for i in np.arange(1, len(pts)):
            #     if pts[i - 1] is None or pts[i] is None:
            #         continue

            #     thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
            #     cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        # show the output frame
    else:
        cv2.putText(original, "Marker not found!" ,(10,50),	cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 2)
    # frame = cv2.flip(frame, 1)
    screen_res = 1920, 720
    scale_width = screen_res[0] / frame.shape[1]
    scale_height = screen_res[1] / frame.shape[0]
    scale = min(scale_width, scale_height)
    #resized window width and height
    window_width = int(frame.shape[1] * scale)
    window_height = int(frame.shape[0] * scale)
    cv2.namedWindow('Resized Window', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Resized Window', window_width, window_height)

    # cv2.imshow('Resized Window', original)
    # key = cv2.waitKey(0)
    # # if the `q` key was pressed, break from the loop
    # # do a bit of cleanup
    # cv2.destroyAllWindows()
    cv2.imwrite(f"out_{image}",original)
    # vs.stop()