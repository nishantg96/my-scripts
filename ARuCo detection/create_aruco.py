import numpy as np
import argparse
import cv2 as cv
import sys

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,
   help="path to output image containing ArUCo tag")

ap.add_argument("-i", "--id", type=int,
   help="ID of ArUCo tag to generate")

ap.add_argument("-t", "--type", type=str,
   default="DICT_ARUCO_ORIGINAL",
   help="type of ArUCo tag to generate")

ap.add_argument("-r", "--range", type=str,
   help="range of ids of ArUCo tag to generate. Separate the range using a comma(,)")

args = vars(ap.parse_args())

ARUCO_DICT = {
   "DICT_4X4_50": cv.aruco.DICT_4X4_50,
   "DICT_4X4_100": cv.aruco.DICT_4X4_100,
   "DICT_4X4_250": cv.aruco.DICT_4X4_250,
   "DICT_4X4_1000": cv.aruco.DICT_4X4_1000,
   "DICT_5X5_50": cv.aruco.DICT_5X5_50,
   "DICT_5X5_100": cv.aruco.DICT_5X5_100,
   "DICT_5X5_250": cv.aruco.DICT_5X5_250,
   "DICT_5X5_1000": cv.aruco.DICT_5X5_1000,
   "DICT_6X6_50": cv.aruco.DICT_6X6_50,
   "DICT_6X6_100": cv.aruco.DICT_6X6_100,
   "DICT_6X6_250": cv.aruco.DICT_6X6_250,
   "DICT_6X6_1000": cv.aruco.DICT_6X6_1000,
   "DICT_7X7_50": cv.aruco.DICT_7X7_50,
   "DICT_7X7_100": cv.aruco.DICT_7X7_100,
   "DICT_7X7_250": cv.aruco.DICT_7X7_250,
   "DICT_7X7_1000": cv.aruco.DICT_7X7_1000,
   "DICT_ARUCO_ORIGINAL": cv.aruco.DICT_ARUCO_ORIGINAL,
   "DICT_APRILTAG_16h5": cv.aruco.DICT_APRILTAG_16h5,
   "DICT_APRILTAG_25h9": cv.aruco.DICT_APRILTAG_25h9,
   "DICT_APRILTAG_36h10": cv.aruco.DICT_APRILTAG_36h10,
   "DICT_APRILTAG_36h11": cv.aruco.DICT_APRILTAG_36h11
}

# verify the type of ArUCo tag
if ARUCO_DICT.get(args["type"], None) is None:
   print("[INFO] ArUCo tag of '{}' is not supported".format(args["type"]))
   sys.exit(0)
# load the ArUCo dictionary
arucoDict = cv.aruco.Dictionary_get(ARUCO_DICT[args["type"]])

if args["id"] is not None:
   print("[INFO] generating ArUCo tag type '{}' with ID '{}'".format(
   args["type"], args["id"]))

   tag = np.zeros((500, 500, 1), dtype="uint8")
   cv.aruco.drawMarker(arucoDict, args["id"], 500, tag, 1)

   dest = args["output"] + "/" + str(args["id"]) + ".png"
   cv.imwrite(dest, tag)
   cv.imshow("ArUCo Tag", tag)
   cv.waitKey(0)

if args["range"] is not None:
   print("[INFO] generating ArUCo tag type '{}' from range '{}'".format(
      args["type"], args["range"]))

   rng = args["range"].split(",")
   for i in range(int(rng[0]),int(rng[1])+1):
       tag = np.zeros((500, 500, 1), dtype="uint8")
       cv.aruco.drawMarker(arucoDict, i , 500, tag, 1)
       
       dest = args["output"] + "/" + str(i) + ".png"
       cv.imwrite(dest, tag)
       print("Tags written to {}".format(dest))
