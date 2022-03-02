from __future__ import print_function
from tk_utils import *
from imutils.video import VideoStream
import time,subprocess

out = "/home/nishantg96/ZeMA/"
rospy.init_node('myNodeName')

width = 1280
height = 720

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

cap2 = cv2.VideoCapture(2, cv2.CAP_V4L2)
cap2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap2.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

mypid = os.getpid()

time.sleep(2.0)

app = camera_gui(cap,cap2,out)
app.root.mainloop()
print("Main Exited Successfully")

com = "pkill -9 -f main.py"
subprocess.Popen(com, stdout = subprocess.PIPE, shell = True)