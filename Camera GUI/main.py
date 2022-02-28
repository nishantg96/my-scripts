from __future__ import print_function
from tk_utils import *
from imutils.video import VideoStream
import time

out = "/home/nishantg96/ZeMA/"

cam = VideoStream(src=2,resolution=(1280,720)).start()
cam2 = VideoStream(src=3,resolution=(1280,720)).start()

time.sleep(2.0)

app = camera_gui(cam,cam2,out)
app.root.mainloop()