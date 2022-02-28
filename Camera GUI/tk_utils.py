from __future__ import print_function
from PIL import Image,ImageTk
import tkinter as tki
import threading
import datetime
import imutils
import cv2
import os

class camera_gui:
    def __init__(self, cam1,cam2, out_path):
        self.cam1 = cam1
        self.cam2 = cam2
        self.out_path = out_path
        self.frame = None
        self.frame2 = None
        self.root = tki.Tk()
        self.panel = None
        self.panel1 = None
        self.thread = None
        self.stopEvent = None
        
        btn = tki.Button(self.root, text="Snapshot!",command=self.takeSnapshot)
        btn.pack(side="bottom", fill="both", expand="yes", padx=10,pady=10)

        self.stopEvent = threading.Event()
        self.thread = threading.Thread(target=self.video_loop, args=())
        self.thread.start()

        self.root.wm_title("ArUCO Detector!")
        self.root.wm_protocol("WM_DELETE_WINDOW", self.onClose)

    
    def video_loop(self):
        try:
            while not self.stopEvent.is_set():

                self.frame = self.cam1.read()
                self.frame2 = self.cam2.read()
                self.frame = imutils.resize(self.frame,width=650)
                self.frame2 = imutils.resize(self.frame2,width=500)

                image = cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = ImageTk.PhotoImage(image)

                image2 = cv2.cvtColor(self.frame2,cv2.COLOR_BGR2RGB)
                image2 = Image.fromarray(image2)
                image2 = ImageTk.PhotoImage(image2)

                if self.panel is None:
                    self.panel = tki.Label(image=image)
                    self.panel.image = image
                    self.panel.pack(side="left", padx=10, pady=10)

                    self.panel1 = tki.Label(image=image2)
                    self.panel1.image = image2
                    self.panel1.pack(side="right", padx=10, pady=10)
        
                # otherwise, simply update the panel
                else:
                    self.panel.configure(image=image)
                    self.panel.image = image

                    self.panel1.configure(image=image2)
                    self.panel1.image = image2
        except RuntimeError:
            print("[INFO] caught a RuntimeError")


    def takeSnapshot(self):

        ts = datetime.datetime.now()
        file = "camera_0_{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        file2 = "camera_1_{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
        p = os.path.sep.join((self.out_path, file))
        p1 = os.path.sep.join((self.out_path, file2))
        cv2.imwrite(p, self.frame.copy())
        cv2.imwrite(p1, self.frame2.copy())
        print("[INFO] saved {} and {}".format(file,file2))

    def onClose(self):
        print("[INFO] closing...")
        self.stopEvent.set()
        self.cam1.stop()
        self.cam2.stop()
        self.root.quit()