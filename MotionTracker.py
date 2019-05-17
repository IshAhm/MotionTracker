from tkinter import *
from tkinter import filedialog
import pandas as pd
import cv2
import numpy as np
import subprocess


# function to set the values from the GUI
def set_val():
    global theShape
    global min_area
    global max_area
    global vidLen
    theShape = shape.get()
    min_area = int(e1.get())
    max_area = int(e2.get())
    vidLen = int(e3.get())


# The GUI
filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                      filetypes=(("video files", "*.mp4"), ("all files", "*.*")))
m = Tk()
m.title('Top-down Tracker')
shape = IntVar()
Label(m, text='Minimum Area (in pixels)').grid(row=0)
Label(m, text='Maximum Area (in pixels)').grid(row=1)
Label(m, text='Length of Video').grid(row=2)
e1 = Entry(m)
e2 = Entry(m)
e3 = Entry(m)
e1.grid(row=0, column=1)
e2.grid(row=1, column=1)
e3.grid(row=2, column=1)
Button(m, text="Set the Values", command=set_val).grid(row=5, column=1)
Button(m, text="Start", command=m.destroy).grid(row=6, column=1)
mainloop()

cap = cv2.VideoCapture(filename)
fgbg = cv2.createBackgroundSubtractorMOG2()

lst = []
ret, first = cap.read()

# Select region of interest
r = cv2.selectROI(first)
# Crop image
imCrop = first[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
while 1:
    cv2.imshow("Image", imCrop)
    k = cv2.waitKey(33)
    if k == 27:    # Esc key to stop
        break

while 1:
    ret, frame = cap.read()
    if frame is None:  # problem of empty frames, which causes an assertion error. so we just skip the empty frames
        break
    blur = cv2.GaussianBlur(frame, (5, 5), 0)  # apply gaussian blur
    crop_blur = blur[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    fgmask = fgbg.apply(crop_blur)  # apply the background subtractor

    screen_res = 1280., 720.  # nifty code to resize the windows so it fits on pretty much all screens
    scale_width = screen_res[0] / cap.get(3)
    scale_height = screen_res[1] / cap.get(4)
    scale = min(scale_width, scale_height)
    window_width = int(cap.get(3) * scale)
    window_height = int(cap.get(4) * scale)
    cv2.namedWindow('res', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('res', window_width, window_height)

    contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        area = cv2.contourArea(c)
        if area in range(min_area, max_area):
            M = cv2.moments(c)
            if M["m00"] != 0:  # sometimes image moment is zero, hand waivy fix
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = (0, 0)
            lst.append((cX, cY))
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
            cv2.circle(frame, (cX, cY), 7, (255, 255, 255), -1)
            cv2.putText(frame, "center", (cX - 20, cY - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.imshow('res', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:     # break out with esc
        break

cap.release()
cv2.destroyAllWindows()

x_val = [x[0] for x in lst]
y_val = [x[1] for x in lst]
times = np.linspace(0, vidLen, len(x_val))
data = [x_val, y_val, times]
dataPan = pd.DataFrame(data)
dataPanTrans = dataPan.T
dataPanTrans.to_csv('mycoords.csv', header=False, index=False)

# R subprocess to get the plots
command = 'Rscript'
path2script = 'C:/Users/where ever it is/traj.r'
subprocess.call([command, path2script], shell=True)
