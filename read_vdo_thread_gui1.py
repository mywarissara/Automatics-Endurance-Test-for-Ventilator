import sys, time, threading, cv2, csv
import queue as Queue
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from datetime import datetime
import numpy as np

IMG_FORMAT  = QImage.Format_RGB888
DISP_SCALE  = 4                 # Image scale
DISP_MSEC   = 50                # Display loops
VIDEO_PATH = "Images/eVentilator_VDO_5min.mp4"
area0 = 0
maxi = 0
mini = 0
max_area = 0
min_area = 0
avg =0 
h0 = 0
flag = 0
count = 0
area = 0
timemax = 0
timemin = 0
est0=0
Eest0=0
percent_change = 0

t0 = time.time()
camera_num  = 1
image_queue = Queue.Queue()
capturing_flag   = True

def grab_images(path, queue):
    cap = cv2.VideoCapture(path)
    while capturing_flag:
        if cap.grab():
            retval, image = cap.retrieve(0)
            if image is not None and queue.qsize() < 2:
                queue.put(image)
            else:
                time.sleep(0.05)
        else:
            print("Error: can't grab camera image")
            break
    cap.release()
    


class ImageWidget(QWidget):
    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        self.setMinimumSize(image.size())
        self.update()

    def paintEvent(self, event):
        qp = QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QPoint(0, 0), self.image)
        qp.end()


class MyWindow(QMainWindow):
    def __init__(self, parent=None):
        QMainWindow.__init__(self, parent)

        self.central     = QWidget(self)
        self.layout      = QVBoxLayout()        # Window layout

        # Layout display
        self.layout_disp = QHBoxLayout()
        self.disp        = ImageWidget(self)    
        self.layout_disp.addWidget(self.disp)

        # Layout menu
        self.layout_menu = QVBoxLayout()
        self.button1 = QPushButton('Start')
        self.button1.released.connect(self.on_button1_released)
        self.button2 = QPushButton('Stop')
        self.button2.released.connect(self.on_button2_released)
        self.layout_menu.addWidget(self.button1)
        self.layout_menu.addWidget(self.button2)

        self.layout.addLayout(self.layout_disp)
        self.layout.addLayout(self.layout_menu)
        self.central.setLayout(self.layout)
        self.setCentralWidget(self.central)

        exitAction = QAction('&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.triggered.connect(self.close)

    def start(self):
        self.timer = QTimer(self)           # Timer to trigger display
        self.timer.timeout.connect(lambda: 
                    self.show_image(image_queue, self.disp, DISP_SCALE))
        self.timer.start(DISP_MSEC)         
        self.capture_thread = threading.Thread(target=grab_images, 
                    args=(VIDEO_PATH, image_queue))
        self.capture_thread.start()         # Thread to grab images

    def stop(self):
        self.timer.stop()

    # Queue > display
    def show_image(self, imageq, display, scale):
        if not imageq.empty():
            image = imageq.get()
            image = self.process(image)
            if image is not None and len(image) > 0:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                self.display_image(img, display, scale)

    # Display
    def display_image(self, img, display, scale=1):
        disp_size = img.shape[1]//scale, img.shape[0]//scale
        disp_bpl = disp_size[0] * 3
        if scale > 1:
            img = cv2.resize(img, disp_size,interpolation=cv2.INTER_CUBIC)
        qimg = QImage(img.data, disp_size[0], disp_size[1], disp_bpl, IMG_FORMAT)
        display.setImage(qimg)

    # Processing
    def process(self, img):
        global area0,maxi,mini,count,t0,avg,flag,h0,minute,area,percent_change, timemax, timemin,est0,Eest0,max_area,min_area
        rowlist = []


        copy = img.copy()
        image = cv2.cvtColor(copy, cv2.COLOR_BGR2HSV)
        # image = cv2.cvtColor(copy, cv2.COLOR_BGR2RGB)
        # cropped = image[70:650, 350:840]
        # image = cv2.GaussianBlur(image, (9, 9 ), 10)
        lower_object_1 = np.array([(115 * 180)//240, (115 * 255)//240, (110 * 255)//240]) 
        upper_object_1 = np.array([(130 * 180)//240, 255, 255])
        mask = cv2.inRange(image,lower_object_1,upper_object_1)
        # mask = cv2.inRange(image,(35,120,140),(130, 250, 250))
        # kernel = np.ones((30,30), np.uint8)
        img_morp = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((25,25), np.uint8))
        img_morp = cv2.morphologyEx(img_morp, cv2.MORPH_CLOSE, np.ones((40,40), np.uint8))
        contours_b, hierarchy = cv2.findContours(img_morp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours_b:
            rect = cv2.boundingRect(c)
            x, y, w, h = rect #x y starting point / w h 
            Area = w * h
            
            if Area > 20000:
                cv2.rectangle(copy, (x, y), (x+w, y+h), (0,250,0), 5)
                # cv2.drawContours(img_morp, contours, -1, (0,255,0), 5)
                area = cv2.contourArea(c)
                M = cv2.moments(c)
                cx = int(M['m10']/M['m00']) 
                cy = int(M['m01']/M['m00'])
                cv2.circle(copy ,(cx,cy),10,(220,50,0),-1)

            minute = (time.time() - t0)/60

            if (h0 - h)>= 12:
                mini = min(area,area0)
                timemin = time.time()

            elif (h - h0) >= 12:
                maxi = max(area0, area)
                timemax = time.time()

            if (timemax - timemin)/ (maxi - mini) < 0 and flag == 0:
            # if area - area0 >=1000  and flag == 0:
                max_area = area
                flag = 1
            elif (timemax - timemin)/ (maxi - mini) > 0 and flag == 1:
            # elif area - area0 >=1000  and flag == 1:
                min_area = area
                count+=1
                avg = count/minute
               
                percent_change = abs(max_area - min_area)/maxi * 100
                est_percent = percent_change/count  # average value
                KGT = ( est_percent - percent_change ) / ( (est_percent - percent_change)+2); #kalman
                est = ( est0 ) + (KGT * ( percent_change - est0 ));
                Eest = ( Eest0 * (1 - KGT));
                est0 = est
                Eest0 = Eest

                stamp = datetime.now()
                array = [count, avg, stamp]
                rowlist.append(array)
            
                with open('bpm.csv', mode='a', newline='') as csv_file: #write
                    writer = csv.writer(csv_file,quoting=csv.QUOTE_NONNUMERIC, delimiter='|')
                    writer.writerow(rowlist) 
                flag = 0

            h0 = h
            area0 = area
            print(max_area)
            print(min_area)
            print("******")
            
            
        cv2.putText(copy, "Number of breathing: "+str(count)+" breaths.", (10,50), cv2.FONT_ITALIC, 1.5, (255,255,255), 3)
        cv2.putText(copy, "Average respiratory rate: "+ "{:.2f}".format(avg) +" breaths per minute.", (10,100), cv2.FONT_ITALIC, 1.5,(255,255,255), 3)
        cv2.putText(copy, "% of volumn changing: "+ "{:.2f}".format(percent_change) +"%", (10,150), cv2.FONT_ITALIC, 1.5,  (255,255,255), 3)
      
        return copy

    def on_button1_released(self):
        self.start()
    def on_button2_released(self):
        self.stop()

    def closeEvent(self, event):
        global capturing_flag
        capturing_flag = False
        self.capture_thread.join()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    win.setWindowTitle("Respirator pumping!")
    win.setWindowIcon(QIcon("tm.jpg"))
    sys.exit(app.exec_())
    
#EOF