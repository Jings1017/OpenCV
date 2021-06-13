import sys
from Ui_hw2 import Ui_MainWindow
import cv2 as cv
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QWidget
import numpy as np
import glob
from matplotlib import pyplot as plt

imgL = cv.imread('./Datasets/Q4_Image/imgL.png',0)
imgR = cv.imread('./Datasets/Q4_Image/imgR.png',0)
imgL= cv.resize(imgL, (705, 480), interpolation=cv.INTER_AREA)
imgR= cv.resize(imgR, (705, 480), interpolation=cv.INTER_AREA)
        
stereo = cv.StereoBM_create(numDisparities=80, blockSize=15)
disparity = stereo.compute(imgL,imgR)
disparity = cv.normalize(disparity, disparity, alpha=5, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
d=0
z=0
text = 'Disparity: '+str(d)+' pixels, Depth: '+str(z)+' mm'

def click_event(event, x, y, flags, params ): 
        # checking for left mouse clicks  
        global d,z,text
        if event == cv.EVENT_LBUTTONDOWN: 
    
            # displaying the coordinates on the Shell 
            #print(y, ' ', x) 
            baseline = 178
            focal_length = 2826
            crl = 123
            d = disparity[y][x]
            z = focal_length*baseline//(d+crl)
            
            text = 'Disparity: '+str(d)+' pixels, Depth: '+str(z)+' mm'
            cv.rectangle(disparity, (0, 450), (705, 480), (0, 0, 0), -1)
            cv.putText(disparity, text, (100, 470), cv.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), 1)
            cv.imshow('Q4', disparity)


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()
        for i in range(1,16):
            self.comboBox.addItem(str(i))

    def onBindingUI(self):
        self.pushButton_11.clicked.connect(self.on_btn_11_click)
        self.pushButton_12.clicked.connect(self.on_btn_12_click)
        self.pushButton_21.clicked.connect(self.on_btn_21_click)
        self.pushButton_22.clicked.connect(self.on_btn_22_click)
        self.pushButton_23.clicked.connect(self.on_btn_23_click)
        self.pushButton_24.clicked.connect(self.on_btn_24_click)
        self.pushButton_31.clicked.connect(self.on_btn_31_click)
        self.pushButton_41.clicked.connect(self.on_btn_41_click)
    
    def on_btn_11_click(self):
        print('clicked 11')
        ## coin01
        image1 = cv.imread('./Datasets/Q1_Image/coin01.jpg')
        gray1 = cv.cvtColor(image1,cv.COLOR_BGR2GRAY)
        blurred1 = cv.GaussianBlur(gray1, (11, 11), 0)
        canny1 = cv.Canny(blurred1, 30, 150)        
        cnts1, hierarchy1 = cv.findContours(canny1.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours1 = image1.copy()
        cv.drawContours(contours1, cnts1, -1, (0, 0, 255), 2)
        cv.imshow('coin01',contours1)

        ## coin02
        image2 = cv.imread('./Datasets/Q1_Image/coin02.jpg')
        gray2 = cv.cvtColor(image2,cv.COLOR_BGR2GRAY)
        blurred2 = cv.GaussianBlur(gray2, (11, 11), 0)
        canny2 = cv.Canny(blurred2, 30, 150)        
        cnts2, hierarchy2 = cv.findContours(canny2.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contours2 = image2.copy()
        cv.drawContours(contours2, cnts2, -1, (0, 0, 255), 2)
        cv.imshow("coin02", contours2)
        cv.waitKey(0)

    
    def on_btn_12_click(self):
        print('clicked 12')
        image01 = cv.imread('./Datasets/Q1_Image/coin01.jpg')
        gray01 = cv.cvtColor(image01,cv.COLOR_BGR2GRAY)
        blurred01 = cv.GaussianBlur(gray01, (11, 11), 0)
        canny01 = cv.Canny(blurred01, 30, 150)
        contours01, hierarchy = cv.findContours(canny01.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        image02 = cv.imread('./Datasets/Q1_Image/coin02.jpg')
        gray02 = cv.cvtColor(image02,cv.COLOR_BGR2GRAY)
        blurred02 = cv.GaussianBlur(gray02, (11, 11), 0)
        canny02 = cv.Canny(blurred02, 30, 150)
        contours02, hierarchy = cv.findContours(canny02.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        str1 = 'There are '+str(len(contours01))+' coins in coin01.jpg'
        str2 = 'There are '+str(len(contours02))+' coins in coin02.jpg'
        self.label_coin1.setText(str1)
        self.label_coin2.setText(str2)

    def on_btn_21_click(self):
        print('clicked 21')
        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((6*7,3), np.float32)
        objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)
        objpoints = []
        imgpoints = [] 
        images = glob.glob('./Datasets/Q2_Image/*.bmp')
        for fname in images:
            img = cv.imread(fname)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, (11,8), None)

            if ret == True:
                objpoints.append(objp)
                corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
                imgpoints.append(corners)
                cv.drawChessboardCorners(img, (11,8), corners2, ret)
                img= cv.resize(img, (705, 480), interpolation=cv.INTER_AREA)
                cv.imshow('img', img)
                cv.waitKey(500)
        cv.destroyAllWindows()

    def on_btn_22_click(self):
        print('clicked 22')
        obj_p = np.zeros((8 * 11, 3), np.float32)
        obj_p[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1 , 2)
        
        obj_ps, img_ps, mtx, dist = [], [], [], []
        
        for i in range(1, 16):
            filename = './Datasets/Q2_Image/' + str(i) + '.bmp'
            img = cv.imread(filename)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, (11, 8), None)
            if ret:
                obj_ps.append(obj_p)
                img_ps.append(corners)
		
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_ps, img_ps, gray.shape[::-1], None, None)

        print('Intrinsic matrix :')
        print(mtx)
    
    def on_btn_23_click(self):
        print('clicked 23')
        
        obj_p = np.zeros((8 * 11, 3), np.float32)
        obj_p[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1 , 2)
        
        obj_ps, img_ps, mtx, dist = [], [], [], []

        axis = np.float32([[1, 1, 0], [3, 5, 0], [5, 1, 0], [3, 3, -3]])
        index = self.comboBox.currentText()

        filename = './Datasets/Q2_Image/' + index + '.bmp'
        img = cv.imread(filename)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (11, 8), None)
        if ret:
            obj_ps.append(obj_p)
            img_ps.append(corners)
		
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_ps, img_ps, gray.shape[::-1], None, None)
        
        R , _ = cv.Rodrigues(rvecs[0])
        '''print('R')
        print(R)'''
        R = R.transpose()
        '''print('R t')
        print(R)'''
        R = np.hstack((R,tvecs[0]))
        print('Extrinsic Matrix')
        print(R)

    
    def on_btn_24_click(self):
        print('clicked 24')
        obj_p = np.zeros((8 * 11, 3), np.float32)
        obj_p[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1 , 2)
        
        obj_ps, img_ps, mtx, dist = [], [], [], []
        
        for i in range(1, 16):
            filename = './Datasets/Q2_Image/' + str(i) + '.bmp'
            img = cv.imread(filename)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, (11, 8), None)
            if ret:
                obj_ps.append(obj_p)
                img_ps.append(corners)
		
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_ps, img_ps, gray.shape[::-1], None, None)

        print('Distortion')
        print(dist)

    def on_btn_31_click(self):
        print('clicked 31')
        obj_p = np.zeros((8 * 11, 3), np.float32)
        obj_p[:,:2] = np.mgrid[0:11, 0:8].T.reshape(-1 , 2)
        
        obj_ps, img_ps, mtx, dist = [], [], [], []
        
        for i in range(1, 6):
            filename = './Datasets/Q3_Image/' + str(i) + '.bmp'
            img = cv.imread(filename)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, (11, 8), None)
            if ret:
                obj_ps.append(obj_p)
                img_ps.append(corners)
		
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_ps, img_ps, gray.shape[::-1], None, None)

        axis = np.float32([[1, 1, 0], [3, 5, 0], [5, 1, 0], [3, 3, -3]])
		
        files, imgs = [], []
        for i in range(1, 6):
            files.append('./Datasets/Q3_Image/'+ str(i) + '.bmp')

        for h, filename in enumerate(files):
            img = cv.imread(filename)
            gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, (11, 8), None)
            if ret:	
                imgpts, jac = cv.projectPoints(axis, rvecs[h], tvecs[h], mtx, dist)

                p1 = tuple(imgpts[0].ravel())
                p2 = tuple(imgpts[1].ravel())
                p3 = tuple(imgpts[2].ravel())
                p4 = tuple(imgpts[3].ravel()) # corner
				
                img = cv.line(img, p1, p2, (0, 0, 255), 10)
                img = cv.line(img, p2, p3, (0, 0, 255), 10)
                img = cv.line(img, p1, p3, (0, 0, 255), 10)
                img = cv.line(img, p1, p4, (0, 0, 255), 10)
                img = cv.line(img, p2, p4, (0, 0, 255), 10)
                img = cv.line(img, p3, p4, (0, 0, 255), 10)
                
                imgs.append(img)	
                height, width = img.shape[:2]

        output = cv.VideoWriter('Q3.avi', cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 2, (width, height))
        
        for i in range(len(imgs)):
            output.write(imgs[i])
        output.release()	
        
        cap = cv.VideoCapture('Q3.avi')

        if (cap.isOpened() == False):
            print('Already opened! ')
            
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                cv.namedWindow('Q3', cv.WINDOW_NORMAL)
                cv.imshow('Q3', frame)
                if cv.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                break
            
        cap.release()
        cv.destroyAllWindows()
    
    def on_btn_41_click(self):
        print('clicked 41')
        cv.namedWindow('Q4')
        cv.setMouseCallback('Q4', click_event)
        cv.imshow('Q4', disparity)
        cv.waitKey(0)
        cv.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())