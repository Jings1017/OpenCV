from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton
import numpy as np
import sys
import cv2

def nothing(x):
        pass

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1080, 480)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setGeometry(QtCore.QRect(20, 40, 191, 321))
        self.groupBox.setObjectName("groupBox")
        self.pushButton_11 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_11.setGeometry(QtCore.QRect(20, 40, 151, 41))
        self.pushButton_11.setCheckable(False)
        self.pushButton_11.setAutoDefault(False)
        self.pushButton_11.setObjectName("pushButton_11")
        self.pushButton_11.clicked.connect(self.clicked11)

        self.pushButton_12 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_12.setGeometry(QtCore.QRect(20, 110, 151, 41))
        self.pushButton_12.setCheckable(False)
        self.pushButton_12.setAutoDefault(False)
        self.pushButton_12.setObjectName("pushButton_12")
        self.pushButton_12.clicked.connect(self.clicked12)

        self.pushButton_13 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_13.setGeometry(QtCore.QRect(20, 180, 151, 41))
        self.pushButton_13.setCheckable(False)
        self.pushButton_13.setAutoDefault(False)
        self.pushButton_13.setObjectName("pushButton_13")
        self.pushButton_13.clicked.connect(self.clicked13)

        self.pushButton_16 = QtWidgets.QPushButton(self.groupBox)
        self.pushButton_16.setGeometry(QtCore.QRect(20, 250, 151, 41))
        self.pushButton_16.setCheckable(False)
        self.pushButton_16.setAutoDefault(False)
        self.pushButton_16.setObjectName("pushButton_16")
        self.pushButton_16.clicked.connect(self.clicked14)

        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(240, 40, 181, 321))
        self.groupBox_2.setObjectName("groupBox_2")
        self.pushButton_21 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_21.setGeometry(QtCore.QRect(20, 50, 141, 41))
        self.pushButton_21.setObjectName("pushButton_21")
        self.pushButton_21.clicked.connect(self.clicked21)

        self.pushButton_22 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_22.setGeometry(QtCore.QRect(20, 150, 141, 41))
        self.pushButton_22.setObjectName("pushButton_22")
        self.pushButton_22.clicked.connect(self.clicked22)

        self.pushButton_23 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_23.setGeometry(QtCore.QRect(20, 250, 141, 41))
        self.pushButton_23.setObjectName("pushButton_23")
        self.pushButton_23.clicked.connect(self.clicked23)

        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setGeometry(QtCore.QRect(450, 40, 211, 321))
        self.groupBox_3.setObjectName("groupBox_3")
        self.pushButton_31 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_31.setGeometry(QtCore.QRect(30, 40, 151, 41))
        self.pushButton_31.setObjectName("pushButton_31")
        self.pushButton_31.clicked.connect(self.clicked31)

        self.pushButton_32 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_32.setGeometry(QtCore.QRect(30, 110, 151, 41))
        self.pushButton_32.setObjectName("pushButton_32")
        self.pushButton_32.clicked.connect(self.clicked32)

        self.pushButton_33 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_33.setGeometry(QtCore.QRect(30, 180, 151, 41))
        self.pushButton_33.setObjectName("pushButton_33")
        self.pushButton_33.clicked.connect(self.clicked33)

        self.pushButton_34 = QtWidgets.QPushButton(self.groupBox_3)
        self.pushButton_34.setGeometry(QtCore.QRect(30, 250, 151, 41))
        self.pushButton_34.setObjectName("pushButton_34")
        self.pushButton_34.clicked.connect(self.clicked34)

        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setGeometry(QtCore.QRect(690, 40, 291, 321))
        self.groupBox_4.setObjectName("groupBox_4")
        self.label = QtWidgets.QLabel(self.groupBox_4)
        self.label.setGeometry(QtCore.QRect(20, 60, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(12)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.rotation_label = QtWidgets.QTextEdit(self.groupBox_4)
        self.rotation_label.setGeometry(QtCore.QRect(100, 60, 121, 31))
        self.rotation_label.setObjectName("rotation_label")
        self.label_2 = QtWidgets.QLabel(self.groupBox_4)
        self.label_2.setGeometry(QtCore.QRect(230, 70, 58, 15))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.groupBox_4)
        self.label_3.setGeometry(QtCore.QRect(20, 120, 71, 16))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.scaling_label = QtWidgets.QTextEdit(self.groupBox_4)
        self.scaling_label.setGeometry(QtCore.QRect(100, 110, 121, 31))
        self.scaling_label.setObjectName("scaling_label")
        self.label_4 = QtWidgets.QLabel(self.groupBox_4)
        self.label_4.setGeometry(QtCore.QRect(20, 170, 71, 16))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.groupBox_4)
        self.label_5.setGeometry(QtCore.QRect(20, 210, 71, 16))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.tx_label = QtWidgets.QTextEdit(self.groupBox_4)
        self.tx_label.setGeometry(QtCore.QRect(100, 160, 121, 31))
        self.tx_label.setObjectName("tx_label")
        self.ty_label = QtWidgets.QTextEdit(self.groupBox_4)
        self.ty_label.setGeometry(QtCore.QRect(100, 200, 121, 31))
        self.ty_label.setObjectName("ty_label")
        self.label_6 = QtWidgets.QLabel(self.groupBox_4)
        self.label_6.setGeometry(QtCore.QRect(230, 170, 58, 15))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(12)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.groupBox_4)
        self.label_7.setGeometry(QtCore.QRect(230, 210, 58, 15))
        font = QtGui.QFont()
        font.setFamily("Adobe Devanagari")
        font.setPointSize(12)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.pushButton_41 = QtWidgets.QPushButton(self.groupBox_4)
        self.pushButton_41.setGeometry(QtCore.QRect(20, 260, 251, 28))
        self.pushButton_41.setObjectName("pushButton_41")
        self.pushButton_41.clicked.connect(self.clicked41)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1034, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    ## problem 1 
    ### load image file
    def clicked11(self):
        img = cv2.imread('./Q1_Image/Uncle_Roger.jpg')
        height,width = img.shape[:2]
        print('Height:  ',height)
        print('Width:   ',width)
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    ### color separation
    def clicked12(self):
        img = cv2.imread('./Q1_Image/Uncle_Roger.jpg')
        ## blue
        b = img.copy()
        b[:, :, 1] = 0
        b[:, :, 2] = 0
        cv2.imshow('blue',b)
        ## red 
        r = img.copy()
        r[:, :, 1] = 0
        r[:, :, 0] = 0
        cv2.imshow('red',r)
        ## green
        g = img.copy()
        g[:, :, 2] = 0
        g[:, :, 0] = 0
        cv2.imshow('green',g)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ### image flipping
    def clicked13(self):
        img = cv2.imread('./Q1_Image/Uncle_Roger.jpg')
        img_flip = cv2.flip(img,1)
        cv2.imshow('Original image',img)
        cv2.imshow('Result',img_flip)
        cv2.imwrite('result.jpg',img_flip)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    ### blending
    def clicked14(self):
        cv2.namedWindow('Blend')
        cv2.createTrackbar('blend','Blend',0,100,nothing)
        img1 = cv2.imread('./Q1_Image/Uncle_Roger.jpg')
        img2 = cv2.imread('./Q1_Image/result.jpg')
        while(1):
            k = cv2.waitKey(1)
            if k == 27:
                break
            ratio = cv2.getTrackbarPos('blend', 'Blend') / 100
            result = cv2.addWeighted(img1, 1-ratio, img2, ratio, 0.0)
            cv2.imshow('Blend', result)
        cv2.destroyAllWindows()

    ## problem 2
    ### median filter
    def clicked21(self):
        image = cv2.imread('./Q2_Image/Cat.png')
        cv2.imshow("Original", image)
        blurred = cv2.medianBlur(image, 7)
        cv2.namedWindow("Median")
        cv2.imshow('Median', blurred)
        cv2.waitKey(0)
    
    ### guassian blur
    def clicked22(self):
        image = cv2.imread('./Q2_Image/Cat.png')
        cv2.imshow("Original", image)
        blurred = cv2.GaussianBlur(image, (3,3), 0)
        cv2.namedWindow("Gaussian")
        cv2.imshow('Gaussian', blurred)
        cv2.waitKey(0)
    
    ### bilateral filter
    def clicked23(self):
        image = cv2.imread('./Q2_Image/Cat.png')
        cv2.imshow("Original", image)
        blurred = cv2.bilateralFilter(image, 9,90,90)
        cv2.namedWindow("Bilateral")
        cv2.imshow('Bilateral', blurred)
        cv2.waitKey(0)

    ## problem 3
    ### guassian blur
    def clicked31(self):
        origin_img = cv2.imread('./Q3_Image/Chihiro.jpg')
        cv2.imshow('Origin', origin_img)
        
        gray = self.rgb2gray(origin_img)
        cv2.imshow('Grayscale',gray)

        gaussian_kernel = self.gaussian_smooth()
        gaussian = self.convolution(gray, gaussian_kernel)
        cv2.imshow('Gaussian Blur', gaussian)
        cv2.waitKey(0)

    ### Sobel X
    def clicked32(self):
        origin_img = cv2.imread('./Q3_Image/Chihiro.jpg')
        cv2.imshow('Origin', origin_img)
        
        gray = self.rgb2gray(origin_img)
        gaussian_kernel = self.gaussian_smooth()
        smooth = self.convolution(gray, gaussian_kernel)
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        result = self.convolution(smooth, Gx)
        cv2.imshow('Sobel X', result)
    
    ### Sobel Y
    def clicked33(self):
        origin_img = cv2.imread('./Q3_Image/Chihiro.jpg')
        cv2.imshow('Origin', origin_img)
        gray = self.rgb2gray(origin_img)
        gaussian_kernel = self.gaussian_smooth()
        smooth = self.convolution(gray, gaussian_kernel)
        Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        result = self.convolution(smooth, Gy)
        cv2.imshow('Sobel Y', result)
    
    ### magnitude
    def clicked34(self):
        origin_img = cv2.imread('./Q3_Image/Chihiro.jpg')
        cv2.imshow('Origin', origin_img)
        gray = self.rgb2gray(origin_img)
        gaussian_kernel = self.gaussian_smooth()
        smooth = self.convolution(gray, gaussian_kernel)
        Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
        result_x = self.convolution(smooth, Gx)
        result_y = self.convolution(smooth, Gy)
        magnitude_img = np.zeros_like(smooth)
        magnitude_img = np.sqrt(result_x ** 2 + result_y ** 2)
        cv2.imshow('Magnitude', magnitude_img)

    ## problem 4
    ### transformation
    def clicked41(self):
        img = cv2.imread('./Q4_Image/Parrot.png')
        angle = float(self.rotation_label.toPlainText())
        scale = float(self.scaling_label.toPlainText())
        tx = float(self.tx_label.toPlainText())
        ty = float(self.ty_label.toPlainText())
        rows, cols = img.shape[:2]
        center = (cols//2 , rows//2)
        R = cv2.getRotationMatrix2D(center, angle, scale)
        result = cv2.warpAffine(img, R, (cols, rows))
        T = np.float32([[1, 0, tx], [0, 1, ty]])
        result = cv2.warpAffine(result, T, (cols, rows))
        cv2.imshow('Transformation', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # image operations
    def rgb2gray(self,origin_img):
        r,g,b = origin_img[:,:,0], origin_img[:,:,1], origin_img[:,:,2]
        gray = 0.2989*r/255 + 0.5870*g/255 + 0.1140*b/255
        return gray

    def convolution(self,gray, gaussian_kernel):
        result = np.zeros_like(gray)
        image_padded = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2))
        image_padded[1:-1, 1:-1] = gray
        for x in range(gray.shape[1]):
            for y in range(gray.shape[0]):
                result[y, x] = (gaussian_kernel * image_padded[y:y+3, x:x+3]).sum()
        return result

    def gaussian_smooth(self):
        x, y = np.mgrid[-1:2, -1:2]
        gaussian_kernel = np.exp(-(x**2+y**2))
        sum = gaussian_kernel.sum()
        gaussian_kernel = gaussian_kernel / sum
        return gaussian_kernel

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "1. Image Processing"))
        self.pushButton_11.setText(_translate("MainWindow", "1.1 Load Image"))
        self.pushButton_12.setText(_translate("MainWindow", "1.2 Color  Separation"))
        self.pushButton_13.setText(_translate("MainWindow", "1.3 Image Flipping"))
        self.pushButton_16.setText(_translate("MainWindow", "1.4 Blending"))
        self.groupBox_2.setTitle(_translate("MainWindow", "2. Iamge Smoothing"))
        self.pushButton_21.setText(_translate("MainWindow", "2.1 Median filter"))
        self.pushButton_22.setText(_translate("MainWindow", "2.2 Gaussian blur"))
        self.pushButton_23.setText(_translate("MainWindow", "2.3 Bilateral filter"))
        self.groupBox_3.setTitle(_translate("MainWindow", "3. Edge Detection"))
        self.pushButton_31.setText(_translate("MainWindow", "3.1 Gaussian Blur"))
        self.pushButton_32.setText(_translate("MainWindow", "3.2 Sobel X"))
        self.pushButton_33.setText(_translate("MainWindow", "3.3 Sobel Y"))
        self.pushButton_34.setText(_translate("MainWindow", "3.4 Magnitude"))
        self.groupBox_4.setTitle(_translate("MainWindow", "4. Transformation"))
        self.label.setText(_translate("MainWindow", "Rotation :"))
        self.label_2.setText(_translate("MainWindow", "deg"))
        self.label_3.setText(_translate("MainWindow", "Scaling :"))
        self.label_4.setText(_translate("MainWindow", "Tx :"))
        self.label_5.setText(_translate("MainWindow", "Ty :"))
        self.label_6.setText(_translate("MainWindow", "pixel"))
        self.label_7.setText(_translate("MainWindow", "pixel"))
        self.pushButton_41.setText(_translate("MainWindow", "4. Transformation"))