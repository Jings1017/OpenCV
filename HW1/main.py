import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton

import Image_Processing

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Image_Processing.Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

