import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import QIcon
# from . import process


class MainWindow(QWidget): #class 자식(부모)
    def __init__(self):                 #초기화 함수
        super().__init__()
        # 부모 클래스의 초기화 메소드 호출
        # 자식 클래스에서 def __init__()을 사용하려면 위의 구문을 써야함.
        self.setupUI()

    def setupUI(self):
        self.setWindowTitle("Project B3") #name on title bar :이름 추천받아욤
        self.setGeometry(100, 100, 300, 400)
        # ↑ 창의 위치를 모니터 좌상단으로부터 (가로,세로, 창의크기 가로,창의크기 세로)
        self.setWindowIcon(QIcon('../B3.png'))
        # ↑ 창의 상단바에 아이콘 추가


        self.square_label = QLabel(self)
        self.square_label.setStyleSheet("color: black;"
                                 "background-color: #D5D5D5;"
                                 "border-radius: 7px;")
        self.square_label.move(40,20)
        self.square_label.resize(230,80)


        self.wafer_label = QLabel("Wafer  ", self)  #waferlabel
        self.wafer_label.move(50, 35)
        self.wafer_label.resize(150, 20)
        font = self.wafer_label.font()              #changing font style
        font.setPointSize(10)
        font.setFamily('Consolas')
        font.setBold(True)
        self.wafer_label.setFont(font)

        #coordinate label
        self.coordinate_label = QLabel("Coordinate  ", self)
        self.coordinate_label.move(50, 70)
        self.coordinate_label.resize(150, 20)
        font2 = self.coordinate_label.font()
        font2.setPointSize(10)
        font2.setFamily('Consolas')
        font2.setBold(True)
        self.coordinate_label.setFont(font2)

        #Input wafer
        self.Input_Wafer = QLineEdit("All", self)
        self.Input_Wafer.move(170, 35)
        self.Input_Wafer.setStyleSheet("color: white;"
                              "background-color: #002266;"
                              "border-style: solid;"
                              "border-width: 2px;"
                              "border-radius: 5px;"
                              "border-color: #002266")
        self.Input_Wafer.move(170, 35)
        self.Input_Wafer.resize(80, 20)

        #input coordinate
        self.Input_coordinate = QLineEdit("All", self)
        self.Input_coordinate.setStyleSheet("color: white;"
                              "background-color: #002266;"
                              "border-style: solid;"
                              "border-width: 2px;"
                              "border-radius: 5px;"
                              "border-color: #002266")
        self.Input_coordinate.move(170, 70)
        self.Input_coordinate.resize(80, 20)


    #Checkbox
        #show checkbox
        self.showEdit = QCheckBox("Show", self)
        self.showEdit.move(100, 120)
        self.showEdit.toggle()

        # SaveFigure checkbox
        self.saveEdit = QCheckBox("Save Figure", self)
        self.saveEdit.move(100, 150)
        self.saveEdit.toggle()

        # Save CSV checkbox
        self.csvEdit = QCheckBox("Save CSV", self)
        self.csvEdit.move(100, 180)
        self.csvEdit.toggle()

        self.label3 = QLabel('', self)
        self.label3.move(50, 170)



    #button
        #Set Data Folder Button
        self.btnOpenFolder = QPushButton("Set Data Folder", self) #openfolder button
        self.btnOpenFolder.setStyleSheet("color: white;"
                              "background-color: #002266;"
                              "border-style: solid;"
                              "border-width: 2px;"
                              "border-radius: 5px;"
                              "border-color: #white")
        self.btnOpenFolder.resize(150, 45) #button size 150*30
        self.btnOpenFolder.move(75, 220) # button location
        self.btnOpenFolder.clicked.connect(self.find_folder) #'clicked' signal이 find_folder 메소드에 연결


        # Open Result Folder Button
        self.btnOpenSave = QPushButton("Open Result Folder", self)
        self.btnOpenSave.setStyleSheet("color: white;"
                              "background-color: #002266;"
                              "border-style: solid;"
                              "border-width: 2px;"
                              "border-radius: 5px;"
                              "border-color: #white")
        self.btnOpenSave.resize(150, 45)
        self.btnOpenSave.move(75, 270)
        self.btnOpenSave.clicked.connect(self.open_folder)


        #OK Button
        self.btnSave = QPushButton("OK", self)
        self.btnSave.setStyleSheet("color: white;"
                      "background-color: #002266;"
                      "border-style: solid;"
                      "border-width: 2px;"
                      "border-radius: 4px;"
                      "border-color: #white")
        self.btnSave.resize(150,45)
        self.btnSave.move(75, 320)
        self.btnSave.clicked.connect(self.btnInput_clicked)


        self.center() #창을 모니터 화면의 가운데에 배치

    def center(self):
        qr = self.frameGeometry()  #frameGeometry() 메서드 사용해서 창의 위치와 크기 정보 가져옴
        cp = QDesktopWidget().availableGeometry().center() #사용하는 모니터 화면의 가운데 위치를 파악
        qr.moveCenter(cp) #창의 직사각형 위치를 화면의 중심 위치로 이동
        self.move(qr.topLeft()) #현재 창을 화면의 중심으로 이동했던 직사각형(qr)의 위치로 이동

    def find_folder(self): #define find_folder function
        FileFolder = QFileDialog.getExistingDirectory(self, 'Find Folder')
        self.label3.setText(FileFolder)

    def open_folder(self):
        process.open() #수정해야될 부분

    def btnInput_clicked(self):
        wafer = self.Input_Wafer.text()
        column = self.Input_coordinate.text()
        save = self.saveEdit.isChecked()
        show = self.showEdit.isChecked()
        csv = self.csvEdit.isChecked()
        data = self.label3.text()
        try:
            if wafer == '' or column == '':
                raise ValueError('There is blank')
            else:
                process.work(wafer, column, save, show, csv, data)
                QMessageBox.information(self, 'Message', str('Done!'))
        except ValueError as e:
            QMessageBox.information(self, 'Error', str(e))
        except:
            QMessageBox.information(self, 'Error', 'Error Unknown')

if __name__ == "__main__":# 현재 모듈의 이름이 저장되는 내장 변수
    # 만약 'moduleA.py' 코드를 import해서 예제 코드를 수행하면 __name__은
    # 'moduleA'가 된다. -> 이 한줄의 코드를 통해 프로그램이 직접 실행되는지 모듈을 통해
    # 실행되는지 확인인   app = QApplication(sys.argv)
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()
    # 프로그램이 종료되길 원하는게 아니라 어떤 명령을 할 때
    # 맞는 행동을 하길 원하므로 프로그램이 꺼지면 안됨.-> 따라서 무한루프 상태로 만들어줌
    # sys.exit(0) -> 프로그램이 정상 종료를 함