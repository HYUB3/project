import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from src import gui


if __name__ == "__main__":# 현재 모듈의 이름이 저장되는 내장 변수
    # 만약 'moduleA.py' 코드를 import해서 예제 코드를 수행하면 __name__은
    # 'moduleA'가 된다. -> 이 한줄의 코드를 통해 프로그램이 직접 실행되는지 모듈을 통해
    # 실행되는지 확인인   app = QApplication(sys.argv)
    app = QApplication(sys.argv)
    window = gui.MainWindow()
    window.show()
    app.exec_()
    # 프로그램이 종료되길 원하는게 아니라 어떤 명령을 할 때
    # 맞는 행동을 하길 원하므로 프로그램이 꺼지면 안됨.-> 따라서 무한루프 상태로 만들어줌
    # sys.exit(0) -> 프로그램이 정상 종료를 함