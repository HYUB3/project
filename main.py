import pandas
import xml.etree.ElementTree as ET
from dateutil.parser import parse
import os
from . import graph
from . import path
from . import directory


def data_save(route, time):
    tree = ET.parse(route)
    root = tree.getroot()

    columns = ['Lot', 'Wafer', 'Mask', 'TestSite', 'Name', 'Date', 'Row', 'Column', 'ErrorFlag', 'Error description', 'Analysis Wavelength', 'Rsq of Ref. spectrum (Nth)', 'Max transmission of Ref. spec. (dB)', 'Rsq of IV', 'I at -1V [A]', 'I at 1V[A]']
    #  'Script ID', 'Script Version', 'Script Owner',
    #  스크립트 정보

    data = []
    values = []

    Info_Data = root.find('TestSiteInfo').attrib
    values.append(Info_Data['Batch'])
    values.append(Info_Data['Wafer'])
    values.append(Info_Data['Maskset'])
    values.append(Info_Data['TestSite'])
    Modulator = root.find('.//Modulator')
    values.append(Modulator.attrib['Name'])
    PortCombo = Modulator.find('PortCombo')
    values.append(parse(PortCombo.attrib['DateStamp']))
    values.append(Info_Data['DieRow'])
    values.append(Info_Data['DieColumn'])

    voltage = list(map(float, root.find('.//IVMeasurement/Voltage').text.split(',')))
    current = list(map(float, root.find('.//IVMeasurement/Current').text.split(',')))


    if graph.ref_max_Rsq >= 0.995 and current[12] < (-(10 ** -4)):
        ErrorFlag = 0
        ErrorDescription = 'No Error'
    elif graph.ref_max_Rsq < 0.995 and current[12] < (-(10 ** -4)):
        ErrorFlag = 1
        ErrorDescription = 'Ref. spec. Error'
    elif graph.ref_max_Rsq >= 0.995 and current[12] >= (-(10 ** -4)):
        ErrorFlag = 2
        ErrorDescription = 'IV-fitting'
    else:
        ErrorFlag = 3
        ErrorDescription = 'Ref. spec. Error. IV-fitting'


    values.append(ErrorFlag)
    values.append(ErrorDescription)
    values.append(root.find(".//DesignParameter[@Name='Design wavelength']").text)
    values.append(graph.ref_max_Rsq)
    values.append(graph.max_IL)

    voltage = list(map(float, root.find('.//IVMeasurement/Voltage').text.split(',')))
    current = list(map(float, root.find('.//IVMeasurement/Current').text.split(',')))
    index1 = voltage.index(-1)
    index2 = voltage.index(1)
    values.append(graph.IV_max_rsq)
    values.append(current[index1])
    values.append(current[index2])
    import xml.etree.ElementTree as ET
    from xml.etree.ElementTree import parse
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy import exp
    import warnings
    import lmfit
    from lmfit import Parameters, fit_report, minimize, Model
    from . import path
    from . import directory

    def polyfit(x, y, degree):
        results = {}
        coeffs = np.polyfit(x, y, degree)
        results['polynomial'] = coeffs.tolist()
        p = np.poly1d(coeffs)
        yhat = p(x)
        ybar = np.sum(y) / len(y)
        ssreg = np.sum((yhat - ybar) ** 2)
        sstot = np.sum((y - ybar) ** 2)
        results['determination'] = ssreg / sstot
        return results

    def IV(x, Is):
        return Is * (exp(x / 0.026) - 1)

    def graph(route, save, show, time):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', np.RankWarning)
            tree = parse(str(route))
            root = tree.getroot()
            plt.figure(figsize=(16, 10))

            # IV Measurement
            plt.subplot(2, 2, 4)
            voltage = list(map(float, root.find('.//IVMeasurement/Voltage').text.split(',')))
            current = list(map(float, root.find('.//IVMeasurement/Current').text.split(',')))
            v = np.array(voltage)
            c = np.array(current)
            gmodel = Model(IV)
            params = gmodel.make_params(Is=1)
            result = gmodel.fit(c, params, x=v)

            c2 = c - result.best_fit
            IV_Rsq = {}
            for i in range(1, 10):
                poly = polyfit(v, c2, i)
                IV_Rsq[i] = poly['determination']
            IV_max_key = max(IV_Rsq, key=lambda key: IV_Rsq[key])
            global IV_max_rsq
            IV_max_rsq = IV_Rsq[IV_max_key]
            polyIV = np.poly1d(np.polyfit(v, c2, IV_max_key))(v)

            plt.plot(v, abs(c), 'ok')
            plt.plot(v, abs(result.best_fit + polyIV), '--', label='Highest order term = {}'.format(IV_max_key))
            plt.legend(loc='lower right')
            plt.xlabel('Voltage[V]')
            plt.ylabel('Current[A]')
            plt.title('IV-analysis')
            plt.yscale('log')

            # Raw Spectrum
            plt.subplot(2, 2, 1)
            TestSiteInfo = root.find('TestSiteInfo')
            TestSite = TestSiteInfo.attrib['TestSite']
            ModulatorName = ".//*[@Name='{}_ALIGN']//".format(TestSite)
            ref_L = list(map(float, root.findtext(str(str(ModulatorName + 'L'))).split(',')))
            ref_IL = list(map(float, root.findtext(str(str(ModulatorName + 'IL'))).split(',')))
            global max_IL
            max_IL = max(ref_IL)
            for wavelengthsweep in root.iter('WavelengthSweep'):
                L = list(map(float, wavelengthsweep.findtext('L').split(',')))
                IL = list(map(float, wavelengthsweep.findtext('IL').split(',')))
                if IL == ref_IL:
                    name = 'Reference'
                else:
                    name = 'DCBias=' + str(wavelengthsweep.attrib['DCBias']) + 'V'
                plt.plot(L, IL, 'o', label=name)

            plt.legend(loc='lower right')
            plt.xlabel('Wavelength[nm]')
            plt.ylabel('Measured transmissions[dB]')
            plt.title('Transmission spectra - as measured')

            # Fitting
            plt.subplot(2, 2, 3)
            x = np.array(ref_L)
            y = np.array(ref_IL)

            plt.scatter(x, y, facecolor='none', edgecolor='r', alpha=0.06, label='Reference')
            ref_Rsq = {}
            for i in range(4, 7):
                poly = polyfit(x, y, i)
                ref_Rsq[i] = poly['determination']
                fit = np.poly1d(poly['polynomial'])(x)
                plt.plot(x, fit, label='Highest order term =' + str(i) + '\nR^2 = ' + str(poly['determination']))

            ref_max_key = max(ref_Rsq, key=lambda key: ref_Rsq[key])
            global ref_max_Rsq
            ref_max_Rsq = ref_Rsq[ref_max_key]
            plt.legend(loc='lower right')
            plt.xlabel('Wavelength[nm]')
            plt.ylabel('Measured transmissions[dB]')
            plt.title('Fitting Function')

            # Modeling
            plt.subplot(2, 2, 2)
            ref = np.poly1d(np.polyfit(x, y, ref_max_key))
            for wavelengthsweep in root.iter('WavelengthSweep'):
                L = list(map(float, wavelengthsweep.findtext('L').split(',')))
                IL = list(map(float, wavelengthsweep.findtext('IL').split(',')))
                l = np.array(L)
                il = np.array(IL)
                if IL == ref_IL:
                    name = 'Reference'
                else:
                    name = 'DCBias=' + str(wavelengthsweep.attrib['DCBias']) + 'V'
                plt.plot(l, il - ref(l), label=name)

            image_path = route.replace("\\", "/").split("/")
            sub_path = ''
            for i in range(-4, -1):
                sub_path += '/' + image_path[i]
            plt.legend(loc='lower right')
            plt.xlabel('Wavelength[nm]')
            plt.ylabel('Transmissions[dB]')
            plt.title('Transmission spectra - fitted')

            if show is True:
                plt.show(block=False)
                plt.pause(3)
            if save is True:
                save_path = path.path() + '/result/graph_{}/lot'.format(time) + sub_path
                directory.create_folder(save_path)
                plt.savefig(save_path + '/' + image_path[-1][:-4] + '.png')
            import sys
            from PyQt5.QtWidgets import *
            from PyQt5.QtCore import *
            from PyQt5.QtGui import QIcon
            from . import process

            class MainWindow(QMainWindow):  # class 자식(부모)
                def __init__(self):  # 초기화 함수
                    super().__init__()
                    # 부모 클래스의 초기화 메소드 호출
                    # 자식 클래스에서 def __init__()을 사용하려면 위의 구문을 써야함.
                    self.setupUI()

                def setupUI(self):
                    self.setWindowTitle("Project B3")  # name on title bar :이름 추천받아욤
                    self.setGeometry(100, 100, 300, 400)
                    # ↑ 창의 위치를 모니터 좌상단으로부터 (가로,세로, 창의크기 가로,창의크기 세로)
                    self.setWindowIcon(QIcon('B3.png'))
                    # ↑ 창의 상단바에 아이콘 추가

                    self.square_label = QLabel(self)
                    self.square_label.setStyleSheet("color: black;"
                                                    "background-color: #D5D5D5;"
                                                    "border-radius: 7px;")
                    self.square_label.move(40, 20)
                    self.square_label.resize(230, 80)

                    self.label = QLabel("Wafer  ", self)  # waferlabel
                    self.label.move(50, 35)
                    self.label.resize(150, 20)
                    font = self.label.font()
                    font.setPointSize(10)
                    font.setFamily('Consolas')
                    font.setBold(True)
                    self.label.setFont(font)

                    self.label2 = QLabel("Coordinate  ", self)
                    self.label2.move(50, 70)
                    self.label2.resize(150, 20)
                    font2 = self.label2.font()
                    font2.setPointSize(10)
                    font2.setFamily('Consolas')
                    font2.setBold(True)
                    self.label2.setFont(font2)

                    self.waferEdit = QLineEdit("All", self)
                    self.waferEdit.move(170, 35)
                    self.waferEdit.setStyleSheet("color: white;"
                                                 "background-color: #002266;"
                                                 "border-style: solid;"
                                                 "border-width: 2px;"
                                                 "border-radius: 3px;"
                                                 "border-color: #white")
                    self.waferEdit.move(170, 35)
                    self.waferEdit.resize(80, 20)

                    self.columnEdit = QLineEdit("All", self)
                    self.columnEdit.setStyleSheet("color: white;"
                                                  "background-color: #002266;"
                                                  "border-style: solid;"
                                                  "border-width: 2px;"
                                                  "border-radius: 3px;"
                                                  "border-color: #white")
                    self.columnEdit.move(170, 70)
                    self.columnEdit.resize(80, 20)

                    self.showEdit = QCheckBox("Show", self)
                    self.showEdit.move(100, 110)
                    self.showEdit.toggle()

                    self.saveEdit = QCheckBox("Save Figure", self)
                    self.saveEdit.move(100, 140)
                    self.saveEdit.toggle()

                    self.csvEdit = QCheckBox("Save CSV", self)
                    self.csvEdit.move(100, 170)
                    self.csvEdit.toggle()

                    self.label3 = QLabel('', self)
                    self.label3.move(50, 170)

                    self.btnOpenFolder = QPushButton("Set Data Folder", self)  # openfolder button
                    self.btnOpenFolder.setStyleSheet("color: white;"
                                                     "background-color: #002266;"
                                                     "border-style: solid;"
                                                     "border-width: 2px;"
                                                     "border-radius: 3px;"
                                                     "border-color: #white")
                    self.btnOpenFolder.resize(150, 45)  # button size 150*30
                    self.btnOpenFolder.move(75, 220)  # button location
                    self.btnOpenFolder.clicked.connect(self.find_folder)  # 'clicked' signal이 find_folder 메소드에 연결

                    self.btnOpenSave = QPushButton("Open Result Folder", self)
                    self.btnOpenSave.setStyleSheet("color: white;"
                                                   "background-color: #002266;"
                                                   "border-style: solid;"
                                                   "border-width: 2px;"
                                                   "border-radius: 3px;"
                                                   "border-color: #white")
                    self.btnOpenSave.resize(150, 45)
                    self.btnOpenSave.move(75, 270)
                    self.btnOpenSave.clicked.connect(self.open_folder)

                    self.btnSave = QPushButton("OK", self)
                    self.btnSave.setStyleSheet("color: white;"
                                               "background-color: #002266;"
                                               "border-style: solid;"
                                               "border-width: 2px;"
                                               "border-radius: 4px;"
                                               "border-color: #white")
                    self.btnSave.resize(150, 45)
                    self.btnSave.move(75, 320)
                    self.btnSave.clicked.connect(self.btnInput_clicked)

                    self.center()  # 창을 모니터 화면의 가운데에 배치

                def center(self):
                    qr = self.frameGeometry()  # frameGeometry() 메서드 사용해서 창의 위치와 크기 정보 가져옴
                    cp = QDesktopWidget().availableGeometry().center()  # 사용하는 모니터 화면의 가운데 위치를 파악
                    qr.moveCenter(cp)  # 창의 직사각형 위치를 화면의 중심 위치로 이동
                    self.move(qr.topLeft())  # 현재 창을 화면의 중심으로 이동했던 직사각형(qr)의 위치로 이동

                def find_folder(self):  # define find_folder function
                    FileFolder = QFileDialog.getExistingDirectory(self, 'Find Folder')
                    self.label3.setText(FileFolder)

                def open_folder(self):
                    process.open()  # 수정해야될 부분

                def btnInput_clicked(self):
                    wafer = self.waferEdit.text()
                    column = self.columnEdit.text()
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
                        import os

                        def path():
                            path = '/'.join(os.getcwd().split(
                                "\\"))  # C:/Users/Huichan/Desktop/YHC/2022 - 1학기/pythonProject/project - B3/src
                            return path

                    except:
                        QMessageBox.information(self, 'Error', 'Error Unknown')

            plt.close()

    # 스크립트 정보 추가 필요

    data.append(values)
    df = pandas.DataFrame(data, columns=columns).set_index("Lot")
    save_route = path.path() + '/result/csv_{}'.format(time)
    directory.create_folder(save_route)
    if not os.path.exists(save_route + '/analyzed.csv'):
        df.to_csv(save_route + '/analyzed.csv', mode='w')
    else:
        df.to_csv(save_route + '/analyzed.csv', mode='a', header=False)
import os


def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)
import pandas as pd
import matplotlib.pyplot as plt
from . import path
from . import directory
import warnings


# wafer-to-wafer
def analyze(time):
    warnings.filterwarnings(action='ignore')
    data = pd.read_csv('./result/csv_{}/analyzed.csv'.format(time))
    y = []
    wafernumber = data['Wafer']

    max_spec = data['Max transmission of Ref. spec. (dB)']
    pos_volt = data['I at -1V [A]']
    neg_volt = data['I at 1V[A]']
    wavelength = data['Analysis Wavelength']

    plt.figure(figsize=(16, 10))
    plt.suptitle('Result of wafer-to-wafer using csv file', fontsize=20)
    plt.subplots_adjust(wspace=0.5, hspace=0.5)

    for i in range(len(wafernumber)):
        if str(wavelength[i]) == '1550':
            plt.subplot(2, 3, 1)
            if str(wafernumber[i]) == 'D07':
                plt.scatter(wafernumber[i], max_spec[i], c='red')
            elif str(wafernumber[i]) == 'D08':
                plt.scatter(wafernumber[i], max_spec[i], c='blue')
            elif str(wafernumber[i]) == 'D23':
                plt.scatter(wafernumber[i], max_spec[i], c='green')
            elif str(wafernumber[i]) == 'D24':
                plt.scatter(wafernumber[i], max_spec[i], c='black')
            plt.title('Max transmission Ref.spec. in 1550nm')
            plt.ylabel('dB')
        else:
            plt.subplot(2, 3, 4)
            if str(wafernumber[i]) == 'D07':
                plt.scatter(wafernumber[i], max_spec[i], c='red')
            elif str(wafernumber[i]) == 'D08':
                plt.scatter(wafernumber[i], max_spec[i], c='blue')
            elif str(wafernumber[i]) == 'D23':
                plt.scatter(wafernumber[i], max_spec[i], c='green')
            elif str(wafernumber[i]) == 'D24':
                plt.scatter(wafernumber[i], max_spec[i], c='black')
            plt.title('Max transmission Ref.spec. in 1310nm')
            plt.ylabel('dB')

    for i in range(len(wafernumber)):
        if str(wavelength[i]) == '1550':
            plt.subplot(2, 3, 2)
            if str(wafernumber[i]) == 'D07':
                plt.scatter(wafernumber[i], pos_volt[i], c='red')
            elif str(wafernumber[i]) == 'D08':
                plt.scatter(wafernumber[i], pos_volt[i], c='blue')
            elif str(wafernumber[i]) == 'D23':
                plt.scatter(wafernumber[i], pos_volt[i], c='green')
            elif str(wafernumber[i]) == 'D24':
                plt.scatter(wafernumber[i], pos_volt[i], c='black')
            plt.title('I at 1V in 1550nm')
            plt.ylabel('Current[A]')
        else:
            plt.subplot(2, 3, 5)
            if str(wafernumber[i]) == 'D07':
                plt.scatter(wafernumber[i], pos_volt[i], c='red')
            elif str(wafernumber[i]) == 'D08':
                plt.scatter(wafernumber[i], pos_volt[i], c='blue')
            elif str(wafernumber[i]) == 'D23':
                plt.scatter(wafernumber[i], pos_volt[i], c='green')
            elif str(wafernumber[i]) == 'D24':
                plt.scatter(wafernumber[i], pos_volt[i], c='black')
            plt.title('I at 1V in 1310nm')
            plt.ylabel('Current[A]')

    for i in range(len(wafernumber)):
        if str(wavelength[i]) == '1550':
            plt.subplot(2, 3, 3)
            if str(wafernumber[i]) == 'D07':
                plt.scatter(wafernumber[i], neg_volt[i], c='red')
            elif str(wafernumber[i]) == 'D08':
                plt.scatter(wafernumber[i], neg_volt[i], c='blue')
            elif str(wafernumber[i]) == 'D23':
                plt.scatter(wafernumber[i], neg_volt[i], c='green')
            elif str(wafernumber[i]) == 'D24':
                plt.scatter(wafernumber[i], neg_volt[i], c='black')
            plt.title('I at -1V in 1550nm')
            plt.ylabel('Current[A]')
        else:
            plt.subplot(2, 3, 6)
            if str(wafernumber[i]) == 'D07':
                plt.scatter(wafernumber[i], neg_volt[i], c='red')
            elif str(wafernumber[i]) == 'D08':
                plt.scatter(wafernumber[i], neg_volt[i], c='blue')
            elif str(wafernumber[i]) == 'D23':
                plt.scatter(wafernumber[i], neg_volt[i], c='green')
            elif str(wafernumber[i]) == 'D24':
                plt.scatter(wafernumber[i], neg_volt[i], c='black')
            plt.title('I at -1V in 1310nm')
            plt.ylabel('Current[A]')

    save_path = path.path() + '/result/csv_{}'.format(time)
    directory.create_folder(save_path)

    plt.savefig(save_path + '/wafer_to_wafer.png')
from . import wtw


now = datetime.datetime.now()
nowDatetime = now.strftime('%Y%m%d_%H%M%S')


def work(wafer, coordinate, save, show, csv, data_path):
    file = []
    if data_path == '':
        if wafer == 'All' and coordinate == 'All':
            file = glob(path.path() + '/data/**/*LMZ*.xml', recursive=True)
        elif coordinate == 'All':
            file = glob(path.path() + f'/data/**/{wafer}/**/*LMZ*.xml', recursive=True)
        else:
            file = glob(path.path() + f'/data/**/{wafer}/**/*{coordinate}*LMZ*.xml', recursive=True)
    else:
        if wafer == 'All' and coordinate == 'All':
            file = glob(data_path + '/**/*LMZ*.xml', recursive=True)
        elif coordinate == 'All':
            file = glob(data_path + f'/**/{wafer}/**/*LMZ*.xml', recursive=True)
        else:
            file = glob(data_path + f'/**/{wafer}/**/*{coordinate}*LMZ*.xml', recursive=True)

    if not file:
        raise ValueError('Not found data')

    for i in tqdm(file):
        graph.graph(i, save, show, nowDatetime)
        if csv is True:
            extract.data_save(i, nowDatetime)
    if csv is True:
        wtw.analyze(nowDatetime)


def open():
    os.startfile(path.path() + '/result')

