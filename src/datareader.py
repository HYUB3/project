import xml.etree.ElementTree as ET
import os
from pathlib import Path
import string
import glob
import pandas as pd


values = []

path = str(os.getcwd()).replace("src", "")
wafer_folder = os.listdir(path + 'dat/HY202103')

def data_reader():
    path = str(os.getcwd()).replace("src", "")
    wafer_folder = os.listdir(path + 'dat/HY202103')
    for wafer in wafer_folder:
        print(path + 'dat/HY202103/' + wafer + '/')
        for file in os.listdir(path + 'dat/HY202103/' + wafer + '/'):
            if file.endswith(".xml"):
                print(os.path.join(path + 'dat/HY202103/' + wafer + '/',
                                   file))
        for root, dirs, files in os.walk(path + 'dat/HY202103/' + wafer + '/'):
            for name in files:
                print(name)
                fpath = os.path.join(root, name)
                tree = ET.parse(fpath)
                wurzel = tree.getroot()
                try:
                    for child in wurzel.find('./ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement'):
                        values.append(child.text)
                except Exception as e:
                    print(e)

data_reader()
# tree = ET.parse('/Users/fabiankading/PycharmProjects/pythonProject3/dat/HY202103')
# root = tree.getroot()
# values = []
# for child in root.find('./ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement'):
# values.append(child.text)
