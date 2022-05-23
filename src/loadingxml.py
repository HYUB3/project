import xml.etree.ElementTree as ET
import os
from loading import *
import pandas as pd

values = []
dataframe_data = []
path1 = str(os.getcwd()).replace("src", "")
global name_list
name_list = []
global iv_data
iv_data = []
global columns
columns = ['Lot', 'Wafer', 'Mask', 'TestSite', 'Name', 'Row', 'Column', 'Voltage', 'Current']

data_reader(name_list)


def findfile(name, path):
    for dirpath, dirname, filename in os.walk(path):
        if name in filename:
            if wafer_cordinate in filename:
                print(dirpath +'\\'+ name)
                return os.path.join(dirpath, name)
            else:




def xml_loader():
    path1 = str(os.getcwd()).replace("src", "")
    for names in name_list:
        wavelength = []
        filepath = findfile(names, path1)
        tree = ET.parse(filepath)
        root = tree.getroot()
        Info_Data = root.find('TestSiteInfo').attrib
        values.append(Info_Data['Batch'])
        values.append(Info_Data['Wafer'])
        values.append(Info_Data['Maskset'])
        values.append(Info_Data['TestSite'])
        names = root.find('.//Modulator')
        values.append(names.attrib['Name'])
        values.append(Info_Data['DieRow'])
        values.append(Info_Data['DieColumn'])
        voltage = list(map(float, root.find('.//IVMeasurement/Voltage').text.split(',')))
        values.append(voltage)
        current = list(map(float, root.find('.//IVMeasurement/Current').text.split(',')))
        values.append(current)
        for child in root.findall('./ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep'):
            wavelength.append(child.attrib)
            for i in child:
                wavelength.append(list(map(float, i.text.split(','))))
        values.append(wavelength)

        dataframe_data.append(values)


xml_loader()

