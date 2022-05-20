import xml.etree.ElementTree as ET
import os


values = []
path1 = str(os.getcwd()).replace("src", "")


def findfile(name, path):
    for dirpath, dirname, filename in os.walk(path):
        if name in filename:
            return os.path.join(dirpath, name)


def xml_loader():
    path1 = str(os.getcwd()).replace("src", "")
    for names in range(0, len(name_list)):
        filepath = findfile(names, path1)
        tree = ET.parse(filepath)
        root = tree.getroot()
        for child in root.find('./ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement'):
            values.append(child.text)
            print(child)

xml_loader()