import xml.etree.ElementTree as ET

xml_data_iv = []


def loading_xml(filenames):
    for i in range(0, len(filenames)):
        tree = ET.parse(i)
        root = tree.getroot()
        values = []
        for child in root.find('./ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement'):
            print(child.text)
            xml_data_iv.append(child.text)
