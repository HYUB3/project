
# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

import xml.etree.ElementTree as ET
global wafer_id
# get input a certain wafer
wafer_list = ['D07','D08','D23','D24']
while True:
    wafer_id = input('wafer_id : ')
    if wafer_id in wafer_list:
        print(wafer_id)
        break
    else:
        if wafer_id == '':
            for wafer_id in wafer_list:
                print(wafer_id)
            break
        else:
            print("This is not a file")

# get the data structure
tree = ET.parse(wafer_id)
root = tree.getroot()
values = []
for child in root.find('./ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement'):
    values.append(child.text)

# load the iv measurement
voltage_string = values[0].split(',')
voltage = []
for i in voltage_string:
    voltage.append(float(i))

current_string = values[1].split(',')
current = []
for i in current_string:
    current.append(float(i))

# get the data structure
import xml.etree.ElementTree as ET
tree = ET.parse(wafer_id)
root = tree.getroot()
values = []
for child in root.find('./ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement'):
    values.append(child.text)

# load the iv measurement
voltage_string = values[0].split(',')
voltage = []
for i in voltage_string:
    voltage.append(float(i))

current_string = values[1].split(',')
current = []
for i in current_string:
    current.append(float(i))