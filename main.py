import os
import xml.etree.ElementTree as ET
global wafer_id
# get input a certain wafer
wafer_list = ['D07','D08','D23','D24']
requested_list = []
final_list = []
while True:
    wafer_id = input('wafer_id : ')
    if wafer_id in wafer_list:
        requested_list.append(wafer_id)
        break
    else:
        if wafer_id == '':
            for wafer_id in wafer_list:
                requested_list.append(wafer_id)
            break
        else:
            print("This is not a file")
print(requested_list)


# define the relative path of the wafer
try:
    for i in requested_list:
        path = os.getcwd()
        os.chdir("../")
        subpath = (path + "\\data\HY202103\\" + i)
        file_list= os.listdir(subpath)
        for j in file_list:
            real_path = subpath + "\\" + j
            file_list2 = os.listdir(real_path)
            for k in file_list2:
                final_list.append(k)
except NotADirectoryError:
    pass

print(final_list)



# get the data structure
def getIVmeasurement(wafer_id):
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
        print(voltage)

    current_string = values[1].split(',')
    current = []
    for i in current_string:
        current.append(float(i))
        print(current)

for i in final_list:
    getIVmeasurement(i)