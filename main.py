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

# define the relative path of the wafer
for i in requested_list:
    os.chdir("../")
    path = os.getcwd()
    subpath = (path + "\\HY202103\\" + wafer_id)
    file_list = os.listdir(subpath)
    for i in file_list:
        real_path = subpath + '\\' + i
        file_list2 = os.listdir(real_path)
        final_list.append(file_list2)
    print(final_list)



# get the data structure
# tree = ET.parse(wafer_id)
# root = tree.getroot()
# values = []
# for child in root.find('./ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement'):
#     values.append(child.text)
#
# # load the iv measurement
# voltage_string = values[0].split(',')
# voltage = []
# for i in voltage_string:
#     voltage.append(float(i))
#
# current_string = values[1].split(',')
# current = []
# for i in current_string:
#     current.append(float(i))
#
# # get the data structure
# import xml.etree.ElementTree as ET
# tree = ET.parse(wafer_id)
# root = tree.getroot()
# values = []
# for child in root.find('./ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement'):
#     values.append(child.text)
#
# # load the iv measurement
# voltage_string = values[0].split(',')
# voltage = []
# for i in voltage_string:
#     voltage.append(float(i))
#
# current_string = values[1].split(',')
# current = []
# for i in current_string:
#     current.append(float(i))