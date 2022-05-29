import os

wafer_list = ['D07', 'D08', 'D23', 'D24']
requested_list = []
values = []
global name_list
name_list = []
global wafer_coordinate
wafer_coordinate = 0


def data_reader(name_list):
    path = str(os.getcwd()).replace("src", "")
    while True:
        wafer_id = input('wafer_id : ')
        wafer_coordinate = input('wafer_coordinate : ')
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

    for wafer in requested_list:
        for folder in os.listdir(path + '/data/HY202103/' + wafer + '/'):
            for root, dirs, files in os.walk(path + '/data/HY202103/' + wafer + '/'):
                for name in files:
                    if name.endswith(".xml"):
                        if f'({wafer_coordinate})' in name:
                            if 'LMZ' in name:
                                name_list.append(name)
    return
