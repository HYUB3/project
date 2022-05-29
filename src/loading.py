import os
import glob

wafer_list = ['D07', 'D08', 'D23', 'D24']
requested_list = []
values = []
global name_list
name_list = []
global wafer_coordinate
wafer_coordinate = 0
filterlist = []

def data_reader(name_list):
    path = str(os.getcwd()).replace("src", "")
    while True:
        wafer_id = input('wafer_id : ')
        wafer_coordinate = input('wafer_coordinate : ')
        if wafer_id in wafer_list:
            listin_folder = os.listdir(f'{path}' + '//data//HY202103//' + f'{wafer_id}')
            requested_list.append(wafer_id)
            if wafer_coordinate == '':
                for numb in listin_folder:
                    try:
                        list_folder = os.listdir(f'{path}' + '//data//HY202103//' + f'{wafer_id}/{numb}')
                        for filt in list_folder:
                            if 'LMZ' in filt:
                                filterlist.append(filt)
                    except NotADirectoryError:
                        break
            print(len(filterlist))
            break
        else:
            if wafer_id == '':
                for wafer_id in wafer_list:
                    requested_list.append(wafer_id)
                    listin_folder = os.listdir(f'{path}' + '//data//HY202103//' + f'{wafer_id}')
                    if wafer_coordinate == '':
                        for numb in listin_folder:
                            try:
                                list_folder = os.listdir(f'{path}' + '//data//HY202103//' + f'{wafer_id}/{numb}')
                                for filt in list_folder:
                                    if 'LMZ' in filt:
                                        filterlist.append(filt)
                            except NotADirectoryError:
                                break
                print(len(filterlist))
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
    print(sorted(list(set(name_list))))
    return