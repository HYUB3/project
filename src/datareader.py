import os

wafer_list = ['D07','D08','D23','D24']
requested_list = []

def data_reader():
    path = str(os.getcwd()).replace("src", "")
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
    for wafer in requested_list:
        for file in os.listdir(path + 'data/HY202103/' + wafer + '/'):
            if file.endswith(".xml"):
                print(os.path.join(path + 'data/HY202103/' + wafer + '/',
                                   file))
        for root, dirs, files in os.walk(path + 'data/HY202103/' + wafer + '/'):
            for name in files:
                print(name)


data_reader()
