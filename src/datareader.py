import os

wafer_list = ['D07','D08','D23','D24']
requested_list = []
global filenames
filenames = []

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
                if name.endswith(".xml"):
                    print(name)
                    filenames.append(name)


substring_LMZO = 'LMZO.xml'
substring_LMZC = 'LMZC.xml'

LMZO = []
LMZC = []

def filter(filenames):
    for i in range(0, len(filenames)):
        control = filenames[i].split("_")
        if substring_LMZC in control:
            print(filenames[i])
            LMZC.append(filenames[i])
        if substring_LMZO in control:
            print(filenames[i])
            LMZO.append(filenames[i])
    print('LMZC', LMZC)
    print('LMZO', LMZO)



data_reader()
filter(filenames)
