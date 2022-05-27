import os

def create_folder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error:Creating directory.' + directory)

path = str(os.getcwd()).replace("src", "")
folderpath = (f'{path}/result')
create_folder(folderpath)
