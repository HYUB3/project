import pandas as pd
from src.data_plotting import *

plotting()

def export_csv():
    path = str(os.getcwd()).replace("src", "")
    path1 = path + '/result/'
    i = 0
    for dict_data in fitted_data:
        data_frame = pd.DataFrame.from_dict(dict_data, orient='index')
        string = str(name_list[i])
        string2 = string.replace('.xml', '.csv')
        data_frame.to_csv(os.path.join(path1, string2))
        i = i + 1




