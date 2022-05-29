import pandas as pd
from src.data_plotting import *

plotting()


def export_csv():
    path = str(os.getcwd()).replace("src", "")
    path1 = path + '/result/'
    i = 0
    for dict_data in fitted_data:
        new_frame = [dict_data['Lot'], dict_data['Wafer'], dict_data['Mask'], dict_data['TestSite'], dict_data['Name'],
                     dict_data['Date'], dict_data['Row'], dict_data['Column'], dict_data['Min'], dict_data['Max'],
                     dict_data['R Square']]

        print(new_frame)
        data_frame = pd.DataFrame({'Lot': dict_data['Lot'],
                                   'Wafer': dict_data['Wafer'],
                                   'Mask': dict_data['Mask'],
                                   'TestSite': dict_data['TestSite'],
                                   'Name:': dict_data['Name'],
                                   'Date': dict_data['Date'],
                                   'Row': dict_data['Row'],
                                   'Column': dict_data['Column'],
                                   'Min': dict_data['Min'],
                                   'Max': dict_data['Max'],
                                   'R Square': dict_data['R Square']})
        print(data_frame)
        string = str(name_list[i])
        string2 = string.replace('.xml', '.csv')
        data_frame.to_csv(os.path.join(path1, string2))
        i = i + 1
