import warnings
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import lmfit
import pandas as pd
from scipy.signal import find_peaks
import glob
from glob import glob
from src.loading import *
from src.loadingxml import *
from src.create import *


wafer_list = ['D07', 'D08', 'D23', 'D24']
name_list = sorted(list(set(name_list)))
print(name_list)

def diode_function(x, a, b, c, d):
    return b * (np.exp((d * x) / (a * c)) - 1)


model2 = lmfit.Model(diode_function)
params2 = lmfit.Parameters()
params2.add("a", value=1)
params2.add("b", value=1)
params2.add("c", value=1)
params2.add("d", value=1)


def function(x, a, b, c, d, e):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e


model1 = lmfit.Model(function)
params = lmfit.Parameters()
params.add("a", value=1)
params.add("b", value=1)
params.add("c", value=1)
params.add("d", value=1)
params.add("e", value=1)

fitted_values = []


def plotting():
    i = 0
    fitting_row_x = np.asarray(0)
    fitting_row_y = np.asarray(0)
    for test in dataframe_data:
        plt.subplot(231)
        plt.plot(test['Voltage'][i], abs(np.asarray(test['Current'][i])), color='black', marker='o',
                 markeredgecolor='black',
                 markerfacecolor='red')
        plt.yscale('log')
        plt.title('IV-Analysis')
        plt.ylabel('Current in A')
        plt.xlabel('Voltage in V')
        plt.grid('true')
        plt.tight_layout()

        result1 = model1.fit(abs(np.asarray(test['Current'][i][0:10])), x=np.asarray(test['Voltage'][i][0:10])
                             , params=params)
        result2 = model2.fit(abs(np.asarray(test['Current'][i][8:13])), x=np.asarray(test['Voltage'][i][8:13])
                             , params=params2)
        plt.subplot(222)
        plt.plot(test['Voltage'][i], abs(np.asarray(test['Current'][i])), 'x', color='black')
        plt.plot(np.asarray(test['Voltage'][i][0:10]), result1.best_fit, '--r')
        plt.plot(np.asarray(test['Voltage'][i][8:13]), result2.best_fit, '--g')
        plt.yscale('log')
        plt.title('IV-Analysis')
        plt.ylabel('Current in A')
        plt.xlabel('Voltage in V')
        plt.grid('true')

        plt.subplot(223)
        liste = list(range(0, len(test['Wavelength'][i]), 3))
        k = 0
        for j in liste:
            plt.plot(test['Wavelength'][i][j + 1], test['Wavelength'][i][j + 2],
                     label=test['Wavelength'][i][j]['DCBias'])
        plt.legend(fontsize='small', title='DCBias in V', ncol=2)
        plt.xlabel('Wavelength in nm')
        plt.ylabel('Measured transmission in dB')
        plt.title('Transmission spectral')

        if test['Wavelength'][i][j]['DCBias'] == '0.0' and k == 0:
            k = k + 1
            plt.subplot(224)
            print(k)
            plt.plot(test['Wavelength'][i][j + 1], test['Wavelength'][i][j + 2],
                     label=test['Wavelength'][i][j]['DCBias'])
            fou_deg = np.poly1d(
                np.polyfit(np.asarray(test['Wavelength'][i][j + 1]), np.asarray(test['Wavelength'][i][j + 2])
                           , 4))
            four_deg_value = fou_deg(np.asarray(test['Wavelength'][i][j + 1]))
            n_max = four_deg_value.argmax()
            print('Maximal value of the fitted function:', four_deg_value[n_max], 'X value:',
                  np.asarray(test['Wavelength'][i][j + 1])[n_max])
            n_min = four_deg_value.argmin()
            print('Minimal value of the fitted function:', four_deg_value[n_min], 'Y Value:',
                  np.asarray(test['Wavelength'][i][j + 1])[n_max])
            x_maximal = np.asarray(test['Wavelength'][i][j + 1])[n_max]
            maximal = four_deg_value[n_max]
            plt.plot(test['Wavelength'][i][j + 1], fou_deg(np.asarray(test['Wavelength'][i][j + 1])),
                     label='4th degree')
            plt.plot(np.asarray(test['Wavelength'][i][j + 1])[n_max], four_deg_value[n_max], 'o', color='black',
                     linewidth=2, label='Maximal Value',
                     markerfacecolor='red')
            plt.plot(np.asarray(test['Wavelength'][i][j + 1])[n_min], four_deg_value[n_min], 'o', color='black',
                     linewidth=2, label='Minimal Value',
                     markerfacecolor='green')
            plt.legend(fontsize='small', title='DCBias in V', ncol=2)
            plt.xlabel('Wavelength in nm')
            plt.ylabel('Measured transmission in dB')
            plt.title('Transmission spectral')

            #fitted_values.extend('Minimal value of the fitted function:', four_deg_value[n_min], 'Y Value:',
                  #np.asarray(test['Wavelength'][i][j + 1])[n_max], test['Lot'], test['Wafer'])
            #fitted_values.extend('Minimal value of the fitted function:', four_deg_value[n_min], 'Y Value:',
                  #np.asarray(test['Wavelength'][i][j + 1])[n_min], test['Lot'], test['Wafer'])
            #print(fitted_values)


            #substraction:
            plt.plot(test['Wavelength'][i][j + 1], test['Wavelength'][i][j + 2],
                     label=test['Wavelength'][i][j]['DCBias'])
            fou_deg(test['Wavelength'][i][j + 1])

            path = str(os.getcwd()).replace("src", "")
            folderpath = (f'{path}/result')
            print(f'{folderpath}' + '/graph_' + f'{wafer1}_{name1.replace("xml", "")}' + 'png')
            plt.savefig(f'{folderpath}' + '/graph_' + f'{wafer1}_{name1.replace("xml", "")}' + 'png', dpi=150,
                        bbox_inches='tight')
            plt.show()

for name1 in name_list:
    for wafer1 in wafer_list:
        if wafer1 in name1:
            path = str(os.getcwd()).replace("src", "")
            # pathroot = (path + 'data\HY202103'+ '\\' + f'{j}' +'\\' + '..\\'+ i)
            file = glob(path + 'data\*\\' + f'{wafer1}\\*\\{name1}', recursive = True)
            for one in file:
                # get the data structure
                tree = ET.parse(one)
                root = tree.getroot()
                values = []
                for child in root.find('./ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/IVMeasurement'):
                    values.append(child.text)
                    plotting()
                    # pandas
                    element = root.find('.//TestSiteInfo')
                    lot = element.attrib['Batch']
                    wafer = element.attrib['Wafer']
                    mask = element.attrib['Maskset']
                    testSite = element.attrib['TestSite']
                    die_row = element.attrib['DieRow']
                    die_column = element.attrib['DieColumn']
                    element1 = root.find('.//Modulator')
                    name = element1.attrib['Name']

                    operator = root.attrib['Operator']
                    date = root.attrib['CreationDate']
                    des_par = root.findall('.//DesignParameter')[1].text

                    ''' #with variable
                    data = []
                    for key, value in sorted(rsq.items()):
                        values = [lot, wafer, mask, testSite, die_row, die_column, date, name, key, round(value, 4), round(maximal[key], 2),
                                  round(x_maximal[key]), des_par]
                        data.append(values)
                    '''

                    values = [lot, wafer, mask, testSite, die_row, die_column, date, name, x_maximal, maximal,
                              rsquare_measure, des_par,
                              rsquare_Iv, current_plus1, current_minus1]
                    df = pd.DataFrame([values],
                                      columns=['Lot', 'Wafer', 'Mask', 'TestSite', 'Die Row', 'Die Column', 'Date',
                                               'Name',
                                               'X Value', 'Max Value in dB', 'Rsquare value of the ',
                                               'Design wavelength [nm]', 'Rsquare of IV', 'I at 1',
                                               'I at -1'])
                    path = str(os.getcwd()).replace("src", "")
                    folderpath = (f'{path}/result')
                    df.to_csv(f'{folderpath}' + '/text_' + f'{wafer1}_{name1.replace("xml", "")}' + 'csv', index=False)
            else:
                continue