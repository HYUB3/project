import time
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
from loading import *
from loadingxml import *
from create import *

wafer_list = ['D07', 'D08', 'D23', 'D24']
name_list = sorted(list(set(name_list)))
print(name_list)


def function(x, a, b, c, d, e):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

def diode_function(x, a, b, c, d):
    return b * (np.exp((d * x) / (a * c)) - 1)

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

                # load the iv measurement
                voltage_string = values[0].split(',')
                voltage = []
                for i in voltage_string:
                    voltage.append(float(i))

                current_string = values[1].split(',')
                current = []
                for i in current_string:
                    current.append(float(i))
                # values at 1, minus 1
                current_plus1 = current[voltage.index(-1)]
                current_minus1 = current[voltage.index(1)]
                print(current_plus1, current_minus1)

                # transform to np.array
                voltage = np.array(voltage)
                current = np.array(current)

                current = [abs(i) for i in current]
                plt.figure(figsize=(18, 13))
                # plot of the iv measurement
                plt.subplot(331)
                plt.plot(voltage, current, color='black', marker='o', markeredgecolor='black', markerfacecolor='red')
                plt.yscale('log')
                plt.title('IV-Analysis')
                plt.ylabel('Current in A')
                plt.xlabel('Voltage in V')
                plt.grid('true')

                plt.tight_layout()

                # fitting the iv measurement
                # first part of funktion
                x1 = np.asarray(voltage[0:10])
                y1 = np.asarray(current[0:10])


                model1 = lmfit.Model(function)
                params = lmfit.Parameters()
                params.add("a", value=1)
                params.add("b", value=1)
                params.add("c", value=1)
                params.add("d", value=1)
                params.add("e", value=1)
                result1 = model1.fit(y1, x=x1, params=params)

                # second part of the function
                x2 = np.asarray(voltage[8:13])
                y2 = np.asarray(current[8:13])



                model2 = lmfit.Model(diode_function)
                params2 = lmfit.Parameters()
                params2.add("a", value=1)
                params2.add("b", value=1)
                params2.add("c", value=1)
                params2.add("d", value=1)
                result2 = model2.fit(y2, x=x2, params=params2)

                # calculate the r square of the function
                rsquare_y1 = r2_score(y1, result1.best_fit)
                print('Rsquare value in %:', rsquare_y1)

                rsquare_y2 = r2_score(y2, result2.best_fit)
                print('Rsquare value in %:', rsquare_y2)

                rsquare_Iv = (rsquare_y2 + rsquare_y1) / 2

                # plotting the iv measurement fitted
                plt.subplot(332)
                plt.plot(voltage, current, 'x', color='black')
                plt.plot(x1, result1.best_fit, '--r')
                plt.plot(x2, result2.best_fit, '--g')
                plt.yscale('log')
                plt.title('IV-Analysis')
                plt.ylabel('Current in A')
                plt.xlabel('Voltage in V')
                plt.grid('true')

                plt.tight_layout()

                # loading the wavelenght measurement
                wavelength = []
                for child in root.findall(
                        './ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep'):
                    wavelength.append(child.attrib)
                    for i in child:
                        wavelength.append(list(map(float, i.text.split(','))))
                liste = list(range(0, len(wavelength), 3))
                for i in liste:
                    wavelength[i + 1] = np.array(wavelength[i + 1])
                    wavelength[i + 2] = np.array(wavelength[i + 2])

                # plot of the wavelenght measurement
                plt.subplot(333)
                liste = list(range(0, len(wavelength), 3))
                for i in liste:
                    plt.plot(wavelength[i + 1], wavelength[i + 2], label=wavelength[i]['DCBias'])
                plt.legend(fontsize='small', title='DCBias in V', ncol=2)
                plt.xlabel('Wavelenght in nm')
                plt.ylabel('Measured transmission in dB')
                plt.title('Transmission spectral')

                # fiiting of the wavelenght measurement
                x = np.array(wavelength[19])
                print(x)
                y = np.array(wavelength[20])
                print(y)

                # fitting functions
                sec_deg = np.poly1d(np.polyfit(x, y, 2))
                thd_deg = np.poly1d(np.polyfit(x, y, 3))
                fou_deg = np.poly1d(np.polyfit(x, y, 4))
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', np.RankWarning)
                    thirty_deg = np.poly1d(np.polyfit(x, y, 30))

                # runtime detection
                rsq = {}
                maximal = {}
                x_maximal = {}
                for i in range(6):
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', np.RankWarning)
                        start = time.time()
                        p = np.poly1d(np.polyfit(x, y, i))
                        end = time.time()
                        r2 = r2_score(y, p(x)) * 100
                        print('Time of calculation', i, 'th degree:', end - start, 's')
                        print('Rsquare value in %:', r2, 'of the', i, 'degree polynomal fit ')
                        # rsq[str(i)] = r2_score(y, p(x))
                        # maximal[str(i)] = max(p(x))  # max value
                        # some = p(x).argmax()
                        # x_maximal[str(i)] = x[some]

                # R^2 score 30th degree
                r2_30 = r2_score(y, thirty_deg(x))
                print('30th degree:', r2_30 * 100)

                # 4th degree values
                four_deg_value = fou_deg(x)

                # extreme values for the 4th degree
                n_max = four_deg_value.argmax()
                print('Maximal value of the fitted function:', four_deg_value[n_max], 'X value:', x[n_max])
                n_min = four_deg_value.argmin()
                print('Minimal value of the fitted function:', four_deg_value[n_min])
                x_maximal = x[n_max]
                maximal = four_deg_value[n_max]
                rsquare_measure = r2_score(y, four_deg_value) * 100

                # plotting
                plt.subplot(334)
                plt.plot(x, y, linewidth=0.5)
                plt.plot(x, sec_deg(x), label='2nd degree')
                plt.plot(x, thd_deg(x), label='3th degree')
                plt.plot(x, fou_deg(x), label='4th degree')
                plt.plot(x, thirty_deg(x), label='30th degree')
                plt.plot(x[n_max], four_deg_value[n_max], 'o', color='black', linewidth=2, label='Maximal Value',
                         markerfacecolor='red')
                plt.plot(x[n_min], four_deg_value[n_min], 'o', color='black', linewidth=2, label='Minimal Value',
                         markerfacecolor='green')
                plt.legend()


                # fitting wavelenght 2
                liste = list(range(0, len(wavelength), 3))
                for i in liste:
                    wavelength[i + 2] = np.array(wavelength[i + 2]) - four_deg_value

                # plot fitted
                plt.subplot(335)
                liste = list(range(0, len(wavelength), 3))
                for i in liste:
                    plt.plot(wavelength[i + 1], wavelength[i + 2], label=wavelength[i]['DCBias'])

                plt.plot(x, four_deg_value, label='Fitted data')
                plt.legend(fontsize='small', title='DCBias in V', ncol=2)
                plt.xlabel('Wavelenght in nm')
                plt.ylabel('Measured transmission in dB')
                plt.title('Transmission spectral')

                # finding the minima and maxima
                plt.subplot(336)
                peaks_list = []
                for i in liste:
                    if i < 15:
                        peaks_pos, _ = find_peaks(wavelength[i + 2], height=-4, distance=800)
                        peaks_neg, _ = find_peaks(-wavelength[i + 2], height=35, distance=800)
                        print(peaks_pos, peaks_neg, i)
                        plt.plot(wavelength[i + 1], wavelength[i + 2], label=wavelength[i]['DCBias'])
                        plt.plot(wavelength[i + 1][peaks_pos], wavelength[i + 2][peaks_pos], "x")
                        plt.plot(wavelength[i + 1][peaks_neg], wavelength[i + 2][peaks_neg], "x")
                        peaks_list.append(
                            [wavelength[i + 1][peaks_pos], wavelength[i + 2][peaks_pos]])  # add values for linear fit

                plt.legend(fontsize='small', title='DCBias in V', ncol=2)
                plt.xlabel('Wavelenght in nm')
                plt.ylabel('Measured transmission in dB')
                plt.title('Transmission spectral min/max values')

                # fitting of the linear line
                x = peaks_list[0][0]
                y = peaks_list[0][1]
                poly1d_fn = np.poly1d(np.polyfit(x, y, 1))
                plt.plot(x, y, 'yo', x, poly1d_fn(x), '--k')
                path = str(os.getcwd()).replace("src", "")
                folderpath = (f'{path}/result')
                print(f'{folderpath}'+'/graph_'+ f'{wafer1}_{name1}'+'.png')
                plt.savefig(f'{folderpath}'+'/graph_'+ f'{wafer1}_{name1}'+'.png', dpi=150, bbox_inches='tight')
                plt.show()

                # # calculating the substraction data
                # plt.subplot(337)
                # new = [None] * 23
                # for i in range(0, 23):
                #     new[i] = i
                #
                # liste = list(range(0, len(wavelength), 3))
                # for i in liste:
                #     new[i + 2] = np.array(wavelength[i + 2]) - poly1d_fn(four_deg_value)
                #
                # liste = list(range(0, len(wavelength), 3))
                # for i in liste:
                #     plt.plot(wavelength[i + 1], wavelength[i + 2], label=wavelength[i]['DCBias'])
                #     plt.plot(wavelength[i + 1], new[i + 2], label=wavelength[i]['DCBias'])
                # plt.show()

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
                                  columns=['Lot', 'Wafer', 'Mask', 'TestSite', 'Die Row', 'Die Column', 'Date', 'Name',
                                           'X Value', 'Max Value in dB', 'Rsquare value of the ',
                                           'Design wavelength [nm]', 'Rsquare of IV', 'I at 1',
                                           'I at -1'])
                path = str(os.getcwd()).replace("src", "")
                folderpath = (f'{path}/result')
                df.to_csv(f'{folderpath}'+'/text_'+ f'{wafer1}_{name1}'+'.png', index=False)
        else:
            continue

