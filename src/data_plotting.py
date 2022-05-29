import matplotlib.pyplot as plt
import numpy as np
import lmfit
import matplotlib.pyplot as plt
from loading import *
from loadingxml import *


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
            print('innerhalb der schkleife')
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






            plt.show()

        i = i + 1



plotting()
