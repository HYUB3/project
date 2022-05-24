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

def plotting():
    i = 0
    for test in dataframe_data:
        plt.subplot(231)
        plt.plot(test['Voltage'][i], abs(np.asarray(test['Current'][i])), color='black', marker='o', markeredgecolor='black',
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
        for j in liste:
            plt.plot(test['Wavelength'][i][i+1], test['Wavelength'][i][j + 2], label=test['Wavelength'][i][j]['DCBias'])
        plt.legend(fontsize='small', title='DCBias in V', ncol=2)
        plt.xlabel('Wavelength in nm')
        plt.ylabel('Measured transmission in dB')
        plt.title('Transmission spectral')

        plt.subplot(224)
        for j in liste:
            if test['Wavelength'][i][j]['DCBias'] == '0.0':
                print('n')
                plt.plot(test['Wavelength'][i][j + 1], test['Wavelength'][i][j + 2],
                        label=test['Wavelength'][i][j]['DCBias'])
            else:
                print(test['Wavelength'][i][j]['DCBias'])
        plt.show()

plotting()


