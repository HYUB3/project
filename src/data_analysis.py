import time
import warnings
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
import lmfit
from scipy.signal import find_peaks

# get the data structure
tree = ET.parse('HY202103_D08_02_LION1_DCM_LMZC.xml')
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
# transform to np.array

voltage = np.array(voltage)
current = np.array(current)

current = [abs(i) for i in current]

# plot of the iv measurement
plt.subplot(231)
plt.plot(voltage, current, color='black', marker='o', markeredgecolor='black', markerfacecolor='red')
plt.yscale('log')
plt.title('IV-Analysis')
plt.ylabel('Current in A')
plt.xlabel('Voltage in V')
plt.grid('true')

plt.tight_layout()
plt.show()

# fitting the iv measurement
# first part of funktion
x1 = np.asarray(voltage[0:10])
y1 = np.asarray(current[0:10])


def function(x, a, b, c, d, e):
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e


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


def diode_function(x, a, b, c, d):
    return b * (np.exp((d * x) / (a * c)) - 1)


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


# plotting the iv measurement fitted
plt.subplot(232)
plt.plot(voltage, current, 'x', color='black')
plt.plot(x1, result1.best_fit, '--r')
plt.plot(x2, result2.best_fit, '--g')
plt.yscale('log')
plt.title('IV-Analysis')
plt.ylabel('Current in A')
plt.xlabel('Voltage in V')
plt.grid('true')

plt.tight_layout()
plt.show()

# loading the wavelenght measurement
wavelength = []
for child in root.findall('./ElectroOpticalMeasurements/ModulatorSite/Modulator/PortCombo/WavelengthSweep'):
    wavelength.append(child.attrib)
    for i in child:
        wavelength.append(list(map(float, i.text.split(','))))
liste = list(range(0, len(wavelength), 3))
for i in liste:
    wavelength[i+1] = np.array(wavelength[i+1])
    wavelength[i+2] = np.array(wavelength[i+2])

# plot of the wavelenght measurement
plt.subplot(233)
liste = list(range(0, len(wavelength), 3))
for i in liste:
    plt.plot(wavelength[i + 1], wavelength[i + 2], label=wavelength[i]['DCBias'])
plt.legend(fontsize='small', title='DCBias in V', ncol=2)
plt.xlabel('Wavelenght in nm')
plt.ylabel('Measured transmission in dB')
plt.title('Transmission spectral')
plt.show()

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
for i in range(6):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        start = time.time()
        p = np.poly1d(np.polyfit(x, y, i))
        end = time.time()
        r2 = r2_score(y, p(x)) * 100
        print('Time of calculation', i, 'th degree:', end - start, 's')
        print('Rsquare value in %:', r2, 'of the', i, 'degree polynomal fit ')
        print()
        rsq[str(i)] = r2_score(y, p(x))

# R^2 score 30th degree
r2_30 = r2_score(y, thirty_deg(x))
print('30th degree:', r2_30 * 100)

# 4th degree values
four_deg_value = fou_deg(x)

# extreme values for the 4th degree
n_max = four_deg_value.argmax()
print('Maximal value of the fitted function:', four_deg_value[n_max])
n_min = four_deg_value.argmin()
print('Minimal value of the fitted function:', four_deg_value[n_min])

# plotting
plt.subplot(234)
plt.plot(x, y, linewidth=0.5)
plt.plot(x, sec_deg(x), label='2nd degree')
plt.plot(x, thd_deg(x), label='3th degree')
plt.plot(x, fou_deg(x), label='4th degree')
plt.plot(x, thirty_deg(x), label='30th degree')
plt.plot(x[n_max], four_deg_value[n_max], 'o', color='black', linewidth=2, label='Maximal Value', markerfacecolor='red')
plt.plot(x[n_min], four_deg_value[n_min], 'o', color='black', linewidth=2, label='Minimal Value',
         markerfacecolor='green')
plt.legend()
plt.show()

# fitting wavelenght 2
liste = list(range(0, len(wavelength), 3))
for i in liste:
    wavelength[i + 2] = np.array(wavelength[i + 2]) - four_deg_value

# plot fitted
plt.subplot(235)
liste = list(range(0, len(wavelength), 3))
for i in liste:
    plt.plot(wavelength[i + 1], wavelength[i + 2], label=wavelength[i]['DCBias'])

plt.plot(x, four_deg_value, label='Fitted data')
plt.legend(fontsize='small', title='DCBias in V', ncol=2)
plt.xlabel('Wavelenght in nm')
plt.ylabel('Measured transmission in dB')
plt.title('Transmission spectral')
plt.show()

# finding the minima and maxima
plt.subplot(236)
peaks_list = []
for i in liste:
    if i < 15:
        peaks_pos, _ = find_peaks(wavelength[i+2], height=-4, distance=800)
        peaks_neg, _ = find_peaks(-wavelength[i+2], height=35, distance=800)
        print(peaks_pos, peaks_neg, i)
        plt.plot(wavelength[i + 1], wavelength[i + 2], label=wavelength[i]['DCBias'])
        plt.plot(wavelength[i + 1][peaks_pos], wavelength[i + 2][peaks_pos], "x")
        plt.plot(wavelength[i + 1][peaks_neg], wavelength[i + 2][peaks_neg], "x")
        peaks_list.append([wavelength[i + 1][peaks_pos], wavelength[i + 2][peaks_pos]]) #add values for linear fit
plt.legend(fontsize='small', title='DCBias in V', ncol=2)
plt.xlabel('Wavelenght in nm')
plt.ylabel('Measured transmission in dB')
plt.title('Transmission spectral min/max values')
plt.show()

#fitting of the linear line
x = peaks_list[0][0]
y = peaks_list[0][1]
poly1d_fn = np.poly1d(np.polyfit(x, y, 1))
plt.plot(x,y, 'yo', x, poly1d_fn(x), '--k')
plt.show()

#calculating the substraction data
plt.plot(wavelength[19], poly1d_fn(wavelength[19]))







