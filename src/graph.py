def graph(route, save, show, time):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.RankWarning)
        tree = parse(str(route))
        root = tree.getroot()
        plt.figure(figsize=(16, 10))

        # IV Measurement
        plt.subplot(331)
        plt.plot(voltage, current, color='black', marker='o', markeredgecolor='black', markerfacecolor='red')
        plt.yscale('log')
        plt.title('IV-Analysis')
        plt.ylabel('Current in A')
        plt.xlabel('Voltage in V')
        plt.grid('true')
        plt.tight_layout()

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
                peaks_list.append([wavelength[i + 1][peaks_pos], wavelength[i + 2][peaks_pos]])  # add values for linear fit

        plt.legend(fontsize='small', title='DCBias in V', ncol=2)
        plt.xlabel('Wavelenght in nm')
        plt.ylabel('Measured transmission in dB')
        plt.title('Transmission spectral min/max values')

        # fitting of the linear line
        x = peaks_list[0][0]
        y = peaks_list[0][1]
        poly1d_fn = np.poly1d(np.polyfit(x, y, 1))
        plt.plot(x, y, 'yo', x, poly1d_fn(x), '--k')
        plt.savefig('fassung.png', dpi=150, bbox_inches='tight')
        plt.show()