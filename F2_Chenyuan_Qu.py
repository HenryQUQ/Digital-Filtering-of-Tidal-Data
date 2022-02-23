import numpy as np
from matplotlib import pyplot as plt
from scipy import signal, optimize

"""
READ ME

This program did the analysis on the velocities of the ocean current in the 'ocean.dat' file.

Please put the 'ocean.dat' file on the same folder with this script file
"""


# Function Part
# Functional functions
def read_data():
    """
    Read the velocity data from 'ocean.dat'.
    :return: list of data with floats
    """
    # find 'ocean.dat' in current path
    data_path = r'./ocean.dat'

    # open data as only-read and save the data with a list called data_str, split by lines.
    with open(data_path, 'r') as d:
        data_str = d.read().splitlines()
        d.close()

    # because the data involves in list is strings, we need to convent them to floats.
    data = []
    for single_data_str in data_str:
        data.append(float(single_data_str))
    return data


def dft(data):
    """
    Discrete Fourier Transform the data
    :param data: the data waiting for fourier transform
    :return: the data after fourier transform
    """
    return np.abs(np.fft.rfft(data))  # need be absolute values, because we only focus the magnitude of the frequency.


def remove_12_filter():
    """
    Create a filter to remove the variation of 12 hours and 25 minutes.
    :return: Filter coordinates
    """
    # use irrnotch function, w0 is the frequency needed to be removed.
    notch_b, notch_a = signal.iirnotch(w0=1 / (12 + 25 / 60), Q=0.23, fs=fs)
    return notch_b, notch_a


def filter_twice(b, a, data):
    """
    Filter the data twice, once forward and once backward.
    :param b: Filter coordinates b
    :param a: Filter coordinates a
    :param data: the raw data waiting to be filtered
    :return: filtered data
    """
    return signal.filtfilt(b, a, data)


def remove_peak_of_12(data):
    """
    The main function to create a filter and filter the data to remove the variation of 12 hours and 25 minutes.
    :param data:the raw data waiting to be filtered
    :return:filtered data
    """
    notch_b, notch_a = remove_12_filter()
    return filter_twice(notch_b, notch_a, data)


def keep_6_filter():
    """
    Create a filter contains high-pass and low-pass, which low-pass is to reduce the noise and high-pass is
    to filter out the low frequency
    :return:the filter coordinates
    """
    Wn = np.array([0.07, 0.18])
    return signal.butter(N=5, Wn=Wn, btype='bandpass', analog=False)


def keep_peak_of_6(data):
    """
    The main function to create a filter and filter the data to only keep the variation of 6.2 hours.
    :param data: raw data of velocity
    :return: data after filtered
    """
    filter_6_b, filter_6_a = keep_6_filter()
    return filter_twice(filter_6_b, filter_6_a, data)


def gauss_function(x, a, x0, sigma):
    """
    Gauss function.
    :param x: x input
    :param a: real constant, height of the curve's peak
    :param x0: real constant, position of the centre of the peak
    :param sigma:non-zero constant, standard deviation
    :return:Gauss function
    """
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


# Plotting Part
def plot_v_vs_t(time_range, data):
    """
    Plot the graph with velocity of the ocean current against the time.
    :param data:raw data of velocity
    :return:True if successive
    """
    time_range = np.arange(0, len(data) * 14, 14)
    plt.plot(time_range, data, '.', label='velocity')
    plt.legend()
    plt.title('Velocity of the Ocean Current vs Time')
    plt.xlabel('Time/min')
    plt.ylabel('Velocity of the Ocean Current/ m*s^(-1)')
    plt.show()
    return True


def plot_fourier_vs_f(frequency, data_fft):
    """
    Plot the graph with frequency of the ocean current against fourier data.
    :param frequency: frequency of the current
    :param data_fft: data after fourier transform
    :return: True if successive
    """
    plt.plot(frequency, data_fft, label='Fourier')
    plt.title("Fourier Amplitude vs Frequency of the Ocean Current")
    plt.xlabel('Frequency of the Ocean Current/ hours^-1')
    plt.ylabel('Fourier Amplitude/ dB')
    plt.legend()
    plt.show()
    return True


def plot_fourier_zoom_in(frequency, data_fft):
    """
    Plot the graph with frequency of the ocean current against fourier data within 0 to 1.
    :param frequency: frequency of the current
    :param data_fft: data after fourier transform
    :return: True if successive
    """
    plt.plot(frequency, data_fft, label='Fourier')
    plt.title("Fourier Amplitude vs Frequency with X-axis Range Between 0 and 1")
    # vertical lines for different peaks
    plt.axvline(0.16, label='6.2 hours', color='aqua')
    plt.axvline(1 / (12 + 60 / 25), label='12 hours and 25 mins', color='violet')
    plt.axvline(0.0416, label='24 hours', color='coral')
    plt.axvline(0.00297, label='336 hours', color='lime')
    plt.xlabel('Frequency of the Ocean Current/ hours^-1')
    plt.ylabel('Fourier Amplitude/ dB')
    plt.xlim(0, 1)
    plt.legend()
    plt.show()
    return True


def plot_fourier_removed_12hours(frequency, data_filtered):
    """
    Plot the graph with frequency of the ocean current against fourier data within 0 to 1
    when the variation of 12 hours has been removed .
    :param frequency: frequency of the current
    :param data_filtered: filtered data after removal of the variation
    :return: True if successive
    """
    plt.plot(frequency, data_filtered, label='Filtered')
    plt.title("Fourier Amplitude vs Frequency Removed 12 Hours Variation")
    plt.xlabel('Frequency of the Ocean Current/ hours^-1')
    plt.ylabel('Fourier Amplitude/ dB')
    plt.axvline(0.0416, label='24 hours', color='coral')
    plt.axvline(0.00297, label='336 hours', color='lime')
    plt.xlim(0, 1)
    plt.legend()
    plt.show()
    return True


def plot_fourier_6(frequency_range, data_fft_6):
    """
    Plot the graph with frequency of the ocean current, which only contains the variation of 6.2 hours,
    against fourier data within 0 to 1
    :param frequency_range: frequency of the current
    :param data_fft_6: filtered data with only variation of 6 hours
    :return: True if succesive
    """
    plt.plot(frequency_range, data_fft_6, label='Filtered')
    plt.title("Fourier Amplitude vs Frequency with Only 12 Hours Variation")
    plt.xlabel('Frequency of the Ocean Current/ hours^-1')
    plt.ylabel('Fourier Amplitude/ dB')
    plt.xlim(0, 1)
    plt.legend()
    plt.show()
    return True


def plot_fourier_6_with_gauss(frequency_range, data_fft_6, fitting_x, gauss_line):
    """
    Plot the graph with frequency of the ocean current, which only contains the variation of 6.2 hours,
    against fourier data within 0 to 1, and the graph of fitting line is also plotted.
    :param frequency_range: frequency of the current
    :param data_fft_6: filtered data with only variation of 6 hours
    :param fitting_x: the x values with range between 0 to 1
    :param gauss_line: fitting gauss line
    :return: True if successive
    """
    plt.plot(frequency_range, data_fft_6, label='Filtered')
    plt.plot(fitting_x, gauss_line, label='Gauss Fitting')
    plt.title("Fourier Amplitude vs Frequency and Gauss Fitting with Only 12 Hours Variation")
    plt.xlabel('Frequency of the Ocean Current/ hours^-1')
    plt.ylabel('Fourier Amplitude/ dB')
    plt.xlim(0, 1)
    plt.legend()
    plt.show()
    return True


# Main body
if __name__ == '__main__':
    # read the data first
    data = read_data()

    # because the data is recorded every 14 minutes
    # the sampling frequency is 60/14 then, with unit of hours
    global fs
    fs = 60 / 14  # in hours

    # plot the raw data between velocity and time
    time_range = np.arange(0, len(data)) * 14 / 60
    plot_v_vs_t(time_range, data)

    # do discrete fourier transform with the raw data
    data_fft = dft(data)

    # plot the fourier amplitude with the frequency
    frequency_range = np.arange(0, len(data_fft)) * fs / len(data_fft)
    plot_fourier_vs_f(frequency_range, data_fft)

    # plot the fourier amplitude with the frequency in the range of 0<frequency<1
    plot_fourier_zoom_in(frequency_range, data_fft)

    # Remove variation on 12 hours and 25 minutes
    filtered_12 = remove_peak_of_12(data)  # get filter data of velocity
    data_fft_12 = dft(filtered_12)

    # plot the fourier amplitude with the frequency when the variable of 12 hours has been removed
    plot_fourier_removed_12hours(frequency_range, data_fft_12)

    # Only keep the variation of the period of 6.2 hours
    filtered_6 = keep_peak_of_6(data)
    data_fft_6 = dft(filtered_6)

    # plot the fourier amplitude with the frequency only with the variable of 6.2 hours
    plot_fourier_6(frequency_range, data_fft_6)

    # create a fitting line with gauss function to fit the fourier amplitude
    popt, pcov = optimize.curve_fit(f=gauss_function, xdata=frequency_range, ydata=data_fft_6)
    fitting_x = np.linspace(0, 1, 1000)
    gauss_line = gauss_function(fitting_x, *popt)

    # plot the fourier amplitude with the frequency only with the variable of 6.2 hours and its fitting line
    plot_fourier_6_with_gauss(frequency_range, data_fft_6, fitting_x, gauss_line)
    exit()
