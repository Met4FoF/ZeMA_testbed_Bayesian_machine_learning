# coding: utf-8

# # Data Import and Visualisation
#
# ### ZeMA testbed dataset

# Testbed at ZeMa is for condition monitoring and lifetime prognoses of the
# electro-mechanical cylinders (EMC).
#
# Data from these 11 sensors is available for estimation of remainig lifetime:
# - Microphone
# - Vibration plain bearing
# - Vibration piston rod
# - Vibration ball bearing
# - Axial force
# - Pressure
# - Velocity
# - Active current
# - Motor current phase 1
# - Motor current phase 2
# - Motor current phase 3
#
# Through this code sensors are represented with numbers 0-10 in same order
# sensors are listed.  Working cycle consists of forward stroke, waiting time
# and return stroke, and available data is from measurements of 1 second of
# return stroke with frequency being 2 kHz [1].
#
# <img src="pictures/stroke.png" width="800">
#
# This specific data consists of 6291 EMC cycles where first cycle represents
# fully functional part and in the last cycle the EMC is fully broken (has no
# remaining lifetime).  Lifetime is represented with percentage values where 0
# % means being brand-new and 100 % being broken.

# ### Importing the data
#
# Data is downloaded from the https://zenodo.org/record/1326278#.XQJMGI_gqUk
# website and following informations about the structure of the data are listed
# there:
# - data saved in HDF5 file as a 3D-matrix
# - one row represents one second of the return stroke of one working cycle
# (6292 rows: 6292 cycles)
# - one column represents one datapoint of the cycle, that is resampled to 2
# kHz (2000 columns)
# - one page represent one sensor (11 pages: 11 sensors)
#
# Last cycle of downloaded data is cut because it has no useful informations.
#
# Data is imported into the list of length 11, where each element consists of
# dataframe whose rows are time-series of cycles.

import datetime
import math
# ## Note!!!
# **Data downloaded from the link have to be located in the same folder as
# Jupyter Notebooks in order to import the data.  Name of the file must remain
# the same (Sensor_data_2kHz.h5).  You can do it on your own or it can be done
# automatically**
import os
import pickle
import warnings

import arviz as az
# os.environ['MKL_THREADING_LAYER'] = 'GNU'
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm
import requests
import theano.tensor as tt
from scipy.interpolate import griddata
from scipy.stats import pearsonr
from sklearn.covariance import EmpiricalCovariance, GraphicalLassoCV, MinCovDet
from sklearn.decomposition import KernelPCA, PCA
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    explained_variance_score,
    max_error,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
# list of classifiers
from sklearn.neighbors import KNeighborsClassifier
# scaling of input data
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from theano import shared
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
plt.style.use('seaborn-darkgrid')
font = {'family': 'Times New Roman',
        'weight': 'normal',
        'size': 13}
mpl.rc('font', **font)


# **********************************************************************************************************************
class Data:

    # class properties
    filename_ = ''
    sensorADC_ = []
    sensor_ = []
    time_ = []
    nb_cycles_ = None
    nb_measurements_ = None
    nb_sensors_ = 11
    conversion_table_ADC_SI_ = {
        'offset': [0, 0, 0, 0, 0.00488591, 0.00488591, 0.00488591, 0.00488591, 1.36e-2, 1.5e-2, 1.09e-2],
        'gain': [5.36e-9, 5.36e-9, 5.36e-9, 5.36e-9, 3.29e-4, 3.29e-4, 3.29e-4, 3.29e-4, 8.76e-5, 8.68e-5, 8.65e-5],
        'b': [1, 1, 1, 1, 1, 1, 1, 1, 5.299641744, 5.299641744, 5.299641744],
        'k': [250, 1, 10, 10, 1.25, 1, 30, 0.5, 2, 2, 2]
    }
    units_ = ['[Pa]', '[g]', '[g]', '[g]', '[kN]', '[bar]', '[mm/s]', '[A]', '[A]', '[A]', '[A]']

    # __________________________________________________________________________________________________________________
    def __init__(self):
        pass

    # __________________________________________________________________________________________________________________
    def download_file(self, url):
        """ downloads sensor measurement files"""

        self.filename_ = url.split('/')[-1]
        if os.path.isfile(self.filename_):
            print("Data already exist.")
        else:
            print("DOWNLOAD DATA: starting...")
            r = requests.get(url, stream=True)
            # Compute parameters for download and corresponding progress bar.
            total_length = int(r.headers.get('content-length'))
            chunk_size = 512 * 1024
            f = open(self.filename_, 'wb')
            for chunk in tqdm(r.iter_content(chunk_size=chunk_size), total=math.ceil(total_length // chunk_size),
                              unit='KB', unit_scale=True):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
            f.close()
            print("DOWNLOAD DATA: done!")

    # __________________________________________________________________________________________________________________
    def import_data(self):
        """ imports data from downloaded h5 file and stores it to sensorADC"""

        print("IMPORT DATA: starting...")
        f = h5py.File(self.filename_, 'r')  # Importing the h5 file.
        print("IMPORT DATA: done!")
        a_group_key = list(f.keys())[0]
        data = list(f[a_group_key])  # Transforming data into list
        sensorADC = []  # Initialising a list "sensor" and
        for i in range(self.nb_sensors_):  # Filling it with data from all sensors
            sensorADC.append(pd.DataFrame(data[i][:][:]))
        for i in range(self.nb_sensors_):
            sensorADC[i] = sensorADC[i].iloc[:, :-1]  # Cuting the last cycle because it contains all zero elements.

        print(
            "OBSERVED DATA: %s, where %s represents the number of measurements in time and %s represents the number of cycles." % (
                np.shape(sensorADC[0]), np.shape(sensorADC[0])[0], np.shape(sensorADC[0])[1]))

        self.nb_cycles_ = np.shape(sensorADC[0])[1]
        self.nb_measurements_ = np.shape(sensorADC[0])[0]
        self.sensorADC_ = sensorADC
        return sensorADC

    # __________________________________________________________________________________________________________________
    def convert_ADC_to_SI_units(self):
        """ converts ADC measurements to SI units according to the given table (offset, gain, b, k, units)"""

        sensor = [0] * len(self.sensorADC_)

        for i in range(len(self.sensorADC_)):
            sensor[i] = ((self.sensorADC_[i] * self.conversion_table_ADC_SI_['gain'][i]) +
                         self.conversion_table_ADC_SI_['offset'][i]) * self.conversion_table_ADC_SI_['b'][i] * \
                        self.conversion_table_ADC_SI_['k'][i]

        self.sensor_ = sensor

        return sensor

    # __________________________________________________________________________________________________________________
    def transform_xaxis_to_time_domain(self):
        """ transforms x-axis from sensor measurement data to time domain"""

        n_of_samples = np.shape(self.sensorADC_[0])[0]
        # print("n_of_samples = %s" % n_of_samples)
        time_ = [i * 0.0005 for i in list(range(np.size(self.sensorADC_[0].iloc[:, 0])))]
        self.time_ = time_
        return time_

    # __________________________________________________________________________________________________________________
    def get_sensor_trace(self, sensor_number=0, cycle_number=3000):
        """ gets a specific one cycle trace from one sensor"""

        assert sensor_number <= self.nb_sensors_ & cycle_number <= self.nb_cycles_
        return self.sensor_[sensor_number][cycle_number]

    # __________________________________________________________________________________________________________________
    def plot_ADC_traces(self, cycle_number):
        """ plot sensor ADC traces in time domain"""

        plt.figure(figsize=(20, 13))
        plt.title("ADC measured values")
        plt.subplot(431)
        plt.plot(self.time_, self.sensorADC_[0].iloc[:, cycle_number])
        plt.ylabel("Microphone", fontsize=10)
        plt.subplot(432)
        plt.plot(self.time_, self.sensorADC_[1].iloc[:, cycle_number])
        plt.ylabel("Vibration plain bearing", fontsize=10)
        plt.title("Cycle %s" % (cycle_number + 1), fontsize=15)
        plt.subplot(433)
        plt.plot(self.time_, self.sensorADC_[2].iloc[:, cycle_number])
        plt.ylabel("Vibration piston rod", fontsize=10)
        plt.subplot(434)
        plt.plot(self.time_, self.sensorADC_[3].iloc[:, cycle_number])
        plt.ylabel("Vibration ball bearing", fontsize=10)
        plt.subplot(435)
        plt.plot(self.time_, self.sensorADC_[4].iloc[:, cycle_number])
        plt.ylabel("Axial force", fontsize=10)
        plt.subplot(436)
        plt.plot(self.time_, self.sensorADC_[5].iloc[:, cycle_number])
        plt.ylabel("Pressure", fontsize=10)
        plt.subplot(437)
        plt.plot(self.time_, self.sensorADC_[6].iloc[:, cycle_number])
        plt.ylabel("Velocity", fontsize=10)
        plt.subplot(438)
        plt.plot(self.time_, self.sensorADC_[7].iloc[:, cycle_number])
        plt.ylabel("Active current", fontsize=10)
        plt.subplot(439)
        plt.plot(self.time_, self.sensorADC_[8].iloc[:, cycle_number])
        plt.ylabel("Motor current phase 1", fontsize=10)
        plt.subplot(4, 3, 10)
        plt.plot(self.time_, self.sensorADC_[9].iloc[:, cycle_number])
        plt.ylabel("Motor current phase 2", fontsize=10)
        plt.subplot(4, 3, 11)
        plt.plot(self.time_, self.sensorADC_[10].iloc[:, cycle_number])
        plt.ylabel("Motor current phase 3", fontsize=10)
        plt.xlabel("Time [s] - all x axes", fontsize=15)
        plt.show()

    # __________________________________________________________________________________________________________________
    def plot_traces(self, cycle_number):
        """ plot sensor traces in time domain (SI units)"""

        plt.figure(figsize=(20, 20))
        plt.title("Converted measured values (SI units)")
        plt.subplot(411)
        plt.plot(self.time_, self.sensor_[0].iloc[:, cycle_number])
        plt.ylabel("Microphone " + str(self.units_[0]))
        plt.title("Cycle %s" % (cycle_number + 1))
        plt.subplot(412)
        plt.plot(self.time_, self.sensor_[1].iloc[:, cycle_number])
        plt.ylabel("Vibration plain bearing " + str(self.units_[1]))
        plt.subplot(413)
        plt.plot(self.time_, self.sensor_[2].iloc[:, cycle_number])
        plt.ylabel("Vibration piston rod " + str(self.units_[2]))
        plt.subplot(414)
        plt.plot(self.time_, self.sensor_[3].iloc[:, cycle_number])
        plt.ylabel("Vibration ball bearing " + str(self.units_[3]))
        plt.xlabel('self.time_ [s]')
        plt.figure(figsize=(20, 20))
        plt.title("Converted measured values (SI units)")
        plt.subplot(411)
        plt.plot(self.time_, self.sensor_[4].iloc[:, cycle_number])
        plt.ylabel("Axial force " + str(self.units_[4]))
        plt.title("Cycle %s" % (cycle_number + 1))
        plt.subplot(412)
        plt.plot(self.time_, self.sensor_[5].iloc[:, cycle_number])
        plt.ylabel("Pressure " + str(self.units_[5]))
        plt.subplot(413)
        plt.plot(self.time_, self.sensor_[6].iloc[:, cycle_number])
        plt.ylabel("Velocity " + str(self.units_[6]))
        plt.subplot(414)
        plt.plot(self.time_, self.sensor_[7].iloc[:, cycle_number])
        plt.ylabel("Active current " + str(self.units_[7]))
        plt.xlabel('Time [s]')
        plt.figure(figsize=(20, 17))
        plt.title("Converted measured values (SI units)")
        plt.subplot(311)
        plt.plot(self.time_, self.sensor_[8].iloc[:, cycle_number])
        plt.ylabel("Motor current phase 1 " + str(self.units_[8]))
        plt.title("Cycle %s" % (cycle_number + 1))
        plt.subplot(312)
        plt.plot(self.time_, self.sensor_[9].iloc[:, cycle_number])
        plt.ylabel("Motor current phase 2 " + str(self.units_[9]))
        plt.subplot(313)
        plt.plot(self.time_, self.sensor_[10].iloc[:, cycle_number])
        plt.ylabel("Motor current phase 3 " + str(self.units_[10]))
        plt.xlabel('Time [s]')
        plt.show()

    # __________________________________________________________________________________________________________________
    def get_data_subset(self, sub_sampling_factor=10):
        """ gets a subset of the current data, ie consider only part of the cycles using subsampling """

        for i in range(self.nb_sensors_):
            # self.sensor_[i] = self.sensor_[i].loc[:, 0::sub_sampling_factor]
            self.sensorADC_[i] = self.sensorADC_[i].loc[:, 0::sub_sampling_factor]
            new_index = [i for i in range(0, len(self.sensorADC_[0].columns))]
            # self.sensor_[i].columns = new_index
            self.sensorADC_[i].columns = new_index

        self.nb_cycles_ = np.shape(self.sensorADC_[0])[1]
        self.nb_measurements_ = np.shape(self.sensorADC_[0])[0]

        print("WORKING DATA: %s, where %s represents number of measurements in time \
        and %s represents number of cycles." % (
            np.shape(self.sensorADC_[0]), np.shape(self.sensorADC_[0])[0], np.shape(self.sensorADC_[0])[1]))


# **********************************************************************************************************************
class Model:

    # class properties
    target_ = []
    target_train_vector_ = []
    target_test_vector_ = []
    sensor_train_ = []
    sensor_test_ = []
    class_target_train_vector_ = []
    class_target_test_vector_ = []
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    X_scaler = []
    Y_scaler = []
    nb_selected_features_per_sensor = []
    classification_metrics = {
        'accuracy': [],
        'macro': {
            'precision': [],
            'recall': [],
            'f-score': []
        },
        'micro': {
            'precision': [],
            'recall': [],
            'f-score': []
        }
    }

    # __________________________________________________________________________________________________________________
    def __init__(self):
        pass

    # __________________________________________________________________________________________________________________
    def create_target_dict(self, data):
        """ target vector is created, which assumes linear correlation between cycle number and the degree of wear of the electro-mechanical cylinder """

        target = list(np.zeros(data.nb_cycles_))  # Making the target list which takes into account number of cycles,
        # which-
        for i in range(data.nb_cycles_):  # goes from 0 to 100, and has number of elements same as number
            # of cycles.
            target[i] = (i / (data.nb_cycles_ - 1)) * 100

        target_matrix = pd.DataFrame(target)
        self.target_ = target_matrix

    # __________________________________________________________________________________________________________________
    def split_dataset(self, data, percentage_train=85):
        """ splits dataset into train/test """

        self.target_train_vector_, self.target_test_vector_ = train_test_split(self.target_,
                                                                               test_size=1 - percentage_train / 100.)
        self.target_ = list(self.target_train_vector_[0])
        print("Number of cycles selected for (training / testing ): (", self.target_train_vector_.shape[0], " / ",
              self.target_test_vector_.shape[0], ")")

        sensor_train = [0] * data.nb_sensors_
        sensor_test = [0] * data.nb_sensors_

        for i in range(data.nb_sensors_):
            sensor_train[i] = data.sensor_[i].loc[:, self.target_train_vector_.index]

        self.sensor_train_ = sensor_train
        # print("Traning data for one sensor has dimensions: ", self.sensor_train_[10].shape,",  ('sensor_train') ")
        # print("and it's target vector has length: ", self.target_train_vector_.shape,",        ('target_train_vector')")

        for i in range(data.nb_sensors_):
            sensor_test[i] = data.sensor_[i].loc[:, self.target_test_vector_.index]

        self.sensor_test_ = sensor_test
        # print("Testing data for one sensor has dimensions: ", self.sensor_test_[10].shape,",   ('sensor_test') ")
        # print("and it's target vector has length: ", self.target_test_vector_.shape,",         ('target_test_vector')")

    # __________________________________________________________________________________________________________________
    def select_percentage_of_fourier_spectrum_per_sensor(self, data, n_of_samples, frequencies=[],
                                                         percentage_features=10):
        """ does fast Fourier transform and chooses N% of sprectrum with highest average of absolute values for each sensor independently. Average of absolute values for one frequency is calculated through all cycles.      
        Inputs
        - data from one sensor <sensor>,                                 
        - number of samples <n_of_samples>,                                    
        - percentage of features to choose <percentage_features> 
        Outputs
        - <freq_of_sorted_values> matrix sized [1, N% of features (amplitudes)] where elements are frequencies which are choosen and they are labels for second output from this function.
        - <sorted_values_matrix> sized [number of cycles, N% of features (amplitudes)] where row represents one cycle and columns are sorted by the average of absolute vales for each frequency (column)."""

        x_measurements = range(data.shape[0])  # Number of measurements samples in time period.
        x = np.true_divide(x_measurements, n_of_samples)  # Time values, used as real time axis.
        freq = np.fft.rfftfreq(x.size, 0.0005)  # Frequency axis, can be used for plotting in frequency domain.
        fft_amplitudes = np.fft.rfft(data, n_of_samples, 0)  # Nd-array of amplitudes after fourier transform.
        fft_matrix = pd.DataFrame(fft_amplitudes)  # Transforming amplitudes into data frame (matrix)-
        # -where one column represents amplitudes of one--cycle.
        # Transposing to matrix where rows are cycles
        fft_matrix = fft_matrix.transpose()
        # n_rows, n_columns = np.shape(fft_matrix)

        # print("Number of cycles is: %s, and number of features is: %s" % (n_rows, n_columns))
        fft_matrix.columns = freq  # Column labels are frequencies.

        if len(frequencies) == 0:  # (train case)
            # Calculating the average of absolute values for each frequency
            # (column).
            absolute_average_values_from_columns = (np.abs(fft_matrix)).mean()

            # Sorting the fft_matrix by the average of absolute vales for each
            # frequency (column).
            fft_matrix = fft_matrix.reindex((np.abs(fft_matrix)).mean().sort_values(ascending=False).index, axis=1)

            # Taking first N percent columns from sorted fft_matrix.
            sorted_values_matrix = fft_matrix.iloc[:, :round((percentage_features / 100.0) * len(freq))]

            # n_rows, n_columns = np.shape(sorted_values_matrix)
            # print("Number of cycles is: %s, and number of selected features is: %s" % (n_rows, n_columns))
            # print(np.shape(sorted_values_matrix))

            # Information about the selected frequencies are columns in sorted
            # data frame.
            freq_of_sorted_values = (pd.DataFrame(sorted_values_matrix.columns)).transpose()
            # print("First 10 selected frequencies are: %s" % freq_of_sorted_values.values[:,:10])

            sorted_values_matrix.columns = range(
                round((percentage_features / 100.0) * len(freq)))  # Resetting the column labels.
            # print("---------------------------------------------------------------------------------")
            # Output "sorted_values_matrix" is data frame whose rows-
            # -are cycles and columns are selected frequencies.  For example,-
            # -value at position (i,j) is amplitude for frequency j in cycle i.
        else:  # (test case)
            # print("Frequencies are the same as in the traning data. First 10 of them: %s" % frequencies.values[:,:10])
            sorted_values_matrix = fft_matrix.loc[:, frequencies.loc[0, :]]
            n_rows, n_columns = np.shape(sorted_values_matrix)
            # print("Number of cycles is: %s, and number of selected features is: %s " % (n_rows, n_columns))
            sorted_values_matrix.columns = range(len(sorted_values_matrix.columns))
            freq_of_sorted_values = []

        return freq_of_sorted_values, sorted_values_matrix

    # __________________________________________________________________________________________________________________
    def feature_extractor(self, method='fft', frequencies=[], percentage_features=10):
        """ uses function select_percentage_of_fourier_spectrum_per_sensor and iterates over the whole set of sensors"""

        if method == 'fft':
            n_of_samples = np.shape(self.sensor_train_[0])[0]
            if len(frequencies) == 0:
                # Initialising the list with 11 elements, which are data frames
                # "sorted_value_matrix" from each sensor.
                freq_of_sorted_values = [0] * len(self.sensor_train_)
                sorted_values_from_all_sensors = [0] * len(self.sensor_train_)

                for i in range(len(self.sensor_train_)):
                    # print("Sensor number %s" % i)
                    # print("---------------------------------------------------------------------------------")
                    freq_of_sorted_values_, \
                    sorted_values_from_all_sensors_ = self.select_percentage_of_fourier_spectrum_per_sensor(
                        self.sensor_train_[i],
                        n_of_samples,
                        frequencies,
                        percentage_features)
                    freq_of_sorted_values[i] = freq_of_sorted_values_
                    sorted_values_from_all_sensors[i] = sorted_values_from_all_sensors_
            else:
                # Storing selected features from the test data into a list
                # "sorted_values_test"
                sorted_values_from_all_sensors = [0] * len(self.sensor_test_)
                for i in range(len(self.sensor_test_)):
                    # print("Sensor number %s" % i)
                    # print("---------------------------------------------------------------------------------")
                    _, \
                    sorted_values_from_all_sensors_ = self.select_percentage_of_fourier_spectrum_per_sensor(
                        self.sensor_test_[i],
                        n_of_samples,
                        frequencies[i],
                        percentage_features)
                    sorted_values_from_all_sensors[i] = sorted_values_from_all_sensors_

                freq_of_sorted_values = []
        else:
            print('Extractor method not implemented yet')
            freq_of_sorted_values, sorted_values_from_all_sensors, n_of_samples = [], [], []

        return freq_of_sorted_values, sorted_values_from_all_sensors, n_of_samples

    # __________________________________________________________________________________________________________________
    def get_extracted_features_for_all_sensors(self, sorted_values_from_all_sensors):
        """ retrieve the extracted features for the whole set of sensors and stores it in a design matrix with columns corresponding to the sensor number and rows to the features """

        pass

    # __________________________________________________________________________________________________________________
    def feature_selection(self, data, sorted_values_from_all_sensors, n_of_samples, n_of_features=500,
                          percentage_features=10, method='pearson_correlation'):
        """ operates feature selection using highest Pearson correlation coefficients"""

        n_features_for_select = 0
        for i in range(len(sorted_values_from_all_sensors)):
            n_features_for_select = n_features_for_select + int(len(sorted_values_from_all_sensors[i].iloc[0][:]))

        # Defining how much of features with biggest Pearson correlation
        # coeff.  will be selected.
        # n_of_features = int(input("How many features out of %s you want to select (recommended is 500): " % n_features_for_select))
        # print("Dimension of target matrix is: ",                    self.target_train_vector_.shape)
        # print("Dimension of amplitude matrix for one sensor is:",   sorted_values_from_all_sensors[0].iloc[:][:].shape)

        # Making list for correlation coefficients
        corr = list(range(data.nb_sensors_))
        p_value = list(range(data.nb_sensors_))

        # Making sublists in "corr" for each sensor
        for j in range(data.nb_sensors_):
            corr[j] = list(range(round((percentage_features / 100.0) * n_of_samples / 2)))
            p_value[j] = list(range(round((percentage_features / 100.0) * n_of_samples / 2)))

        # Calculating correlation coefficients for each column of each sensor
        # with respect to target.
        for j in range(data.nb_sensors_):
            for i in range(round((percentage_features / 100.0) * n_of_samples / 2)):
                corr[j][i], p_value[j][i] = pearsonr(np.abs(sorted_values_from_all_sensors[j].iloc[:][i]),
                                                     self.target_train_vector_[0])
        # matrix_corr_coeff = np.transpose(pd.DataFrame(corr))# Transforming
        # list of correlation coefficients to data frame.
        # Transforming list of correlation coefficients to numpy array
        corr_array = np.array(corr)
        # print("Array of correlation coefficients has size:", corr_array.shape)

        # sensor_n is the index of the sensor number.
        # feature_n is the index of the feature number for each sensor number.
        # TODO: check if balanced data is better for bayesian fusion
        # sensor_n, feature_n = self.largest_indices(corr_array, n_of_features)
        sensor_n, feature_n = self.balanced_largest_indices_per_sensor(corr_array, n=50)

        # print("Sensor indices of location of features in <sorted_values_from_all_sensors> matrix: ", sensor_n)
        # print("Column indices of location of features in <sorted_values_from_all_sensors> matrix: ", feature_n)
        return sensor_n, feature_n

    # __________________________________________________________________________________________________________________
    def arrange_features(self, sorted_values_from_all_sensors, data, sensor_n, feature_n, flag='train'):
        """ initialises a list of best features. 11 sublists containing features from each sensor, respectively. Merges sublists into one list with all elements and finally works with absolute values."""

        top_n_features = [[], [], [], [], [], [], [], [], [], [], []]
        transform_matrix = np.zeros((data.nb_sensors_, len(sensor_n)), dtype=float)

        for i in range(data.nb_sensors_):
            for j in range(len(sensor_n)):
                if sensor_n[j] == i:
                    top_n_features[i].append(sorted_values_from_all_sensors[i].iloc[:][feature_n[j]])
                    transform_matrix[i, j] = 1

        for i in range(data.nb_sensors_):
            for j in range(len(top_n_features[i])):
                top_n_features[i][j] = list(top_n_features[i][j])

        # Merging sublists into one list with all elements.
        top_n_together = [j for i in top_n_features for j in i]

        top_n_together_matrix = np.transpose(pd.DataFrame(top_n_together))
        # print(type(top_n_together_matrix), "\n")

        # Continue working with absolute values.
        abs_top_n_together_matrix = np.abs(top_n_together_matrix)
        if flag == 'train':
            percentage = list(range(data.nb_sensors_))
            k = 0
            tmp = np.zeros((1, data.nb_sensors_))
            for i in range(data.nb_sensors_):
                # print(top_n_features_matrix.shape)
                tmp[0][i] = len(top_n_features[i])
                print("Number of features from sensor %2.0f is: %3.0f or  %4.2f %%" % (
                    i, len(top_n_features[i]), len(top_n_features[i]) / len(sensor_n) * 100))
                percentage[i] = len(top_n_features[i])
                k = k + len(top_n_features[i]) / len(sensor_n) * 100
            self.nb_selected_features_per_sensor = tmp
            print("----------------------------------------------------")
            print("                                             %4.2f" % (k))
        return abs_top_n_together_matrix, transform_matrix

    # __________________________________________________________________________________________________________________
    def create_classification_dataset(self, feature_matrix, feature_matrix_test):
        """ create standard classification inputs/outputs. Definition of new target with rounding to first higher number."""

        self.class_target_train_vector_ = np.ceil(self.target_train_vector_[0])

        for i in self.class_target_train_vector_.index:
            if self.class_target_train_vector_[i] == 0:
                self.class_target_train_vector_[i] = 1  # Fixing the zero element.

        self.class_target_test_vector_ = np.ceil(self.target_test_vector_[0])

        for i in self.class_target_test_vector_.index:
            if self.class_target_test_vector_[i] == 0:
                self.class_target_test_vector_[i] = 1  # Fixing the zero element.

        self.X_train = np.array(feature_matrix)  # Feature matrix.
        self.y_train = np.array(self.class_target_train_vector_)  # Target vector.
        self.X_test = np.array(feature_matrix_test)  # Feature matrix.
        self.y_test = np.array(self.class_target_test_vector_)  # Target vector.

    # __________________________________________________________________________________________________________________
    def mahalanobis_distance_classifier(self):
        """ classifier using Mahalanobis distance criterion"""

        unique_elements, counts = np.unique(self.y_train, return_counts=True)
        # Initialising list for Mahalanobis distance calculation.
        X_train_splitted = list(np.zeros(len(np.unique(self.y_train))))
        # Initialising 100 matrices (one for each class label) in list
        # "X_train_splitted".
        #   - Counts[i] number of rows for each class.
        #   - Y_train.shape[1] number of columns (same for every class).
        for i in range(len(counts)):
            X_train_splitted[i] = np.zeros((counts[i], self.X_train.shape[1]))
            # Each matrix is data that belongs to one class represented by
        # "unique_elements" array, respectively.
        # Filling matrices for each class with coresponding observations.
        # This loop checks all rows in training data-set, then checks class for
        # that row, and based on that class,
        # puts that row into different groups.
        for j in np.int_(unique_elements):
            k = 0
            for i in range(len(self.y_train)):
                if self.y_train[i] == j:
                    X_train_splitted[int(np.where(unique_elements == j)[0])][k, :] = self.X_train[i, :]  # j-1
                    k = k + 1

        X_mahal_distances = np.zeros((len(self.X_test), len(unique_elements)))

        for i in np.int_(unique_elements):
            robust_cov = MinCovDet().fit(X_train_splitted[int(np.where(unique_elements == i)[0])])
            X_mahal_distances[:, int(np.where(unique_elements == i)[0])] = robust_cov.mahalanobis(self.X_test)

        print(X_mahal_distances.shape)
        # This part is working only if classes are like in this case, going from
        # 1 to 100.
        # Column index in X_mahal_distances matrix is going from 0 to 99, and
        # location on column axis is indicating a class.
        predicted = np.argmin(X_mahal_distances, axis=1)  # Finds the indices of minimum values from each row.
        predicted_class = predicted + 1

        return predicted_class

    # __________________________________________________________________________________________________________________
    def classification(self, method='LDA', plot=True):

        if method == 'LDA':
            clf = LinearDiscriminantAnalysis(n_components=3, priors=None, shrinkage=None, solver='eigen')
            X_train_transform = clf.fit_transform(self.X_train, self.y_train)
            predictions = clf.predict(self.X_test)
            if plot:
                plt.figure(figsize=(20, 10))
                plt.subplot(121)
                plt.title("Two features correlating", fontsize=18)
                plt.scatter(self.X_train[:, 9], self.X_train[:, 69], c=self.y_train, cmap="viridis")
                plt.xlabel("Feature 10", fontsize=15)
                plt.ylabel("Feature 70", fontsize=15)
                plt.colorbar().set_label('% of wear', fontsize=15, rotation=90)
                plt.subplot(122)
                plt.title("First two disriminant functions for traning data", fontsize=18)
                plt.scatter(X_train_transform[:, 0],
                            X_train_transform[:, 1],
                            c=self.y_train,
                            cmap="viridis")
                plt.xlabel("First discriminant function", fontsize=15)
                plt.ylabel("Second discriminant function", fontsize=15)
                plt.show()
                # 3D plot
                fig = plt.figure(figsize=(18, 12))
                ax = fig.gca(projection='3d')
                p = ax.scatter(X_train_transform[:, 0],
                               X_train_transform[:, 1],
                               X_train_transform[:, 2],
                               c=self.y_train,
                               cmap="viridis")
                ax.set_xlabel("First discriminant function", fontsize=15)
                ax.set_ylabel("Second discriminant function", fontsize=15)
                ax.set_zlabel("Third discriminant function", fontsize=15)
                ax.set_title("First three disriminant functions for traning data", fontsize=18)
                fig.colorbar(p).set_label('% of wear', fontsize=15, rotation=90)
                plt.show()
        elif method == 'Mahalanhobis_distance':
            predictions = self.mahalanobis_distance_classifier()
            clf = []
        else:
            print('classifier not implemented yet')

        return clf, predictions

    # __________________________________________________________________________________________________________________
    def dimensionality_reduction(self, method='LDA', plot=True, n_comp=20):
        """ reduces the dimension of input features prior to classification"""

        # Preparation of features for Mahalanobis distance classification
        # method.

        if method == 'LDA':
            # Applying LDA dimensionality reduction
            dim_reduction = LinearDiscriminantAnalysis(n_components=n_comp, priors=None, shrinkage=None, solver='eigen',
                                                       store_covariance=False, tol=1e-4)
            self.X_train = dim_reduction.fit_transform(self.X_train, self.y_train)
            self.X_test = dim_reduction.transform(self.X_test)
        elif method == 'PCA':
            dim_reduction = PCA(n_components=10)
            self.X_train = dim_reduction.fit_transform(self.X_train, self.y_train)
            self.X_test = dim_reduction.transform(self.X_test)
            print("PCA explained variance in %%: ", dim_reduction.explained_variance_ratio_ * 100)
            print("PCA singular values:", dim_reduction.singular_values_)
        elif method == 'Kernel_PCA':
            dim_reduction = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
            self.X_train = dim_reduction.fit_transform(self.X_train, self.y_train)
            self.X_test = dim_reduction.transform(self.X_test)
        else:
            print('algorithm for dimensionality reduction not implemented yet')

        print("Number of features in data is:                       ", self.X_train.shape[1])
        print("Number of observations in training data is:          ", self.X_train.shape[0])
        print("Number of observations in test data is:              ", self.X_test.shape[0])
        print("X_train dimensions after dimensionality reduction:   ", self.X_train.shape)
        print("X_test dimensions after dimensionality reduction:    ", self.X_test.shape)

        if plot:
            plt.figure(1)
            plt.figure(figsize=(20, 10))
            plt.subplot(121)
            plt.scatter(self.X_train[:, 0], self.X_train[:, 1], c=self.y_train, cmap="viridis")
            plt.title('Train data', fontsize=22)
            plt.xlabel("First discriminant function", fontsize=18)
            plt.ylabel("Second discriminant function", fontsize=18)
            plt.subplot(122)
            plt.scatter(self.X_test[:, 0], self.X_test[:, 1], c=self.y_test, cmap="viridis")
            plt.title('Test data (real class labels - not classified yet)', fontsize=22)
            plt.xlabel("First discriminant function", fontsize=18)
            plt.ylabel("Second discriminant function", fontsize=18)
            plt.show()

    # __________________________________________________________________________________________________________________
    def get_regression_scores(self, clf, predictions=[]):
        """ computes scores for regression"""

        if len(predictions) == 0:
            predictions = clf.predict(self.X_test)

        print('-' * 80)
        print('REGRESSION METRICS')
        print('-' * 80)
        print('     explained_variance_score        : {:10.2f}'.format(explained_variance_score(self.y_test, predictions)))
        print('     max_error                       : {:10.2f}'.format(max_error(self.y_test, predictions)))
        print('     mean_absolute_error             : {:10.2f}'.format(mean_absolute_error(self.y_test, predictions)))
        print('     mean_squared_error              : {:10.2f}'.format(mean_squared_error(self.y_test, predictions)))
        # print('     mean_squared_log_error          : {:10.2f}'.format(mean_squared_log_error(self.y_test, predictions)))
        print('     median_absolute_error           : {:10.2f}'.format(median_absolute_error(self.y_test, predictions)))
        print('     r2_score                        : {:10.2f}'.format(r2_score(self.y_test, predictions)))
        print('-' * 80)

    # __________________________________________________________________________________________________________________
    def get_classification_scores(self, clf, predictions=[], plot=False):
        """ computes scores for classification"""

        # print("Size of training dataset is: ", self.X_train.shape)
        # print("Size of testing dataset is: ", self.X_test.shape)
        if len(predictions) == 0:
            predictions = clf.predict(self.X_test)
            error_clf = np.abs(predictions - self.y_test[:])
            # rmse_ = sqrt(mean_squared_error((clf.predict(self.X_test)), self.y_test[:]))

            # get precision, recall, f1-score, macro-average and micro-average as well as accuracy using the classification report function
            #   TP, True Positive
            #   TN, True Negative
            #   FP, False Positive
            #   FN, False Negative
            #   precision               = TP/(TP+FP)
            #   recall                  = TP/(TP+FN)             [also called sensitivity, hit rate or True positive Rate (TPR)]
            #   True Negative Rate TNR  = TN/(TN+FP)             [also called specificity, selectivity]
            #   f1-score                = 2*TP/(2*TP + FP + FN)  [harmonic mean of precision and recall]
            #   accuracy                = (TP+TN)/(TP+TN+FP+FN)
            #   balanced accuracy       = (TPR+TNR)/2
            #   macro average           = sum( precision per class ) / number_of_classes
            #   micro average           = sum( TP per class ) / sum( (TP+FP) per class )
            precision, recall, fscore, _ = precision_recall_fscore_support(self.y_test[:], predictions, average='macro')
            precision_, recall_, fscore_, _ = precision_recall_fscore_support(self.y_test[:], predictions,
                                                                              average='micro')
            accuracy = accuracy_score(self.y_test[:], predictions)

            report = classification_report(self.y_test[:], predictions)
            # print(report)
        else:
            error_clf = np.abs(predictions - self.y_test[:])
            precision, recall, fscore, _ = precision_recall_fscore_support(self.y_test[:], predictions, average='macro')
            precision_, recall_, fscore_, _ = precision_recall_fscore_support(self.y_test[:], predictions,
                                                                              average='micro')
            accuracy = accuracy_score(self.y_test[:], predictions)

            report = classification_report(self.y_test[:], predictions)
            # print(report)
            # rmse_ = sqrt(mean_squared_error(predictions, self.y_test[:]))
            # print("Root mean squared error of classification is:", rmse_, " %.")

        print('-' * 80)
        print("CLASSIFICATION METRICS")
        print('-' * 80)
        print('     Accuracy  : {:10.2f}'.format(accuracy))
        print('MACRO')
        print('     Precision : {:10.2f}'.format(precision))
        print('     Recall    : {:10.2f}'.format(recall))
        print('     F-score   : {:10.2f}'.format(fscore))
        print('MICRO')
        print('     Precision : {:10.2f}'.format(precision_))
        print('     Recall    : {:10.2f}'.format(recall_))
        print('     F-score   : {:10.2f}'.format(fscore_))

        self.classification_metrics['accuracy'] = accuracy
        self.classification_metrics['macro']['precision'] = precision
        self.classification_metrics['macro']['recall'] = recall
        self.classification_metrics['macro']['f-score'] = fscore
        self.classification_metrics['micro']['precision'] = precision_
        self.classification_metrics['micro']['recall'] = recall_
        self.classification_metrics['micro']['f-score'] = fscore_

        if plot:
            print("Error is shown as a function of real classes for all %s testing cycles." % self.y_test[:].shape)
            plt.figure(figsize=(20, 10))
            plt.stem(self.y_test[:], error_clf, markerfmt=' ')
            plt.xlabel("Class labels for predicted data", size='20')
            plt.ylabel("class diff [%]", size='20')
            plt.title('Prediction error - real and predicted class difference', size='22')
            plt.show()
        return report

    # __________________________________________________________________________________________________________________
    def classifiers_numerical_comparison(self):
        """ compares standards classifiers scores """

        print("---------------------------------------------------------------------------------")
        print("Start classifiers numerical comparison")
        print("---------------------------------------------------------------------------------")
        print("Size of training dataset is: ", self.X_train.shape)
        print("Size of testing dataset is: ", self.X_test.shape)
        names = ["Nearest Neighbors",
                 "Linear SVM",
                 # "RBF SVM",
                 # "Gaussian Process",
                 "Decision Tree",
                 "Random Forest",
                 # "Neural Net",
                 "AdaBoost",
                 "Naive Bayes",
                 "LDA",
                 "QDA"]

        classifiers = [KNeighborsClassifier(3),
                       SVC(kernel="linear", C=0.025, gamma='scale'),
                       SVC(kernel="poly", C=0.025, gamma='scale'),
                       SVC(kernel="rbf", C=0.025, gamma='scale'),
                       SVC(kernel="sigmoid", C=0.025, gamma='scale'),
                       # GaussianProcessClassifier(1.0 * RBF(1.0)),
                       DecisionTreeClassifier(max_depth=5),
                       RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
                       # MLPClassifier(alpha=1, max_iter=1000),
                       AdaBoostClassifier(n_estimators=50, learning_rate=1),
                       GaussianNB(),
                       LinearDiscriminantAnalysis(n_components=3, priors=None, shrinkage=None, solver='eigen'),
                       QuadraticDiscriminantAnalysis()]
        for name, clf in zip(names, classifiers):
            clf.fit(self.X_train, self.y_train)
            score = clf.score(self.X_test, self.y_test)
            print("score %s = %.2f %%" % (name, score * 100))

    # __________________________________________________________________________________________________________________
    def normalize_data(self):
        """ apply normalization to the input features used for classification """

        x_scale = StandardScaler()

        x_scale.fit(np.concatenate((self.X_train, self.X_test)))
        self.X_train = x_scale.transform(self.X_train)
        self.X_test = x_scale.transform(self.X_test)

        self.y_train = self.y_train/100
        self.y_test = self.y_test/100

        # self.X_train = x_scale.fit_transform(self.X_train)
        # self.X_test = x_scale.fit_transform(self.X_test)

        # y_scale = MinMaxScaler(feature_range=(0, 1))
        # self.y_train = y_scale.fit_transform(self.y_train)
        # self.y_test = y_scale.fit_transform(self.y_test)

        self.X_scaler = x_scale

    # __________________________________________________________________________________________________________________
    def store_doe_outputs_to_dictionary(self, doe_output_, offset_level, noise_level):
        """ stores the Design of Experiments results to a single dictionary """
        doe_output_['offset_level'].append(offset_level)
        doe_output_['noise_level'].append(noise_level)
        doe_output_['nb_selected_features'].append(self.nb_selected_features_per_sensor)
        doe_output_['accuracy'].append(self.classification_metrics['accuracy'])
        doe_output_['macro']['precision'].append(self.classification_metrics['macro']['precision'])
        doe_output_['macro']['recall'].append(self.classification_metrics['macro']['recall'])
        doe_output_['macro']['f-score'].append(self.classification_metrics['macro']['f-score'])
        doe_output_['micro']['precision'].append(self.classification_metrics['micro']['precision'])
        doe_output_['micro']['recall'].append(self.classification_metrics['micro']['recall'])
        doe_output_['micro']['f-score'].append(self.classification_metrics['micro']['f-score'])
        return doe_output_

    # __________________________________________________________________________________________________________________
    def replace_sensor_data_with_simulated_data(self, data, y_gen, sensor_number, cycle_range):

        for cycle_number in range(cycle_range[0], cycle_range[1]):
            current_cycle = cycle_number - cycle_range[0]
            data.sensor_[sensor_number][cycle_number] = y_gen[current_cycle, :]

        # data.sensor_[sensor_number][cycle_range[1] + 1:] = 0
        # data.nb_cycles_ = nb_cycles

        return data.sensor_

    # __________________________________________________________________________________________________________________
    def full_process(self,
                     data,
                     feature_extraction_param,
                     feature_selection_param,
                     data_processing_param,
                     dimensionality_reduction_param,
                     classification_param,
                     bayesian_fusion_param,
                     deployment_param):

        """ runs TEST 1 process (extraction, selection and classification) """

        self.create_target_dict(data)

        print("SPLIT TRAIN/TEST DATA")
        self.split_dataset(data,
                           percentage_train=data_processing_param['percentage_training'])

        # _________________________________________________ TRAIN ______________________________________________________

        print("FEATURE EXTRACTION (TRAIN): method %s" % feature_extraction_param['method'])

        freq_of_sorted_values, \
        sorted_values_train_, \
        n_of_samples = self.feature_extractor(method=feature_extraction_param['method'],
                                              frequencies=[])

        print("FEATURE SELECTION (TRAIN): method %s " % feature_selection_param['method'])

        sensor_n, \
        feature_n = self.feature_selection(data,
                                           sorted_values_train_,
                                           n_of_samples,
                                           n_of_features=feature_selection_param['nb_features_to_keep'],
                                           percentage_features=feature_extraction_param['percentage_of_features'],
                                           method=feature_selection_param['method'])

        features_train_, \
        transform_matrix_train_ = self.arrange_features(sorted_values_train_,
                                                        data,
                                                        sensor_n,
                                                        feature_n)

        # ________________________________________________ TEST_________________________________________________________
        print("FEATURE EXTRACTION (TEST): method %s" % feature_extraction_param['method'])
        _, \
        sorted_values_test, \
        _ = self.feature_extractor(method=feature_extraction_param['method'],
                                   frequencies=freq_of_sorted_values)

        features_test_, \
        transform_matrix_test_ = self.arrange_features(sorted_values_test,
                                                       data,
                                                       sensor_n,
                                                       feature_n,
                                                       flag='test')

        self.create_classification_dataset(feature_matrix=features_train_,
                                           feature_matrix_test=features_test_)

        if data_processing_param['data_normalization']:
            self.normalize_data()

        if dimensionality_reduction_param['active']:

            if not (dimensionality_reduction_param['method'] is None):

                self.dimensionality_reduction(method=dimensionality_reduction_param['method'],
                                              plot=dimensionality_reduction_param['plot'],
                                              n_comp=dimensionality_reduction_param['n_components'])

                if deployment_param['active']:
                    bayesian_model  = BayesianFeatureSelectionModel()
                    post_pred, model_RVM, trace_RVM = bayesian_model.deploy(model_fpath=os.path.join(os.getcwd(), deployment_param['model_to_load']),
                                                                            X_test=self.X_test,
                                                                            y_test=self.y_test,
                                                                            transform_matrix_test=transform_matrix_test_)
                    # show predictions
                    bayesian_model.plot_model_vs_data(post_pred['y_obs'], self.y_test)
                else:
                    print("BAYESIAN FEATURE SELECTION MODEL is set to Relevance Vector Machine (RVM). It will keep the same settings.")
                    bayesian_model = BayesianFeatureSelectionModel()
                    bayesian_model.set_params(bayesian_fusion_param)
                    bayesian_model.method = 'RVM'
                    bayesian_model.display_model_settings()

                    model_RVM, \
                    X_shared, \
                    y_shared = bayesian_model.RVM_model(X=self.X_train,
                                                        y=self.y_train,
                                                        n_sensors=data.nb_sensors_)

                    trace_RVM = bayesian_model.inference(model_RVM)

                    print(bayesian_model.resume_estimates(trace_RVM, model_RVM))

                    bayesian_model.save_model(model_fpath=os.path.join(os.getcwd(), 'current_model_RVM.pkl'),
                                              model=model_RVM,
                                              trace=trace_RVM,
                                              X_shared=X_shared,
                                              y_shared=y_shared,
                                              transform_matrix_shared=[]
                                              )

                    # save trace
                    bayesian_model.save(trace_RVM, "no_grouping")

                    # ridge plots
                    bayesian_model.plot(trace_RVM,
                                        plot_type='ridge',
                                        var_name='w_')
                    bayesian_model.plot(trace_RVM,
                                        plot_type='ridge',
                                        var_name='lambda1_')

                    # use model to do predictions
                    post_pred = bayesian_model.RVM_predict(model_RVM,
                                                           trace_RVM,
                                                           X_shared, y_shared,
                                                           self.X_test,
                                                           self.y_test)
                    # show predictions
                    bayesian_model.plot_model_vs_data(post_pred['y_obs'], self.y_test)

        else:

            if bayesian_fusion_param['method'] == 'RGS':

                if deployment_param['active']:
                    bayesian_model  = BayesianFeatureSelectionModel()
                    post_pred, model_RGS, trace_RGS = bayesian_model.deploy(model_fpath=os.path.join(os.getcwd(), deployment_param['model_to_load']),
                                                                            X_test=self.X_test,
                                                                            y_test=self.y_test,
                                                                            transform_matrix_test=transform_matrix_test_)

                    # convergence checking
                    bayesian_model.convergence_graphical_checking(trace_RGS,
                                                                  var_name='lambda2_')

                    # show predictions
                    bayesian_model.plot_model_vs_data(post_pred['y_obs'],
                                                      self.y_test)
                else:
                    bayesian_model = BayesianFeatureSelectionModel()
                    bayesian_model.set_params(bayesian_fusion_param)
                    bayesian_model.display_model_settings()

                    model_RGS, \
                    X_shared, \
                    y_shared, \
                    transform_matrix_shared = bayesian_model.RGS_model(X=self.X_train,
                                                                       y=self.y_train,
                                                                       n_sensors=data.nb_sensors_,
                                                                       transform_matrix=transform_matrix_train_)

                    trace_RGS = bayesian_model.inference(model_RGS)

                    print(bayesian_model.resume_estimates(trace_RGS, model_RGS))

                    bayesian_model.save_model(model_fpath=os.path.join(os.getcwd(), 'current_model_RGS_' + datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)") + '.pkl'),
                                              model=model_RGS,
                                              trace=trace_RGS,
                                              X_shared=X_shared,
                                              y_shared=y_shared,
                                              transform_matrix_shared=transform_matrix_shared
                                              )
                    # save trace
                    bayesian_model.save(trace_RGS, "no_grouping")

                    # ridge plots
                    bayesian_model.plot(trace_RGS,
                                        plot_type='ridge',
                                        var_name='w_')
                    bayesian_model.plot(trace_RGS,
                                        plot_type='ridge',
                                        var_name='lambda1_')
                    bayesian_model.plot(trace_RGS,
                                        plot_type='ridge',
                                        var_name='lambda2_')

                    # use model to do predictions
                    post_pred = bayesian_model.RGS_predict(model_RGS,
                                                           trace_RGS,
                                                           X_shared,
                                                           y_shared,
                                                           transform_matrix_shared,
                                                           self.X_test,
                                                           self.y_test,
                                                           transform_matrix_test_)
                    # show predictions
                    bayesian_model.plot_model_vs_data(post_pred['y_obs'], self.y_test)

            else:

                if deployment_param['active']:
                    bayesian_model  = BayesianFeatureSelectionModel()
                    post_pred, model_RVM, trace_RVM = bayesian_model.deploy(model_fpath=os.path.join(os.getcwd(), deployment_param['model_to_load']),
                                                                            X_test=self.X_test,
                                                                            y_test=self.y_test,
                                                                            transform_matrix_test=transform_matrix_test_)
                    # show predictions
                    bayesian_model.plot_model_vs_data(post_pred['y_obs'], self.y_test)
                else:
                    bayesian_model = BayesianFeatureSelectionModel()
                    bayesian_model.set_params(bayesian_fusion_param)
                    bayesian_model.display_model_settings()

                    model_RVM, \
                    X_shared, \
                    y_shared = bayesian_model.RVM_model(X=self.X_train,
                                                        y=self.y_train,
                                                        n_sensors=data.nb_sensors_)

                    trace_RVM = bayesian_model.inference(model_RVM)

                    print(bayesian_model.resume_estimates(trace_RVM, model_RVM))

                    bayesian_model.save_model(model_fpath=os.path.join(os.getcwd(), 'current_model_RVM_' + datetime.datetime.now().strftime("%d-%b-%Y (%H:%M:%S.%f)") + '.pkl'),
                                              model=model_RVM,
                                              trace=trace_RVM,
                                              X_shared=X_shared,
                                              y_shared=y_shared,
                                              transform_matrix_shared=[]
                                              )

                    # save trace
                    bayesian_model.save(trace_RVM, "no_grouping")
                    # ridge plots
                    bayesian_model.plot(trace_RVM,
                                        plot_type='ridge',
                                        var_name='w_')
                    bayesian_model.plot(trace_RVM,
                                        plot_type='ridge',
                                        var_name='lambda1_')

                    # use model to do predictions
                    post_pred = bayesian_model.RVM_predict(model_RVM,
                                                           trace_RVM,
                                                           X_shared, y_shared,
                                                           self.X_test,
                                                           self.y_test)
                    # show predictions
                    bayesian_model.plot_model_vs_data(post_pred['y_obs'], self.y_test)

        self.get_regression_scores([], predictions=post_pred['y_obs'].mean(axis=0))

        # clf, \
        # predictions = self.classification(method=classification_param['method'],
        #                                   plot=classification_param['plot'])

        # report_classification = self.get_classification_scores(clf)

        # if classification_param['classifier_comparison']:
        #     print("CLASSIFICATION USING %s" % classification_param['method'])
        #     self.classifiers_numerical_comparison()
        #
        # return report_classification

    # __________________________________________________________________________________________________________________
    @staticmethod
    def preprocess_measured_data_for_simulation(data, sensor_number, cycle_range, plot=True):
        """ center-reduces the measure data for one sensor and keep only part of the measurement points and cycles """

        # time vector
        X_original = np.array(data.time_)
        y_original = np.array(data.sensor_[sensor_number][:]).T
        y_mean = np.mean(y_original, axis=0)
        # computes std curve (2000 values)
        # y_std = np.std(y_original, axis=0)
        # center data
        # y_center = y_original - y_mean[np.newaxis, :]
        # sample a number of cycles
        # sample_cycles = np.sort(random.sample(range(y_center.shape[0]),
        #                                             nb_sampled_cycles),
        #                                             axis=None)
        # sample a number of time points
        # sample_measurements = np.sort(random.sample(range(y_center.shape[1]),
        #                                                   nb_sampled_measurement_time),
        #                                                   axis=None)

        # keep only part of measurement points and cycles
        # y_subset = y_center[cycle_range[0]:cycle_range[1], :]
        y_subset = y_original[cycle_range[0]:cycle_range[1], :]
        y_mean_on_subset = y_mean
        X_subset = X_original
        X_subset = X_subset.reshape(-1, 1)

        # print('Shape of y_original: ',          np.shape(y_original))
        # print('Shape of y_mean: ',              np.shape(y_mean))
        # print('Shape of y_center: ',            np.shape(y_center))
        # print('Shape of X_subset: ',            np.shape(X_subset))
        # print('Shape of y_subset: ',            np.shape(y_subset))
        # print('Shape of y_mean_on_subset: ',    np.shape(y_mean_on_subset))

        if plot:
            plt.figure()
            plt.plot(X_original, y_original.T, 'b-', lw=1)
            plt.plot(X_original, y_mean, 'k-', lw=2)
            plt.show()

            # plt.figure()
            # plt.plot(X_original, y_center.T, 'b-', lw=1)
            # plt.show()

            plt.figure()
            plt.plot(X_subset, y_subset.T, 'b--', lw=1)  # y_subset[<cycles>, <time>]
            plt.plot(X_subset, y_mean_on_subset, 'r*', lw=2)
            plt.show()

        return X_original, y_original, X_subset, y_subset, y_mean_on_subset

    # __________________________________________________________________________________________________________________
    @staticmethod
    def estimate_covariance_function(y, method='Empirical', plot=True):
        """ estimates the empirical covariance function from data """

        # estimates covariance function using the subset
        if method == 'Empirical':
            cov_ = EmpiricalCovariance().fit(y)
        elif method == 'Graphical_Lasso':
            cov_ = GraphicalLassoCV(cv=5)
            cov_.fit(y)
        else:
            print("method is not implemented yet")
        emp_cov_ = cov_.covariance_

        if plot:
            vmax = emp_cov_.max()
            plt.figure()
            plt.imshow(emp_cov_, interpolation='nearest', vmin=-vmax, vmax=vmax, cmap=plt.cm.RdBu_r)
            plt.colorbar()
            plt.xticks(())
            plt.yticks(())
            plt.title('Covariance estimate using %s method' % method)

        return emp_cov_

    # __________________________________________________________________________________________________________________
    @staticmethod
    def generate_new_data_samples(emp_mean, emp_cov, nb_samples, X_subset, X_original, y_original, plot=True):
        """
        draws samples from a Mutlivariate Gaussian distribution given the mean and covariance function (estimates)
        """

        prng = np.random.RandomState(1)
        y_gen = prng.multivariate_normal(emp_mean, emp_cov, size=nb_samples)

        if plot:
            plt.figure()
            plt.plot(X_original, y_original[0:100:, :].T, 'k-', lw=1)  # 1 sample from the original dataset
            plt.plot(X_subset, y_gen.T, '-', lw=0.5)  # <nb_samples> Multivariate Gaussian generated
            plt.show()

        return y_gen

    # __________________________________________________________________________________________________________________
    @staticmethod
    def add_offset(y_gen, percentage=0.05):
        """ add offset to the simulated measurement """

        y_gen -= percentage * np.abs(y_gen)
        return y_gen

    # __________________________________________________________________________________________________________________
    @staticmethod
    def add_white_noise(y_gen, percent_relative_noise=0.05, plot=True):
        """ add white noise to the simulated measurement """

        # Calculate signal power and convert to dB 
        for sample_ in range(np.shape(y_gen)[0]):
            # add gaussian noise with standard deviation specified as percentage of the mean function
            noise_sigma = percent_relative_noise * y_gen[sample_, :];
            noise_ = noise_sigma * np.random.standard_normal(len(y_gen[sample_, :]));
            y_gen[sample_, :] += noise_

        if plot:
            plt.figure()
            plt.plot(y_gen[0, :])
            plt.show()

        return y_gen

    # __________________________________________________________________________________________________________________
    @staticmethod
    def save_to_pickle(output_, output_filename):
        """ saves current object to pickle file """
        with open(output_filename, 'wb') as handle:
            pickle.dump(output_, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # __________________________________________________________________________________________________________________
    @staticmethod
    def load_pickle(filename):
        """ loads from .pkl file """
        with open(filename, 'rb') as handle:
            out_ = pickle.load(handle)
        return out_

    # __________________________________________________________________________________________________________________
    @staticmethod
    def arrange_doe_outputs_to_dataframe(results, noise_range, offset_range, nb_sensors):
        """ stores the Design of Experiments results to a dataframe """

        count = 0
        sensor_cols_names = ["#features_sensor " + str(i) for i in range(0, nb_sensors)]
        columns_headers = [["relative_noise"], ["relative_bias"], ["accuracy"], sensor_cols_names]
        flatten = lambda l: [item for sublist in l for item in sublist]
        columns_ = flatten(columns_headers)
        nb_offset_levels = len(offset_range)
        nb_noise_levels = len(noise_range)
        pd_doe = pd.DataFrame(data=[], columns=columns_)
        for noise_level in range(nb_noise_levels):
            for offset_level in range(nb_offset_levels):
                features_number = [results['nb_selected_features'][count][0][i] for i in range(0, nb_sensors)]
                tmp_ = [[noise_range[noise_level]], [offset_range[offset_level]], [results['accuracy'][count]],
                        features_number]
                pd_doe.loc[count] = flatten(tmp_)
                count += 1

        return pd_doe

    # __________________________________________________________________________________________________________________
    @staticmethod
    def plot_doe_results(pd_doe, noise_range, offset_range, nb_points_grid=100, interp_method='cubic'):
        """
        plots the Design of Experiments results as 3D surface and 2D contours that represent
        the accuracy in % as a function of the relative noise and bias introduced in the simulate data
        """

        x, y, z = pd_doe['relative_noise'].values, pd_doe['relative_bias'].values, pd_doe['accuracy'].values
        points = np.transpose(np.asarray([x, y]))
        values = np.asarray(pd_doe['accuracy'].values)

        x_grid = np.linspace(offset_range[0], offset_range[-1], nb_points_grid)
        y_grid = np.linspace(noise_range[0], noise_range[-1], nb_points_grid)

        X, Y = np.meshgrid(x_grid, y_grid)
        Z = griddata(points, values, (X, Y), method=interp_method)

        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1, projection='3d')
        # 3D plot
        ax.plot_surface(X * 100, Y * 100, Z * 100, rstride=1, cstride=1, cmap=plt.cm.RdYlGn, edgecolor='none')
        ax.scatter(x * 100, y * 100, z * 100, marker='o', c='r')
        ax.set_title('Effect of noise and bias on the classification accuracy');
        ax.set_xlabel('relative noise (% of the mean)')
        ax.set_ylabel('relative bias (% of the mean)')
        ax.set_zlabel('classification accuracy %')
        ax = fig.add_subplot(1, 2, 2)
        # 2D contour
        cs = ax.contourf(X * 100, Y * 100, Z * 100, cmap=plt.cm.RdYlGn)
        ax.contour(cs, colors='k')
        ax.scatter(x * 100, y * 100, z * 100, marker='o', c='r')
        ax.set_xlabel('relative noise (% of the mean)')
        ax.set_ylabel('relative bias (% of the mean)')
        cbar = fig.colorbar(cs)
        cbar.ax.set_ylabel('classification accuracy %')
        plt.show()

    # __________________________________________________________________________________________________________________
    @staticmethod
    def largest_indices(array, n):
        """Returns the n largest indices from a numpy array."""

        flat = array.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, array.shape)

    # __________________________________________________________________________________________________________________
    @staticmethod
    def balanced_largest_indices_per_sensor(array, n):
        """ returns the n largest indices for a given dimension of the array"""

        # get matrix of size <nb_sensors>*<n>
        nb_sensors = np.shape(array)[0]
        feature_n = np.reshape(np.argsort(array)[:, :n], (n*nb_sensors))
        # sensor_n = np.reshape((np.reshape(np.tile(np.arange(0, n), nb_sensors), (nb_sensors, n)).T), (n * nb_sensors))
        sensor_n = np.reshape((np.reshape(np.tile(np.arange(0, nb_sensors), n), (n, nb_sensors)).T), (n * nb_sensors))

        return sensor_n, feature_n


# **********************************************************************************************************************
class BayesianFeatureSelectionModel:
    """ method implemented according to the article below
    # Subrahmanya, N., Shin, Y. C., & Meckl, P. H. (2010).
    # A Bayesian machine learning method for sensor selection and fusion with application to on-board fault diagnostics.
    # Mechanical Systems and Signal Processing, 24(1), 182–192. https://doi.org/10.1016/j.ymssp.2009.06.010
    """
    # class properties
    method = 'RGS'
    a1_ = a2_ = b_ = 1e-2
    c_ = d_ = 1e-2
    init = 'jitter+adapt_diag'
    nb_samples_posterior = 50000
    nb_samples_prediction = 10000
    chains = 4
    target_accept = 0.95
    tune = 5000
    thin = 1
    burn = 5000
    cores = 1

    # __________________________________________________________________________________________________________________
    def __init__(self):
        pass

    # __________________________________________________________________________________________________________________
    def set_params(self, input_param):
        """ fill model parameters with input dictionary"""

        self.method = input_param['method']
        self.init = input_param['sampler_init_method']
        self.chains = input_param['chains']
        self.nb_samples_posterior = input_param['nb_samples_posterior']
        self.target_accept = input_param['target_accept']
        self.tune = input_param['tune']
        self.nb_samples_prediction = input_param['nb_samples_prediction']
        self.burn = input_param['burn']
        self.thin = input_param['thin']
        self.cores = input_param['cores']

    # __________________________________________________________________________________________________________________
    def display_model_settings(self):
        """ shows to the user the model parameters in use for the current instance of the model"""

        print('-' * 80)
        print('MODEL PARAMETERS')
        print('-' * 80)
        print('Model parameters / method: {}, sampling initialization algorithm: {}, number of chains: {}, number of draws: {}, tuning iterations: {}, number of burning samples: {}, thining param: {}, number of samples to predict: {}'.format(self.method,
                                                                                                                                                                                                                                                  self.init,
                                                                                                                                                                                                                                                  self.chains,
                                                                                                                                                                                                                                                  self.nb_samples_posterior,
                                                                                                                                                                                                                                                  self.tune,
                                                                                                                                                                                                                                                  self.burn,
                                                                                                                                                                                                                                                  self.thin,
                                                                                                                                                                                                                                                  self.nb_samples_prediction))

    # __________________________________________________________________________________________________________________
    def RVM_model(self, X, y, n_sensors):
        """ Relevance vector machine PyMC3 implementation"""
        print("RVM MODEL DEFINITION: starting...")
        nb_observations = np.shape(X)[0]
        d = np.shape(X)[1]
        m = n_sensors
        # d_per_sensor = n_features_per_sensor

        X_shared = shared(X)
        y_shared = shared(y)

        with pm.Model() as model:
            # Inverse Gamma priors for lambda1_ and tau_
            lambda1_ = pm.Gamma('lambda1_', self.a1_, self.b_, shape=(d,))
            # lambda1_ = pm.HalfCauchy('lambda1_', beta=10, shape=(d,))
            tau_ = pm.Gamma('tau_', self.c_, self.d_)

        with model:
            w_ = pm.Normal('w_', mu=0, tau=lambda1_, shape=(d,))
            mu_ = tt.dot(tt.transpose(w_), X_shared.T)
            pm.Normal('y_obs', mu=mu_, tau=tau_, observed=y_shared)

        print("RVM MODEL DEFINITION: done!")
        return model, X_shared, y_shared

    # __________________________________________________________________________________________________________________
    def RGS_model(self, X, y, n_sensors, transform_matrix, n_features_per_sensor=[]):
        """ Relevance Group Selection algorithm - PyMC3 based implementation """

        print("RGS MODEL DEFINITION: starting...")
        nb_observations = np.shape(X)[0]
        d = np.shape(X)[1]
        m = n_sensors
        # d_per_sensor = n_features_per_sensor

        X_shared = shared(X)
        y_shared = shared(y)
        transform_matrix_shared = []

        with pm.Model() as model:

            # Declare conjugate prior to control the model parameters variance
            lambda1_ = pm.Gamma('lambda1_', self.a1_, self.b_, shape=(d,))
            lambda2_ = pm.Gamma('lambda2_', self.a2_, self.b_, shape=(m,))

            # Declare conjugate prior for the inverse noise variance (Inverse Gamma)
            tau_ = pm.Gamma('tau_', self.c_, self.d_)
            intercept = pm.Normal('Intercept', 0, sigma=20)

        transform_matrix_shared = shared(transform_matrix)
        with model:
            tau_w = lambda1_ + tt.dot(tt.transpose(lambda2_), transform_matrix_shared)
            w_ = pm.Normal('w_', mu=0, tau=tau_w, shape=(d,))
            mu_ = tt.dot(tt.transpose(w_), X_shared.T) + intercept
            pm.Normal('y_obs', mu=mu_, tau=tau_, observed=y_shared)

        print("RGS MODEL DEFINITION: done!")
        return model, X_shared, y_shared, transform_matrix_shared

    # __________________________________________________________________________________________________________________
    def inference(self, model):
        """ infers the model parameters based on NUTS sampler """

        print("BAYESIAN INFERENCE: starting...")
        with model:
                trace = pm.sample(draws=self.nb_samples_posterior,
                                  init=self.init,
                                  chains=self.chains,
                                  tune=self.tune,
                                  target_accept=self.target_accept,
                                  cores=self.cores,# TODO multiprocessing issues (theano problem)
                                  progressbar=True)

        print("BAYESIAN INFERENCE: done!")
        return trace

    # __________________________________________________________________________________________________________________
    def RGS_predict(self, model, train_trace, X_shared, y_shared, transform_matrix_shared, X_test, y_test,
                    transform_matrix_test):

        X_shared.set_value(X_test)
        y_shared.set_value(y_test)  # dummy values
        transform_matrix_shared.set_value(transform_matrix_test)

        samples_ = np.max([self.nb_samples_prediction, 10000])
        with model:
            ppc = pm.sample_posterior_predictive(train_trace[self.burn::self.thin], samples=samples_, model=model)

        return ppc

    # __________________________________________________________________________________________________________________
    def RVM_predict(self, model, train_trace, X_shared, y_shared, X_test, y_test):

        X_shared.set_value(X_test)
        y_shared.set_value(y_test)

        with model:
            ppc = pm.sample_posterior_predictive(train_trace[self.burn::self.thin], samples=self.nb_samples_prediction)

        return ppc

    # __________________________________________________________________________________________________________________
    def deploy(self, model_fpath, X_test, y_test, transform_matrix_test):
        """ uses predicted weight to predict new samples"""

        model, \
            trace, \
            X_shared, \
            y_shared, \
            transform_matrix_shared = \
            self.load_model(model_fpath=model_fpath)

        print(self.resume_estimates(trace, model))
        # use model to do predictions
        post_pred = self.RGS_predict(model,
                                     trace,
                                     X_shared,
                                     y_shared,
                                     transform_matrix_shared,
                                     X_test,
                                     y_test,
                                     transform_matrix_test)

        return post_pred, model, trace

    # __________________________________________________________________________________________________________________
    def plot(self, trace, plot_type='forest', var_name='lambda1_'):
        """ plotting function (forest plot and standard traces of the MCMC chains) """

        if plot_type == 'forest':
            az.plot_forest(trace[self.burn::self.thin],
                           kind='forestplot',
                           var_names=[var_name],
                           combined=True,
                           ridgeplot_overlap=3,
                           figsize=(6, 14))
        elif plot_type == 'ridge':
            az.plot_forest(trace[self.burn::self.thin],
                           kind='ridgeplot',
                           var_names=[var_name],
                           combined=True,
                           ridgeplot_overlap=3,
                           figsize=(6, 14))
        else:
            az.plot_trace(trace[self.burn::self.thin],
                          var_names=[var_name],
                          compact=True)

        plt.show()

    # __________________________________________________________________________________________________________________
    @staticmethod
    def plot_model_vs_data(y_hat, y, sorting=True, max_test_samples=100, plot_type='boxplot'):
        """ plots the model versus the observed data """

        if plot_type == 'boxplot':
            if sorting:
                ind_                = np.argsort(y)
                y_sorted            = y[ind_]
                y_hat_sorted        = y_hat[ind_]
            else:
                y_sorted            = y
                y_hat_sorted        = y_hat

            plt.boxplot(y_hat_sorted[:, :max_test_samples])
            plt.plot(np.arange(1, max_test_samples+1), y_sorted[:max_test_samples], 'ro')
            plt.show()
        else:
            "get mean estimate"
            y_hat_mean              = y_hat.mean(axis=0)
            y_hat_std               = y_hat.std(axis=0)
            y_hat_mean              = y_hat_mean[:max_test_samples]
            y_hat_std               = y_hat_std[:max_test_samples]
            y                       = y[:max_test_samples]
            if sorting:
                ind_                    = np.argsort(y)
                y_sorted                = y[ind_]
                y_hat_mean_sorted       = y_hat_mean[ind_]
                y_hat_std_sorted        = y_hat_std[ind_]
            else:
                y_sorted = y
                y_hat_mean_sorted = y_hat_mean
                y_hat_std_sorted = y_hat_std
            x                       = np.arange(0, len(y_hat_mean))
            plt.errorbar(x, y_hat_mean_sorted, xerr=0., yerr=y_hat_std_sorted, fmt='o')
            plt.plot(y_sorted,      'k*')
            plt.show()

    # __________________________________________________________________________________________________________________
    @staticmethod
    def convergence_graphical_checking(trace, var_name='lambda2_'):
        """ plots the estimate for the mean of log(var_name) cumulating mean """

        val_var_name = trace[var_name][:, 0]
        mean_log_var_name = [np.mean(val_var_name[:i]) for i in np.arange(1, len(val_var_name))]
        plt.figure(figsize=(15, 4))
        plt.plot(mean_log_var_name, lw=2.5)
        plt.xlabel('Iteration')
        plt.ylabel('MCMC mean of ({:})'.format(var_name))
        plt.title('MCMC estimation of ({:})'.format(var_name))
        plt.show()

    # __________________________________________________________________________________________________________________
    @staticmethod
    def save(trace, filename):
        """ saves inference results to .pkl file """
        df_summary = pm.summary(trace)
        output_name = "./estimations_" + filename + ".pkl"
        df_summary.to_pickle(output_name)

    # __________________________________________________________________________________________________________________
    @staticmethod
    def show_graph(model):
        """ shows bayesian digraph """
        pm.model_to_graphviz(model)
        plt.show()

    # __________________________________________________________________________________________________________________
    def save_model(self, model_fpath, model, trace, X_shared, y_shared, transform_matrix_shared=[]):
        """ saves the current trained model. It can then be used later for predictions"""

        with open(model_fpath, 'wb') as buff:
            pickle.dump(
                {
                'method':                   self.method,
                'init':                     self.init,
                'chains':                   self.chains,
                'nb_samples_posterior':     self.nb_samples_posterior,
                'target_accept':            self.target_accept,
                'tune':                     self.tune,
                'nb_samples_prediction':    self.nb_samples_prediction,
                'burn':                     self.burn,
                'thin':                     self.thin,
                'model':                    model,
                'trace':                    trace,
                'X_shared':                 X_shared,
                'y_shared':                 y_shared,
                'transform_matrix_shared':  transform_matrix_shared
            },  buff
            )

    # __________________________________________________________________________________________________________________
    def load_model(self, model_fpath):
        with open(model_fpath, 'rb') as buff:
            data = pickle.load(buff)

        self.method = data['method']
        self.init = data['init']
        self.chains = data['chains']
        self.nb_samples_posterior = data['nb_samples_posterior']
        self.target_accept = data['target_accept']
        self.tune = data['tune']
        self.nb_samples_prediction = data['nb_samples_prediction']
        self.burn = data['burn']
        self.thin = data['thin']

        model = data['model']
        trace = data['trace']
        X_shared = data['X_shared']
        y_shared = data['y_shared']
        transform_matrix_shared = data['transform_matrix_shared']

        print('I AM LOADING AN EXISTING MODEL')
        self.display_model_settings()

        return model, trace, X_shared, y_shared, transform_matrix_shared

    # __________________________________________________________________________________________________________________
    def resume_estimates(self, trace, model, save_to_csv=True):
        """ summary statistics for model parameters estimations (after training) """

        print('-' * 80)
        print('Parameter Estimates [Statistical Summary]')
        df = pm.summary(trace[self.burn::self.thin], var_names=model.vars)
        if save_to_csv:
            df.to_csv(os.path.join(os.getcwd(), "BayesianTrainingParamEstimatesSummary.csv"))
        return df
