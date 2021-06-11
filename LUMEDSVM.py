import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided as ast
from pandas.core.frame import DataFrame
from sklearn import svm
from scipy.io import loadmat
import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams
from brainflow.exit_codes import *

########################################################################
########################################################################


# FUNCTIONS ############################################################
########################################################################

# Windowing function 
# Taken from : https://gist.github.com/mattjj/5213172
def chunk_data(data,window_size,overlap_size=0,flatten_inside_window=True):
    assert data.ndim == 1 or data.ndim == 2
    if data.ndim == 1:
        data = data.reshape((-1,1))

    # get the number of overlapping windows that fit into the data
    num_windows = (data.shape[0] - window_size) // (window_size - overlap_size) + 1
    overhang = data.shape[0] - (num_windows*window_size - (num_windows-1)*overlap_size)

    # if there's overhang, need an extra window and a zero pad on the data
    # (numpy 1.7 has a nice pad function I'm not using here)
    if overhang != 0:
        num_windows += 1
        newdata = np.zeros((num_windows*window_size - (num_windows-1)*overlap_size,data.shape[1]))
        newdata[:data.shape[0]] = data
        data = newdata

    sz = data.dtype.itemsize
    ret = ast(
            data,
            shape=(num_windows,window_size*data.shape[1]),
            strides=((window_size-overlap_size)*data.shape[1]*sz,sz)
            )

    if flatten_inside_window:
        return ret
    else:
        return ret.reshape((num_windows,-1,data.shape[1]))

# This function condenses labels into two classes 
# The label coding is as follows 
# 1,6-Sad
# 2-Neutral
# 3,5-Happy
# 4-Baseline

def code_labels(coded_df):
    
    # Positive Emotions
    coded_df = coded_df.replace(to_replace = 3, value= 8)    
    coded_df = coded_df.replace(to_replace = 5, value= 8)
    
    # Negative Emotions
    coded_df = coded_df.replace(to_replace = 1, value= 9)    
    coded_df = coded_df.replace(to_replace = 6, value= 9)

    return coded_df

def stringify_classes(df_to_stringify):

    df_to_stringify = df_to_stringify.replace(to_replace = 8, value = "Positive")
    df_to_stringify = df_to_stringify.replace(to_replace = 9, value = "Negative")

    return df_to_stringify

###########################################################################
###########################################################################

# SCRIPT ##################################################################
###########################################################################

# Load the Files
print("Loading data ...")

mat1 = loadmat("DataSets/LUMED/EEG_GSR_Data/s01.mat")
mat2 = loadmat("DataSets/LUMED/EEG_GSR_Data/s02.mat")
mat3 = loadmat("DataSets/LUMED/EEG_GSR_Data/s03.mat")
mat4 = loadmat("DataSets/LUMED/EEG_GSR_Data/s04.mat")
mat5 = loadmat("DataSets/LUMED/EEG_GSR_Data/s05.mat")
mat6 = loadmat("DataSets/LUMED/EEG_GSR_Data/s06.mat")
mat7 = loadmat("DataSets/LUMED/EEG_GSR_Data/s07.mat")
mat8 = loadmat("DataSets/LUMED/EEG_GSR_Data/s08.mat")
mat9 = loadmat("DataSets/LUMED/EEG_GSR_Data/s09.mat")
mat10 = loadmat("DataSets/LUMED/EEG_GSR_Data/s10.mat")
mat11 = loadmat("DataSets/LUMED/EEG_GSR_Data/s11.mat")
mat12 = loadmat("DataSets/LUMED/EEG_GSR_Data/s12.mat")
mat13 = loadmat("DataSets/LUMED/EEG_GSR_Data/s13.mat")

# Extact the data, ingoring matlab meta data
print("Extracting Values")

struct1 = mat1["EEGData"]
struct2 = mat2["EEGData"]
struct3 = mat3["EEGData"]
struct4 = mat4["EEGData"]
struct5 = mat5["EEGData"]
struct6 = mat6["EEGData"]
struct7 = mat7["EEGData"]
struct8 = mat8["EEGData"]
struct9 = mat9["EEGData"]
struct10 = mat10["EEGData"]
struct11 = mat11["EEGData"]
struct12 = mat12["EEGData"]
struct13 = mat13["EEGData"]

# Deconstruct the matlab struct into its constituents 
print("Deconstructing structs")

data1 = struct1[0,0]
data2 = struct2[0,0]
data3 = struct3[0,0]
data4 = struct4[0,0]
data5 = struct5[0,0]
data6 = struct6[0,0]
data7 = struct7[0,0]
data8 = struct8[0,0]
data9 = struct9[0,0]
data10 = struct10[0,0]
data11 = struct11[0,0]
data12 = struct12[0,0]
data13 = struct13[0,0]

# Convert them all into dataframes
print("Constructing EEG dataframes")

EEG_data_df1 = pd.DataFrame(data1['Data'])
EEG_data_df2 = pd.DataFrame(data2['Data'])
EEG_data_df3 = pd.DataFrame(data3['Data'])
EEG_data_df4 = pd.DataFrame(data4['Data'])
EEG_data_df5 = pd.DataFrame(data5['Data'])
EEG_data_df6 = pd.DataFrame(data6['Data'])
EEG_data_df7 = pd.DataFrame(data7['Data'])
EEG_data_df8 = pd.DataFrame(data8['Data'])
EEG_data_df9 = pd.DataFrame(data9['Data'])
EEG_data_df10 = pd.DataFrame(data10['Data'])
EEG_data_df11 = pd.DataFrame(data11['Data'])
EEG_data_df12 = pd.DataFrame(data12['Data'])
EEG_data_df13 = pd.DataFrame(data13['Data'])

# Convert them all into dataframes
print("Constructing label dataframes")

Label_data_df1 = pd.DataFrame(data1['Labels'].astype(np.int32))
Label_data_df2 = pd.DataFrame(data2['Labels'].astype(np.int32))
Label_data_df3 = pd.DataFrame(data3['Labels'].astype(np.int32))
Label_data_df4 = pd.DataFrame(data4['Labels'].astype(np.int32))
Label_data_df5 = pd.DataFrame(data5['Labels'].astype(np.int32))
Label_data_df6 = pd.DataFrame(data6['Labels'].astype(np.int32))
Label_data_df7 = pd.DataFrame(data7['Labels'].astype(np.int32))
Label_data_df8 = pd.DataFrame(data8['Labels'].astype(np.int32))
Label_data_df9 = pd.DataFrame(data9['Labels'].astype(np.int32))
Label_data_df10 = pd.DataFrame(data10['Labels'].astype(np.int32))
Label_data_df11 = pd.DataFrame(data11['Labels'].astype(np.int32))
Label_data_df12 = pd.DataFrame(data12['Labels'].astype(np.int32))
Label_data_df13 = pd.DataFrame(data13['Labels'].astype(np.int32))

# Downsample EEG data to 250hz
print("Downsampling EEG data")

EEG_data_df1 = EEG_data_df1.iloc[::2,:]
EEG_data_df2 = EEG_data_df2.iloc[::2,:]
EEG_data_df3 = EEG_data_df3.iloc[::2,:]
EEG_data_df4 = EEG_data_df4.iloc[::2,:]
EEG_data_df5 = EEG_data_df5.iloc[::2,:]
EEG_data_df6 = EEG_data_df6.iloc[::2,:]
EEG_data_df7 = EEG_data_df7.iloc[::2,:]
EEG_data_df8 = EEG_data_df8.iloc[::2,:]
EEG_data_df9 = EEG_data_df9.iloc[::2,:]
EEG_data_df10 = EEG_data_df10.iloc[::2,:]
EEG_data_df11 = EEG_data_df11.iloc[::2,:]
EEG_data_df12 = EEG_data_df12.iloc[::2,:]
EEG_data_df13 = EEG_data_df13.iloc[::2,:]

# Downsample Labels to 250hz
print ("Downsampling labels")

Label_data_df1 = Label_data_df1.iloc[::2,:]
Label_data_df2 = Label_data_df2.iloc[::2,:]
Label_data_df3 = Label_data_df3.iloc[::2,:]
Label_data_df4 = Label_data_df4.iloc[::2,:]
Label_data_df5 = Label_data_df5.iloc[::2,:]
Label_data_df6 = Label_data_df6.iloc[::2,:]
Label_data_df7 = Label_data_df7.iloc[::2,:]
Label_data_df8 = Label_data_df8.iloc[::2,:]
Label_data_df9 = Label_data_df9.iloc[::2,:]
Label_data_df10 = Label_data_df10.iloc[::2,:]
Label_data_df11 = Label_data_df11.iloc[::2,:]
Label_data_df12 = Label_data_df12.iloc[::2,:]
Label_data_df13 = Label_data_df13.iloc[::2,:]

# Group the EEG dataframes together and downsample to 250hz
print("Grouping EEG dataframes")

EEG_frames = [
    EEG_data_df1,
    EEG_data_df2,
    EEG_data_df3,
    EEG_data_df4,
    EEG_data_df5,
    EEG_data_df6,
    EEG_data_df7,
    EEG_data_df8,
    EEG_data_df9,
    EEG_data_df10,
    EEG_data_df11,
    EEG_data_df12,
    EEG_data_df13
]

# Group the label dataframes together and downsample to 250hz
print("Grouping label dataframes")

Label_frames = [
    Label_data_df1,
    Label_data_df2,
    Label_data_df3,
    Label_data_df4,
    Label_data_df5,
    Label_data_df6,
    Label_data_df7,
    Label_data_df8,
    Label_data_df9,
    Label_data_df10,
    Label_data_df11,
    Label_data_df12,
    Label_data_df13
]

# Concatenate EEG dataframes into a single EEG dataframe and then delete the GSR data
print("Conatenating EEG dataframes")

EEG_df = pd.concat(EEG_frames)
del EEG_df[8]



# Concatenate training EEG dataframes into a single EEG dataframe and recode the values
print("Conatenating label dataframes")

Label_df = pd.concat(Label_frames)
Label_df = code_labels(Label_df)
# Label_df = stringify_classes(Label_df)

# Add labels to a master df
print("Generating Mainframe")

Mainframe = EEG_df
Mainframe[8] = Label_df
Mainframe = Mainframe[Mainframe[8] != 4]
Mainframe = Mainframe[Mainframe[8] != 0] 
Mainframe = Mainframe[Mainframe[8] != 2]


# #################################################################
# # Initialise some settings for pre-processing the data 
# print("Initialising pre-processing settings")

# board_id = BoardIds.CYTON_BOARD.value

# # BoardShim.enable_dev_board_logger()
# params = BrainFlowInputParams()
# # MLModel.enable_ml_logger()

# board = BoardShim(board_id, params)
# sampling_rate = BoardShim.get_sampling_rate(board_id)
# eeg_channels = [0, 1, 2, 3, 4, 5, 6, 7]
# nfft = DataFilter.get_nearest_power_of_two(sampling_rate)

# #######################################################################
# # Functions for data pre-processing 
# print("Preparing pre-processing functions")

# def calculate_psd(_data, _eeg_channels):
#     for channel in _eeg_channels:
#         DataFilter.get_psd_welch(_data[channel], nfft, nfft // 2, sampling_rate, WindowFunctions.BLACKMAN_HARRIS.value)
#     return _data

# def filter_signal(_data, _eeg_channels):
#     for channel in _eeg_channels:
#         #5hz - 59hz bandpass
#         DataFilter.perform_bandpass(_data[channel], BoardShim.get_sampling_rate(board_id), 26.5, 21.5, 4, FilterTypes.BESSEL.value, 0)
#         # 50hz filter
#         DataFilter.perform_bandstop(_data[channel], BoardShim.get_sampling_rate(board_id), 50, 1.0, 3, FilterTypes.BUTTERWORTH.value, 0)
#         #Anti-wifi
#         DataFilter.perform_bandstop(_data[channel], BoardShim.get_sampling_rate(board_id), 24.25, 1.0, 3, FilterTypes.BUTTERWORTH.value, 0)
#         #Denoise
#         DataFilter.perform_wavelet_denoising(_data[channel], 'coif3', 3)
#     return _data

# def detrend_signal(_data, _eeg_channels):
#     for channel in _eeg_channels:
#         DataFilter.detrend(_data[_eeg_channels], DetrendOperations.LINEAR.value)
#     return _data

# def calculate_psd(_data, _eeg_channels):
#     for channel in _eeg_channels:
#         DataFilter.get_psd_welch(_data[channel], nfft, nfft // 2, sampling_rate, WindowFunctions.BLACKMAN_HARRIS.value)
#     return _data

# def get_bands(_data, _eeg_channels):
#     return DataFilter.get_avg_band_powers(_data, _eeg_channels, sampling_rate, True)

# ###################################################################################################

# arrayframe = Mainframe.to_numpy()
# arrayframe = np.transpose(arrayframe)

# # Initialise data as brainflow-like object 
# eeg_array = arrayframe[eeg_channels]
# eeg_array = eeg_array.astype(float)
# label_array = arrayframe[-1]


# # Data pre-processing
# print("Pre-processing data")

# eeg_array = filter_signal(eeg_array, eeg_channels)

# for count, channel in enumerate(eeg_channels):
#     eeg_for_fft = eeg_array
#     fft_data = DataFilter.perform_fft(eeg_for_fft[channel], WindowFunctions.NO_WINDOW.value)

# print(fft_data)

# psd = calculate_psd(eeg_array, eeg_channels)
# bands = get_bands(eeg_array, eeg_channels)



# print(len(psd))
# print(len(bands))
# print(nfft)


# MACHINE LEARNING STUFF ########################################################
#################################################################################

# Initialise Training Variables
X = Mainframe[[0, 1, 2, 3, 4, 5, 6, 7]]
y = Mainframe[8]

print("X Shape:", X.shape)
print("Y Shape:", y.shape)

# Split data into a traning set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("Fitting data to model... this may take a while")
classifier = svm.LinearSVC(verbose=True).fit(X_train, y_train.values.ravel())

np.set_printoptions(precision=2)

print("Saving classifier to disk")

# s = pickle.dump(classifier, open('EmoSVM_TS_Linear.sav', 'wb'))
s = pickle.dump(classifier, open('EmoSVM_TS.sav', 'wb'))
# s = pickle.dump(classifier, open('EmoSVM_FFT_Linear.sav', 'wb'))
# s = pickle.dump(classifier, open('EmoSVM_FFT.sav', 'wb'))
# s = pickle.dump(classifier, open('EmoSVM_PSD_Linear.sav', 'wb'))
# s = pickle.dump(classifier, open('EmoSVM_PSD.sav', 'wb'))
# s = pickle.dump(classifier, open('EmoSVM_Bands_Linear.sav', 'wb'))
# s = pickle.dump(classifier, open('EmoSVM_Bands.sav', 'wb'))

# Plot non-normalized confusion matrix
print("Plotting data to confusion matrix")

titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, X_test, y_test, cmap=plt.cm.Blues, normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

