from __future__ import division
import re
import numpy as np
from numpy.lib import unique
import pandas as pd
import math
from scipy.io import loadmat
from numpy.lib.stride_tricks import as_strided as ast
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
from tensorflow.python.ops.gen_array_ops import Reshape, reshape, shape
from tensorflow.python.ops.gen_math_ops import bucketize, mod, xdivy_eager_fallback
from tensorflow.python.ops.nn_ops import pool

# Paper this script is following: https://www.mdpi.com/1424-8220/20/7/2034

# Avoid GPU Errors #####################################################
########################################################################

physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

tf.executing_eagerly()

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

# Function to reshape data in shape suitable for neural net input
def make_input_tensor(data, nchannels: int,NNC: int, dims: int):
    samples = len(data)
    array = np.zeros(shape=(samples, (nchannels + nchannels*NNC)), dtype=int)
    tensor = np.zeros(shape=(samples, (nchannels + nchannels*NNC), dims))
    for dim in range(dims):
        array = np.zeros(shape=(samples, (nchannels + nchannels*NNC)), dtype=int)
        for channel in range(nchannels):
            #put the OG channel vector in the first column
            channelvector = data[:,channel]
            array[:,channel*(NNC + 1)] = channelvector
            for noisychannel in range(NNC):
                #making a vector with length "samples", by sampling from the normal/gaussian distribution with a mean of 0, standard deviation of 0.01
                noisyvector = np.random.normal(0,0.01*1000000,samples)
                #add channelvector and the noisyvector and pop it into our output array
                array[:, channel*(NNC + 1) + noisychannel+1] = (channelvector + noisyvector)
        tensor[:,:,dim] = array
    return tensor

# This function condenses labels into two classes 
# The label coding is as follows 
# 1,6-Sad
# 2-Neutral
# 3,5-Happy
# 4-Baseline

def code_labels(labels_df):
    
    # Positive Emotions
    labels_df = labels_df.replace(to_replace = 3, value= 8)    
    labels_df = labels_df.replace(to_replace = 5, value= 8)
    
    # Negative Emotions
    labels_df = labels_df.replace(to_replace = 1, value= 9)    
    labels_df = labels_df.replace(to_replace = 6, value= 9)
    return labels_df

###########################################################################
###########################################################################

# SCRIPT ##################################################################
###########################################################################

# Load the File
mat = loadmat("DataSets/LUMED/EEG_GSR_Data/s01.mat")

# Extact the data, ingoring matlab meta data
struct = mat["EEGData"]

# Deconstruct the matlab struct into its constituents 
data = struct[0,0]

# Create a dataframe of the EEG+GSR data and then delete the GSR Data
EEG_data_df = pd.DataFrame(data['Data'])
del EEG_data_df[8]

# Create a dataframe of the data Lables
labels_df = pd.DataFrame(data['Labels'].astype(np.int32))


# Initialise a new Dataframe with the EEG data to begin, and then append labels as the final columnn
df = EEG_data_df
df[8] = labels_df
df[8] = code_labels(df[8])
df = df[df[8] > 7]

# cat_df = df[8]

# cat_df = cat_df.astype("category")

# print(cat_df)



# Turn the dataset back into a numpy array called DataSet
DataSet = df.to_numpy()

# Window the Dataset in chunks of 300 samples with 50 sample overlaps (4.1 in article)
Chunkyset = chunk_data(DataSet, 300, overlap_size=50, flatten_inside_window=False) #############################################

ChunkyLabels = Chunkyset[:,:,-1]

label_test = ChunkyLabels

ChunkyLabels = ChunkyLabels[:,:,np.newaxis]
ChunkyLabels3D = np.append(ChunkyLabels, ChunkyLabels, axis = 2)
ChunkyLabels3D = np.append(ChunkyLabels3D, ChunkyLabels, axis = 2)
ChunkyLabels4D = ChunkyLabels3D[:,:,np.newaxis, :]
ChunkyLabels3DLong = np.reshape(ChunkyLabels3D,(850*300,1,3))
ChunkyLabelsSlice = ChunkyLabels3D[:,0,0]

# Initialise variable for reshaping data (4.2 in article)
dims = 3
nchannels = 8
N1 = 80
NNC = math.ceil(N1/nchannels)-1
windowlen = len(Chunkyset[0])
Chunkytensors = np.zeros(shape=(len(Chunkyset),windowlen,N1,dims))
NormChunkyTensors = np.zeros(shape=(len(Chunkyset),windowlen,N1,dims))

# Feed our data into initialised tensor shape which will be called "Chunkytensors" (4.2 in article)
# This tensor has the guassian noise agumented data, and this function will combine our data set with the augmented data tensor
for window in range(len(Chunkyset)):
    data = Chunkyset[window, :, :]
    Chunkytensors[window,:,:,:] = make_input_tensor(data,nchannels,NNC,dims) # noise augmentation to make data correct shape for keras model
    for dim in range(dims): # normalisation 
        avg = np.mean(Chunkytensors[window, : , : ,dim], axis=0)
        NormChunkyTensors[window, : , : ,dim] = Chunkytensors[window, : , : ,dim] - avg

#NormChunkyTensorsLong = np.reshape(NormChunkyTensors, (850*300,80,3))


x_data = tf.convert_to_tensor(NormChunkyTensors)

y_labels = tf.constant(ChunkyLabelsSlice)


###########################################################################################################################

x2 = tf.convert_to_tensor(np.random.random((850, 300, 80, 3)))

y2 = tf.constant(np.random.random((850, 1)))

#########################################################################################################################


base_model = keras.applications.InceptionResNetV2(
    input_shape=(300,80,3),
    include_top=False,
    )
base_model.trainable = False

# inputs = keras.Input(shape=(300,80,3))
# x = base_model(inputs, training = False)
# x = layers.GlobalAveragePooling2D()(x)
# # x = layers.Dense(1024, activation='relu'),
# # x = layers.Dense(1024, activation='relu'),
# # x = layers.Dense(1024, activation='relu'),
# # x = layers.Dense(512, activation='relu'),
# outputs = keras.layers.Dense(1, activation= 'softmax')(x)

# model = keras.Model(inputs, outputs)

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1024, activation='relu'),
    layers.Dense(1024, activation='relu'),
    layers.Dense(1024, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='softmax'),
    
])


model.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

model.summary()

# model.fit(x_data, y_labels, batch_size=64, shuffle = True, epochs= 100)
model.fit(x2, y2, batch_size=64, shuffle = True, epochs= 100)

