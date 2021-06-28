import sys
import time
import numpy as np
import keyboard
from collections import Counter
import pandas as pd
import pickle

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams
from brainflow.exit_codes import *

import collections
import serial
import random
import json

# run this script 

# arduino serial port global
# note check port on arduino software - it changes depending on connector dongle being used
# with simple usb connector it is '/dev/cu.usbmodem14201'
arduino = serial.Serial(port='/dev/cu.usbmodem142101', baudrate=115200, timeout=.1)

def main():
    board_choice = input("Press 1 for Synthetic or 2 for Cyton")

    if board_choice == "1":
        board_id = BoardIds.SYNTHETIC_BOARD.value
    if board_choice == "2":
        board_id = BoardIds.CYTON_BOARD.value

    # BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    # MLModel.enable_ml_logger()

    if board_id == 0:
        params.serial_port = "/dev/cu.usbserial-DM03GT61"
        # params.serial_port = "COM4"

    board = BoardShim(board_id, params)
    sampling_rate = BoardShim.get_sampling_rate(board_id)
    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)

    emoSVM = pickle.load(open('EmoSVM_Bands.sav', 'rb'))

#### PRE-PROCESSING ########################

    def filter_signal(_data, _eeg_channels):
        for channel in _eeg_channels:
            #5hz - 59hz bandpass
            DataFilter.perform_bandpass(_data[channel], BoardShim.get_sampling_rate(board_id), 37.5, 75, 4, FilterTypes.BESSEL.value, 0)
            # 50hz filter
            DataFilter.perform_bandstop(_data[channel], BoardShim.get_sampling_rate(board_id), 50, 1.0, 3, FilterTypes.BUTTERWORTH.value, 0)
            #Denoise
            DataFilter.perform_wavelet_denoising(_data[channel], 'coif3', 3)
        return _data

    def detrend_signal(_data, _eeg_channels):
        for channel in _eeg_channels:
            DataFilter.detrend(_data[channel], DetrendOperations.LINEAR.value)
        return _data

    def calculate_psd(_data, _eeg_channels):
        for channel in _eeg_channels:
            DataFilter.get_psd_welch(_data[channel], nfft, nfft // 2, sampling_rate, WindowFunctions.BLACKMAN_HARRIS.value)
        return _data

    def get_bands(_data, _eeg_channels):
        return DataFilter.get_avg_band_powers(_data, _eeg_channels, sampling_rate, True)

#### CLASSIFICATIONS ######################

    def get_concentration(_bands):
        feature_vector = np.concatenate((_bands[0], _bands[1]))
        concentration_params = BrainFlowModelParams(BrainFlowMetrics.CONCENTRATION.value, BrainFlowClassifiers.KNN.value)
        concentration = MLModel(concentration_params)
        concentration.prepare()
        conc = concentration.predict(feature_vector)
        concentration.release()
        return conc

    def get_relaxation(_bands):
        feature_vector = np.concatenate((_bands[0], _bands[1]))
        relaxation_params = BrainFlowModelParams(BrainFlowMetrics.RELAXATION.value, BrainFlowClassifiers.REGRESSION.value)
        relaxation = MLModel(relaxation_params)
        relaxation.prepare()
        relax = relaxation.predict(feature_vector)
        relaxation.release()
        return relax

    def get_valance_TS(_data, _eeg_channels, _emoSVM):                                                                                                                                     
        _df = pd.DataFrame()
        _df = pd.DataFrame(np.transpose(_data))
        _df = _df[_eeg_channels]

        _values = _emoSVM.predict(_df)
        _value_counts = Counter(_values)
        pos_count = _value_counts['Positive']
        neg_count = _value_counts['Negative']

        if pos_count > neg_count:
            _valance = 'Positive'
        else:
            _valance = 'Negative'
        return _valance

    def get_valance_bands_avg(_bands, _emoSVM):                                                                                                                                     
        _avg_bands_df = pd.DataFrame()
        _avg_bands_df = pd.DataFrame(_bands)
        
        _valance = _emoSVM.predict(_avg_bands_df)

        return _valance

    def get_valance_bands(_data, _emoSVM):

        row_of_bands = []

        fp1_bands = DataFilter.get_avg_band_powers(_data, [0], sampling_rate, True)
        fp2_bands = DataFilter.get_avg_band_powers(_data, [1], sampling_rate, True)
        f3_bands = DataFilter.get_avg_band_powers(_data, [2], sampling_rate, True)
        f4_bands = DataFilter.get_avg_band_powers(_data, [3], sampling_rate, True)
        f7_bands = DataFilter.get_avg_band_powers(_data, [4], sampling_rate, True)
        f8_bands = DataFilter.get_avg_band_powers(_data, [5], sampling_rate, True)
        t7_bands = DataFilter.get_avg_band_powers(_data, [6], sampling_rate, True)
        t8_bands = DataFilter.get_avg_band_powers(_data, [7], sampling_rate, True)

        row_of_bands = [fp1_bands[0], fp2_bands[0], f3_bands[0], f4_bands[0], f7_bands[0], f8_bands[0], t7_bands[0], t8_bands[0]]
        row_of_bands = np.concatenate([row_of_bands], axis=None)
        bands_array = np.array([row_of_bands])

        print(bands_array)

        _valance = _emoSVM.predict(bands_array)

        return _valance

    def get_emotion(_valance, _relax):
        _emotion = 'null'
        if _relax < 0.5:
            _arousal = 'High'
        if _relax > 0.5:
            _arousal = 'Low'
        
        if _arousal == 'High' and _valance == 'Positive':
            _emotion = "Happy"
        if _arousal == 'High' and _valance == 'Negative':
            _emotion = "Stessed"
        if _arousal == 'Low' and _valance == 'Positive':
            _emotion = 'Relaxed'
        if _arousal == 'Low' and _valance == 'Negative':
            _emotion = 'Sad'
        return _emotion

    def get_emotions_mode(_emotions_list):
        # calculate the frequency of each item
        _emotions_data = collections.Counter(_emotions_list)
        _emotions_data_frequency = dict(_emotions_data)

        # Print the items with frequency
        print(_emotions_list)
        print(_emotions_data_frequency)

        # Find the highest frequency
        max_value = max(list(_emotions_data.values()))
        mode_val = [num for num, freq in _emotions_data_frequency.items() if freq == max_value]
        print(mode_val)
        if len(mode_val) == len(_emotions_list):
            # print("No mode in the list")
            # favour positive - if 2 positives
            return "null"
        elif len(mode_val) > 1 and "Happy" in mode_val:
            return "Happy"
        elif len(mode_val) > 1 and "Stressed" in mode_val:
            return "Stressed"
        elif len(mode_val) > 1:
            return mode_val[0]
        else:
            # print("The Mode of the list is : " + ', '.join(map(str, mode_val)))
            return ', '.join(map(str, mode_val))

#### OUTPUT ######################

    def send_EMS(_emotion):
        if _emotion == "Happy":
            print("emotion : Happy")
            arduino.write(bytes("1", 'utf-8'))
            time.sleep(0.05)
        
        elif _emotion == "Sad":
            print("emotion : Sad")
            arduino.write(bytes("2", 'utf-8'))
            time.sleep(0.05)
        
        elif _emotion == "Stressed":
            print("emotion : Stressed")
            arduino.write(bytes("3", 'utf-8'))
            time.sleep(0.05)

        elif _emotion == "Relaxed":
            print("emotion : Relaxed")
            arduino.write(bytes("0", 'utf-8'))
            time.sleep(0.05)
        
        elif _emotion == "null":
            print("emotion : null")
            arduino.write(bytes("0", 'utf-8'))
            time.sleep(0.05)

    def write_JSON(_emotions_list):
        _emotions_dict = {"Happy": 0, "Sad": 0, "Stressed": 0, "Relaxed":0}
        # calculate the frequency of each item
        _emotions_data = collections.Counter(_emotions_list)
        _emotions_data_frequency = dict(_emotions_data)
        # update dict
        for key in _emotions_data_frequency:
            _emotions_dict[key] = _emotions_data_frequency[key]
        # write to file
        with open('emotion-output.json', 'w') as fp:
            json.dump(_emotions_dict, fp)

##########################################        

    def stop_stream():
        board.stop_stream()
        board.release_session()

    def start_stream():
        board.prepare_session()
        board.start_stream()
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'starting stream')
        eeg_channels = BoardShim.get_eeg_channels(board_id)

        _emotion_aggregate = []

        while True:
            time.sleep(5)
            data = board.get_board_data()
            pdData = pd.DataFrame(np.transpose(data))

            data = filter_signal(data, eeg_channels)
            # data = detrend_signal(data, eeg_channels)
            psd = calculate_psd(data, eeg_channels)
            bands = get_bands(data, eeg_channels)

            conc = get_concentration(bands)
            relax = get_relaxation(bands)
            # avg_valance = get_valance_bands_avg(np.array(bands[0]).reshape(1,-1), emoSVM)
            valance = get_valance_bands(data, emoSVM)
            emotion = get_emotion(valance, relax)
            print("relaxation: " + str(relax))
            print("valance: " + str(valance))
            print("arousal: " + str(1 - relax))
            #_ran_emotion = ["Happy", "Sad", "Stressed", "Relaxed"]
            emotion = "Sad" #random.choice(_ran_emotion)
            print(emotion)
            # add to _emotion_aggregate if < 6 items
            if len(_emotion_aggregate) < 5:
                _emotion_aggregate.append(emotion)
            else:
                # get mode & send to EMS
                #_emotion_mode = get_emotions_mode(["Stressed", "Sad", "Relaxed", "Relaxed", "Sad"])
                _emotion_mode = get_emotions_mode(_emotion_aggregate)
                send_EMS(_emotion_mode)
                write_JSON(_emotion_aggregate)
                # reset list
                _emotion_aggregate = []

    start_stream()

if __name__ == "__main__":
    main()
