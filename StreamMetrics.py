import time
import numpy as np
import keyboard
import pandas as pd

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, WindowFunctions, DetrendOperations
from brainflow.ml_model import MLModel, BrainFlowMetrics, BrainFlowClassifiers, BrainFlowModelParams
from brainflow.exit_codes import *


def main():
    board_choice = input("Press 1 for Synthetic or 2 for Cyton")

    if board_choice == "1":
        board_id = BoardIds.SYNTHETIC_BOARD.value
    if board_choice == "2":
        board_id = BoardIds.CYTON_BOARD.value

    BoardShim.enable_dev_board_logger()
    params = BrainFlowInputParams()
    MLModel.enable_ml_logger()

    if board_id == 0:
        params.serial_port = "/dev/cu.usbserial-DM01MPXO"
        # params.serial_port = "COM4"

    sampling_rate = BoardShim.get_sampling_rate(board_id)
    board = BoardShim(board_id, params)
    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)

#### PRE-PROCESSING ########################

    def filter_signal(_data, _eeg_channels):
        for channel in _eeg_channels:
            #5hz - 59hz bandpass
            DataFilter.perform_bandpass(_data[channel], BoardShim.get_sampling_rate(board_id), 26.5, 21.5, 4, FilterTypes.BESSEL.value, 0)
            #50hz filter
            DataFilter.perform_bandstop(_data[channel], BoardShim.get_sampling_rate(board_id), 50, 1.0, 3, FilterTypes.BUTTERWORTH.value, 0)
            #Anti-wifi
            DataFilter.perform_bandstop(_data[channel], BoardShim.get_sampling_rate(board_id), 24.25, 1.0, 3, FilterTypes.BUTTERWORTH.value, 0)
            #Denoise
            DataFilter.perform_wavelet_denoising(_data[channel], 'coif3', 3)
        return _data

    def detrend_signal(_data, _eeg_channels):
        for channel in _eeg_channels:
            DataFilter.detrend(_data[_eeg_channels], DetrendOperations.LINEAR.value)
        return _data

    def calculate_psd(_data, _eeg_channels):
        for channel in _eeg_channels:
            DataFilter.get_psd_welch(_data[channel], nfft, nfft // 2, sampling_rate, WindowFunctions.BLACKMAN_HARRIS.value)
            print(_data)
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

##########################################        

    def stop_stream():
        board.stop_stream()
        board.release_session()

    def start_stream():
        board.prepare_session()
        board.start_stream()
        BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'starting stream')
        eeg_channels = BoardShim.get_eeg_channels(board_id)

        while True:
            time.sleep(2)
            data = board.get_board_data()

            psd = calculate_psd(data, eeg_channels)
            bands = get_bands(data, eeg_channels)

            conc = get_concentration(bands)
            relax = get_relaxation(bands)

            print('conentration:', conc, 'relaxation:', relax)

    start_stream()

if __name__ == "__main__":
    main()
