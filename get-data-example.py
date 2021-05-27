import argparse
import time
import numpy as np

import brainflow
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations


def main():
    BoardShim.enable_dev_board_logger()

    params = BrainFlowInputParams()
    # with headset
    params.serial_port = "/dev/cu.usbserial-DM03GT61"
    

    # with headset
    board_id = 0

    # without headset
    #board_id = BoardIds.SYNTHETIC_BOARD.value

    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    time.sleep(10)
    data = board.get_board_data()  # get all data and remove it from internal buffer
    board.stop_stream()
    board.release_session()

    print(data)


if __name__ == "__main__":
    main()