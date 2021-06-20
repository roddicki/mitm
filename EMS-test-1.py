# Importing Libraries
import serial
import time
arduino = serial.Serial(port='/dev/cu.usbmodem142101', baudrate=115200, timeout=.1)
# note check port on arduino software - it changes depending on connector dongle being used
# with simple usb connector it is '/dev/cu.usbmodem14201'

def write_read(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
    data = arduino.readline()
    return data

while True:
    num = input("0:off, 1:Happy, 2:Sad, 3:Stressed\nEnter a number: ") # Taking input from user
    value = write_read(num)
    print(value) # printing the value