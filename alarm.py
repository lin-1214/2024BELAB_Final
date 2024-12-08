import RPi.GPIO as GPIO
import time
import serial
import sys
from train_w_lstm import LSTMClassifier
import torch
import csv
import numpy as np
import os
from scipy.signal import butter, filtfilt

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Model parameters
INPUT_DIM = 3
HIDDEN_DIM = 64
NUM_LAYERS = 1

# Define GPIO pins
BUZZER_PIN = 24  # GPIO24
LED_PIN = 4      # GPIO4

# HC-05 Serial setup
# SERIAL_PORT = '/dev/ttyS0'
SERIAL_PORT = '/dev/serial0'
BAUD_RATE = 9600

TEST_FILE = '1'

def setup_bluetooth():
    try:
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            timeout=1,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        )

        time.sleep(0.1)
        return ser
    except Exception as e:
        print(f"❌ Error setting up HC-05: {e}")
        return None

def send_message(ser, message):
    try:
        # Convert string message to bytes using encode()
        ser.write(message.encode('utf-8'))  
        print(f"✅ Sent byte value: {message.encode('utf-8')}")
    except Exception as e:
        print(f"❌ Error sending message: {e}")

def blink_alarm(times=3, interval=0.5):
    try:
        for _ in range(times):
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(interval/2)
            GPIO.output(BUZZER_PIN, GPIO.LOW)
            GPIO.output(LED_PIN, GPIO.LOW)
            time.sleep(interval)
            
    except KeyboardInterrupt:
        pass
    
    finally:
        GPIO.output(BUZZER_PIN, GPIO.LOW)
        GPIO.output(LED_PIN, GPIO.LOW)

def process_data(file_path, window_size=120):
    try:
        # Read data from CSV
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header
            data = []
            for row in csv_reader:
                # Assuming columns are in order: alpha, beta, theta
                data.append([float(row[0]), float(row[1]), float(row[2])])
        
        # Convert to numpy array
        data = np.array(data)
        
        # Calculate number of complete windows
        n_samples = len(data)
        n_windows = n_samples // window_size
        
        # Truncate data to fit complete windows
        data = data[:n_windows * window_size]
        
        # Reshape to (n_windows, window_size, 3)
        reshaped_data = data.reshape(n_windows, window_size, 3)
        
        print(f"✅ Data shaped to: {reshaped_data.shape}")
        return reshaped_data
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return None

if __name__ == "__main__":
    bluetooth_serial = None

    try:
        model = LSTMClassifier(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS)
        model.load_state_dict(torch.load('./lstm_model_20241208-214752.pth'))
        model = model.to('cpu')
        
        # Setup GPIO pins
        GPIO.setup(BUZZER_PIN, GPIO.OUT)
        GPIO.setup(LED_PIN, GPIO.OUT)
        
        # Test blink_alarm
        blink_alarm()
        
        # Setup HC-05
        bluetooth_serial = setup_bluetooth()
        if bluetooth_serial is None:
            print("❌ Could not establish Bluetooth connection. Exiting...")
            sys.exit(1)
        
        # Process data
        data = process_data(f'./testData/{TEST_FILE}.csv')

        # Predict
        outputs = model(data)
        preds = (outputs >= 0.5).float()

        alarm_count = 0
        for i in range(len(preds)/5):
            if sum(preds[i*5:(i+1)*5]) >= 3:
                alarm_count += 1
            else:
                alarm_count = 0
            
            if alarm_count >= 3:
                print("Alarm")
                send_message(bluetooth_serial, "1")
                blink_alarm()
                alarm_count = 0
        
        

                
    except KeyboardInterrupt:
        print("\n⚠️ Program interrupted by user")
    finally:
        if bluetooth_serial:
            bluetooth_serial.close()
        GPIO.cleanup()
        print("🧹 Cleanup complete")
