import RPi.GPIO as GPIO
import time
import serial
import sys
import torch
import csv
import numpy as np
import os
from tqdm import tqdm
from scipy.signal import butter, filtfilt
import torch.nn as nn



GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Model parameters
INPUT_DIM = 3
HIDDEN_DIM = 128
NUM_LAYERS = 2

# Define GPIO pins
BUZZER_PIN = 24  # GPIO24
LED_PIN = 4      # GPIO4

# HC-05 Serial setup
# SERIAL_PORT = '/dev/ttyS0'
SERIAL_PORT = '/dev/serial0'
BAUD_RATE = 9600

# Model parameters
MODEL_PATH = './model_path'

# Test data
FOLDER_NUM = 31
SET_NUM = 4
# generated by gen_testData.py
EXEPECTED_RES = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # LSTM output: (batch, seq_len, hidden_dim), (h_n, c_n)
        x = x.float()
        lstm_out, _ = self.lstm(x)  
        # We can take the output of the last time step for classification
        last_step = lstm_out[:, -1, :]  # shape (batch, hidden_dim)
        logits = self.fc(last_step)     # shape (batch, 1)

        return logits

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
        # print(f"✅ Sent byte value: {message.encode('utf-8')}")
    except Exception as e:
        print(f"❌ Error sending message: {e}")

def blink_alarm(times=1, interval=0.5):
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
        
        # Normalize each window independently
        normalized_data = np.zeros_like(reshaped_data)

        for i in range(n_windows):
            window = reshaped_data[i]
            # Normalize each window independently
            window_mean = window.mean(axis=0)
            window_std = window.std(axis=0)
            normalized_data[i] = (window - window_mean) / (window_std + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Convert numpy array to PyTorch tensor
        normalized_data = torch.FloatTensor(normalized_data)
        
        return normalized_data
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return None

if __name__ == "__main__":
    bluetooth_serial = None

    try:
        model = LSTMClassifier(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS)
        model.load_state_dict(torch.load(f'{MODEL_PATH}/best_lstm_model_0.8250.pth'))
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

        res = []
        # Process data
        for i in tqdm(range(FOLDER_NUM), desc="Processing folders"):
            for j in range(SET_NUM):
                data = process_data(f'./testData/{i}/{j}.csv')
                # Predict
                outputs = model(data)
                # print(f"outputs: {outputs}")
                preds = (torch.sigmoid(outputs) >= 0.5).float()

                alarm_count = 0
                for i in range(len(preds)//5):
                    if sum(preds[i*5:(i+1)*5]) >= 3:
                        alarm_count += 1
                    else:
                        alarm_count = 0
                    
                if alarm_count >= 3:
                    # print("Alarm")
                    send_message(bluetooth_serial, "1")
                    blink_alarm()
                    alarm_count = 0
                    res.append(1)
                else:
                    res.append(0)
        
        # Convert lists to numpy arrays for comparison
        res_array = np.array(res)
        expected_array = np.array(EXEPECTED_RES)
        
        # Add debug information
        print(f"Length of res_array: {len(res_array)}")
        print(f"Length of expected_array: {len(expected_array)}")

        print(f"result: {res_array}")
        print(f"expected: {expected_array}")

        if len(res_array) != len(expected_array):
            print("❌ Length mismatch between result and expected arrays")
            sys.exit(1)
        
        # Calculate accuracy
        accuracy = np.sum(res_array == expected_array) / len(res_array)
        print(f'Total accuracy for {len(res_array)} cases: {accuracy:.4f}')
                
    except KeyboardInterrupt:
        print("\n⚠️ Program interrupted by user")
    finally:
        if bluetooth_serial:
            bluetooth_serial.close()
        GPIO.cleanup()
        print("🧹 Cleanup complete")
