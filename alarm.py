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

INPUT_DIM = 3
HIDDEN_DIM = 128
NUM_LAYERS = 2

BUZZER_PIN = 24
LED_PIN = 4

SERIAL_PORT = '/dev/serial0'
BAUD_RATE = 9600

MODEL_PATH = './model_path'

FOLDER_NUM = 31
SET_NUM = 4

EXEPECTED_RES = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        x = x.float()
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        logits = self.fc(last_step)
        return logits

def send_message(message):
    try:
        # Setup new connection
        ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUD_RATE,
            timeout=1,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            bytesize=serial.EIGHTBITS
        )
        time.sleep(0.1)
        
        ser.write(message.encode('utf-8'))
        
        ser.close()
    except Exception as e:
        print(f"❌ Error sending message: {e}")

def blink_alarm(times=2, interval=0.5):
    try:
        for _ in range(times):
            GPIO.output(BUZZER_PIN, GPIO.HIGH)
            GPIO.output(LED_PIN, GPIO.HIGH)
            time.sleep(interval/8)
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
        
        with open(file_path, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip header
            data = []
            for row in csv_reader:
                data.append([float(row[0]), float(row[1]), float(row[2])])
        
        data = np.array(data)

        n_samples = len(data)
        n_windows = n_samples // window_size
        
        data = data[:n_windows * window_size]
        
        reshaped_data = np.zeros((n_windows, window_size, 3))

        for i in range(n_windows):
            reshaped_data[i] = data[i*window_size:(i+1)*window_size]
        
        for i in range(reshaped_data.shape[0]):
            reshaped_data[i] = (reshaped_data[i] - reshaped_data[i].mean()) / reshaped_data[i].std()

        reshaped_data = torch.FloatTensor(reshaped_data)
        
        return reshaped_data
        
    except Exception as e:
        print(f"❌ Error processing data: {e}")
        return None

if __name__ == "__main__":
    try:
        model = LSTMClassifier(INPUT_DIM, HIDDEN_DIM, NUM_LAYERS)
        model.load_state_dict(torch.load(f'{MODEL_PATH}/best_lstm_model_0.8250.pth'), strict=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Setup GPIO pins
        GPIO.setup(BUZZER_PIN, GPIO.OUT)
        GPIO.setup(LED_PIN, GPIO.OUT)
        
        blink_alarm()
        
        res = []
        for i in tqdm(range(FOLDER_NUM), desc="Processing folders"):
            for j in range(SET_NUM):
                data = process_data(f'./testData/{i}/{j}.csv')
                outputs = model(data)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                alarm_count = 0

                for k in range(len(preds)//5):
                    if sum(preds[k*5:(k+1)*5]) >= 3:
                        alarm_count += 1
                    else:
                        alarm_count = 0
                    
                if alarm_count >= 3:
                    send_message("1\n")
                    blink_alarm()
                    alarm_count = 0
                    res.append(1)
                else:
                    res.append(0)

                time.sleep(0.35)
        
        res_array = np.array(res)
        expected_array = np.array(EXEPECTED_RES)

        print(f"result: {res_array}")
        print(f"expected: {expected_array}")

        if len(res_array) != len(expected_array):
            print("❌ Length mismatch between result and expected arrays")
            sys.exit(1)
        
        accuracy = np.sum(res_array == expected_array) / len(res_array)
        print(f'Total accuracy for {len(res_array)} cases: {accuracy:.4f}')
                
    except KeyboardInterrupt:
        print("\n⚠️ Program interrupted by user")
    finally:
        GPIO.cleanup()
        print("🧹 Cleanup complete")
