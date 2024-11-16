import csv
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt
import json

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

BEGIN = config['begin']
END = config['end']
DATA_PATH = config['data_path']
TIME_PAIR = config['time_pair']
OUTPUT_DIR = config['output_dir']

files = []
for i in range(BEGIN, END + 1):
    files.append(f"{i}.csv")

for idx, file in enumerate(files):
    raw_data = []
    EEG = []
    with open(f'./data/{file}', 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader) 

        alpha = []
        beta = []
        theta = []
        for row in csv_reader:
            try:
                alpha.append(float(row[1]))
                beta.append(float(row[2]))
                theta.append(float(row[4]))
            except:
                continue

        raw_data = [alpha, beta, theta]
        sample_rate = np.round(len(raw_data[0]) / TIME_PAIR[idx][2])
        fs = sample_rate
        filtered_data = []
        processed_waves = []

        freq_bands = {
            'alpha': (8, 13),
            'beta': (13, 30),
            'theta': (4, 8)
        }

        for wave, wave_type in zip(raw_data, ['alpha', 'beta', 'theta']):
            try:
                lowcut, highcut = freq_bands[wave_type]
                filtered_wave = butter_bandpass_filter(wave, lowcut, highcut, fs)
                filtered_data.append(filtered_wave)

                processed_wave = []
                for j in range(len(filtered_wave) // int(sample_rate)):
                    if idx == 0 and 990 <= j <= 1090:
                        continue
                    segment = filtered_wave[j * int(sample_rate):(j + 1) * int(sample_rate)]
                    if segment.size > 0:
                        avg = np.mean(segment)
                        processed_wave.append(avg)
                
                processed_waves.append(processed_wave)
            except (ValueError, ZeroDivisionError) as e:
                print(f"Error processing {wave_type} wave: {e}")
                continue

        EEG = processed_waves

        fig1, axs = plt.subplots(3, 2, figsize=(15, 15))
        wave_types = ['Alpha', 'Beta', 'Theta']
        colors = ['r', 'g', 'y']
        
        for i, (wave_data, wave_type, color) in enumerate(zip(raw_data, wave_types, colors)):
            axs[i][0].plot(wave_data, color=color, label=wave_type)
            axs[i][0].set_title(f'Raw {wave_type} Wave')
            axs[i][0].set_xlabel('Time Points')
            axs[i][0].set_ylabel('Amplitude')
            axs[i][0].legend()
            axs[i][0].grid(True)
        
        for i, (wave_data, wave_type, color) in enumerate(zip(filtered_data, wave_types, colors)):
            axs[i][1].plot(wave_data, color=color, label=wave_type)
            axs[i][1].set_title(f'Filtered {wave_type} Wave')
            axs[i][1].set_xlabel('Time Points')
            axs[i][1].set_ylabel('Amplitude')
            axs[i][1].legend()
            axs[i][1].grid(True)
        
        fig1.suptitle(f'EEG Waves Comparison - File {idx + 1}', fontsize=16)
        
        file_output_dir = os.path.join(OUTPUT_DIR, f"{idx + 1}")
        if not os.path.exists(file_output_dir):
            os.makedirs(file_output_dir)

        plt.tight_layout()
        plt.savefig(os.path.join(file_output_dir, 'comparison.png'))
        plt.close()

        fig2, axs = plt.subplots(3, 1, figsize=(12, 12))
        
        for wave_data, wave_type, color, ax in zip(EEG, wave_types, colors, axs):
            ax.plot(wave_data, color=color, label=wave_type, linewidth=2)
            ax.set_title(f'{wave_type} Wave')
            ax.set_xlabel('Time Points')
            ax.set_ylabel('Amplitude')
            ax.legend()
            ax.grid(True)
        
        fig2.suptitle(f'EEG Waves - File {idx + 1}', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(file_output_dir, 'combined.png'))
        plt.close()
    
    # TODO:  output the data every 1-2 minutes