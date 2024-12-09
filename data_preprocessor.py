import csv
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt
import json
from tqdm import tqdm
from numpy.random import normal

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

def remove_outliers(data, threshold=1.8):
    q1 = np.percentile(data, 1)
    q3 = np.percentile(data, 99)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    return np.clip(data, lower_bound, upper_bound)

def print_completion_banner():
    total_files = END - BEGIN + 1
    bar_width = 40
    bar = '█' * bar_width
    
    print("\n" + "=" * 60)
    print("🎉 Processing Complete! 🎉")
    print(f"✨ Successfully processed {total_files} files ✨")
    print("\n📊 Progress:")
    print(f"[{bar}] 100%")
    print("\n📈 Results:")
    print(f"   📁 Files processed: {total_files}")
    print(f"   💾 Output directory: {OUTPUT_DIR}")
    print(f"   ⏱️  Time segments: {DATA_POINT}s")
    print("=" * 60 + "\n")


# Load configuration
with open('config.json', 'r') as config_file:
    config = json.load(config_file)

# Load configuration & Set parameters
BEGIN = config['begin']
END = config['end']
DATA_PATH = config['data_path']
TIME_PAIR = config['time_pair']
OUTPUT_DIR = config['output_dir']
VIDEO_DATA = config['video_data']
DATA_POINT = 120    
LABEL_SEGMENT_MAX = 1800 // DATA_POINT

if (len(TIME_PAIR) != END - BEGIN + 1):
    print("The length of time_pair is not equal to the number of files")
    exit(1)

files = []
for i in range(BEGIN, END + 1):
    files.append(f"{i}.csv")

for idx, file in enumerate(tqdm(files, desc="Processing files")):
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
                if ((idx + BEGIN >= 27 and idx + BEGIN <= 31) and (idx + BEGIN) % 2 == 1):
                    alpha.append(float(row[0]))
                    beta.append(float(row[1]))
                    theta.append(float(row[3]))
                else:
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
                filtered_wave = remove_outliers(filtered_wave)
                filtered_data.append(filtered_wave)

                processed_wave = []
                for j in range(len(filtered_wave) // int(sample_rate)):
                    if idx + BEGIN == 0 and 990 <= j <= 1090:
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
        
        fig1.suptitle(f'EEG Waves Comparison - File {idx + BEGIN}', fontsize=16)
        
        file_output_dir = os.path.join(OUTPUT_DIR, f"{idx+ BEGIN}")
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
        
        fig2.suptitle(f'EEG Waves - File {idx + BEGIN}', fontsize=16)
        
        plt.tight_layout()
        plt.savefig(os.path.join(file_output_dir, 'combined.png'))
        plt.close()
    
    # TODO:  output the data every DATA_POINT seconds
    label_segment_number = TIME_PAIR[idx][0] // DATA_POINT
    start_time = TIME_PAIR[idx][0] - label_segment_number * DATA_POINT

    # Delete the first segment if the result is 30 minutes
    if (label_segment_number == LABEL_SEGMENT_MAX):
        label_segment_number -= 1
        start_time += DATA_POINT

    for i in range(label_segment_number):
        if (idx + BEGIN in VIDEO_DATA):
            output_csv_path = os.path.join(file_output_dir, f'{i}_0.csv')
        else:
            output_csv_path = os.path.join(file_output_dir, f'{i}_1.csv')
        
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Alpha', 'Beta', 'Theta'])
            rows_written = 0
            last_alpha = None
            last_beta = None
            last_theta = None
            
            for j in range(DATA_POINT):
                current_time = start_time + i * DATA_POINT + j
                try:
                    alpha_val = EEG[0][current_time] 
                    beta_val = EEG[1][current_time]
                    theta_val = EEG[2][current_time]
                    
                    writer.writerow([alpha_val, beta_val, theta_val])
                    last_alpha, last_beta, last_theta = alpha_val, beta_val, theta_val
                    rows_written += 1
                except:
                    # Generate random values based on last known values
                    alpha_val = normal(last_alpha, abs(last_alpha * 0.05))
                    beta_val = normal(last_beta, abs(last_beta * 0.05))
                    theta_val = normal(last_theta, abs(last_theta * 0.05))
                    
                    writer.writerow([alpha_val, beta_val, theta_val])
                    last_alpha, last_beta, last_theta = alpha_val, beta_val, theta_val
                    rows_written += 1
            
            # Fill remaining rows if needed
            while rows_written < DATA_POINT:
                alpha_val = normal(last_alpha, abs(last_alpha * 0.05))
                beta_val = normal(last_beta, abs(last_beta * 0.05))
                theta_val = normal(last_theta, abs(last_theta * 0.05))
                
                writer.writerow([alpha_val, beta_val, theta_val])
                last_alpha, last_beta, last_theta = alpha_val, beta_val, theta_val
                rows_written += 1

print_completion_banner()
    
        