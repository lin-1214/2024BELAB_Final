import csv
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

output_data_folders = ['31', '33', '34', '35', '36', '37']
CSV_NUM = 14
CSV_REQUIRED = 30
SETS_PER_COUNT = 4
MAX_TARGET_COUNT = 30

test_set = []
for target_count in tqdm(range(MAX_TARGET_COUNT + 1), desc="Generating test sets"):
    for set_num in range(SETS_PER_COUNT):
        while True:
            files_target = []
            files_others = []
            
            for folder in output_data_folders:
                for file_num in range(CSV_NUM):
                    suffix = '0' if folder in ['31', '33', '37'] else '1'
                    file_path = f'./output/{folder}/{file_num}_{suffix}.csv'
                    if folder in ['31', '33', '37']:
                        files_target.append(file_path)
                    else:
                        files_others.append(file_path)
            
            if target_count > len(files_target) or (CSV_REQUIRED - target_count) > len(files_others):
                print(f"Skipping target_count {target_count} due to insufficient files.")
                break
            
            csv_files = random.sample(files_target, target_count) + \
                       random.sample(files_others, CSV_REQUIRED - target_count)
            
            random.shuffle(csv_files)
            
            actual_count = sum(1 for path in csv_files if ('/31/' in path or '/33/' in path or '/37/' in path))
            if actual_count == target_count:
                test_set.append(csv_files)
                break

os.makedirs('./testData', exist_ok=True)

res = []

for idx, file_set in tqdm(enumerate(test_set), desc="Processing files", total=len(test_set)):
    target_count = idx // SETS_PER_COUNT
    set_num = idx % SETS_PER_COUNT
    
    file_suffixes = [file_path.split('_')[-1].split('.')[0] for file_path in file_set]
    
    longest_play_time = 0
    current_play_time = 0
    
    for i in range(0, len(file_suffixes), 5):
        group = file_suffixes[i:i+5]
        ones_count = sum(1 for suffix in group if suffix == '1')
        
        if ones_count >= 3:
            current_play_time += 10
            longest_play_time = max(longest_play_time, current_play_time)
        else:
            current_play_time = 0
    
    res.append(1 if longest_play_time >= 30 else 0)

    print(f"Set {idx} (Target count: {target_count}, Set: {set_num}) - Longest play time: {longest_play_time} minutes")
    
    os.makedirs(f'./testData/{target_count}', exist_ok=True)
    output_file = f'./testData/{target_count}/{set_num}'
    
    combined_data = []
    
    for csv_file in file_set:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            if not combined_data:
                combined_data.append(header)
            for row in reader:
                combined_data.append(row)
    
    with open(f'{output_file}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(combined_data)
    
    data_array = np.array(combined_data[1:], dtype=float)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'EEG Bands (Target Count: {target_count}, Set: {set_num})')
    
    ax1.plot(data_array[:, 0], label='Alpha', color='r', linewidth=2)
    ax1.set_ylabel('Alpha')
    ax1.grid(True)
    ax1.legend()
    
    ax2.plot(data_array[:, 1], label='Beta', color='g', linewidth=2)
    ax2.set_ylabel('Beta')
    ax2.grid(True)
    ax2.legend()
    
    ax3.plot(data_array[:, 2], label='Theta', color='y', linewidth=2)
    ax3.set_ylabel('Theta')
    ax3.set_xlabel('Sample')
    ax3.grid(True)
    ax3.legend()
    
    plt.tight_layout()
    
    plt.savefig(f'{output_file}.png', dpi=300, bbox_inches='tight')
    plt.close()

print(f'Expected result: {res}')