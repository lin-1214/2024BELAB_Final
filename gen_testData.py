import csv
import random
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# output_data_folders = ['31', '33', '34', '35', '36', '37']
output_data_folders = ['27', '28', '34', '35', '36', '37']
CSV_NUM = 14
CSV_REQUIRED = 30
SETS_PER_COUNT = 4  # Generate 4 arrays for each target count
MAX_TARGET_COUNT = 30  # Now we can go up to 30 since we have folder 30 as well

test_set = []
for target_count in tqdm(range(MAX_TARGET_COUNT + 1), desc="Generating test sets"):
    for set_num in range(SETS_PER_COUNT):
        while True:
            # Create pools of possible files
            files_target = []
            files_others = []
            
            # Generate all possible file paths for each folder
            for folder in output_data_folders:
                for file_num in range(CSV_NUM):
                    suffix = '0' if folder in ['27', '28', '37'] else '1'
                    file_path = f'./output/{folder}/{file_num}_{suffix}.csv'
                    if folder in ['27', '28', '37']:
                        files_target.append(file_path)
                    else:
                        files_others.append(file_path)
            
            # Check if we have enough files to sample
            if target_count > len(files_target) or (CSV_REQUIRED - target_count) > len(files_others):
                print(f"Skipping target_count {target_count} due to insufficient files.")
                break
            
            # Randomly select files
            csv_files = random.sample(files_target, target_count) + \
                       random.sample(files_others, CSV_REQUIRED - target_count)
            
            # Shuffle the final list to ensure random order
            random.shuffle(csv_files)
            
            # Verify we have exactly the target number of files from 31/33/37
            actual_count = sum(1 for path in csv_files if ('/27/' in path or '/28/' in path or '/37/' in path))
            if actual_count == target_count:
                test_set.append(csv_files)
                break

# Create output directory if it doesn't exist
os.makedirs('./testData', exist_ok=True)

res = []  # Initialize empty res array

# Process each set of files
for idx, file_set in tqdm(enumerate(test_set), desc="Processing files", total=len(test_set)):
    target_count = idx // SETS_PER_COUNT
    set_num = idx % SETS_PER_COUNT
    
    # Extract suffixes from filenames
    file_suffixes = [file_path.split('_')[-1].split('.')[0] for file_path in file_set]
    
    # Calculate longest consecutive playing time
    longest_play_time = 0
    current_play_time = 0
    
    # Process in groups of 5 files
    for i in range(0, len(file_suffixes), 5):
        group = file_suffixes[i:i+5]
        ones_count = sum(1 for suffix in group if suffix == '1')
        
        if ones_count >= 3:
            current_play_time += 10
            longest_play_time = max(longest_play_time, current_play_time)
        else:
            current_play_time = 0
    
    # Add 1 to res if longest play time is >= 30 minutes
    res.append(1 if longest_play_time >= 30 else 0)

    print(f"Set {idx} (Target count: {target_count}, Set: {set_num}) - Longest play time: {longest_play_time} minutes")
    
    # Create directory if it doesn't exist
    os.makedirs(f'./testData/{target_count}', exist_ok=True)
    output_file = f'./testData/{target_count}/{set_num}'
    
    # Initialize combined data storage
    combined_data = []
    
    # Read and combine all CSV files in the set
    for csv_file in file_set:
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)
            if not combined_data:
                combined_data.append(header)
            for row in reader:
                combined_data.append(row)
    
    # Write combined data to CSV
    with open(f'{output_file}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(combined_data)
    
    # Create plot from the combined data
    data_array = np.array(combined_data[1:], dtype=float)  # Skip header
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'EEG Bands (Target Count: {target_count}, Set: {set_num})')
    
    # Plot Alpha (column 0) - using red
    ax1.plot(data_array[:, 0], label='Alpha', color='r', linewidth=2)
    ax1.set_ylabel('Alpha')
    ax1.grid(True)
    ax1.legend()
    
    # Plot Beta (column 1) - using green
    ax2.plot(data_array[:, 1], label='Beta', color='g', linewidth=2)
    ax2.set_ylabel('Beta')
    ax2.grid(True)
    ax2.legend()
    
    # Plot Theta (column 2) - using yellow
    ax3.plot(data_array[:, 2], label='Theta', color='y', linewidth=2)
    ax3.set_ylabel('Theta')
    ax3.set_xlabel('Sample')
    ax3.grid(True)
    ax3.legend()
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save plot and close
    plt.savefig(f'{output_file}.png', dpi=300, bbox_inches='tight')
    plt.close()


print(f'Expected result: {res}')