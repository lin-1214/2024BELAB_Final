import pandas as pd
import os

BASE_NUM = 41

def split_csv_files():
    # Process files 0.csv to 3.csv
    for file_num in range(4):
        input_path = f'./testData/30/{file_num}.csv'
        
        # Create output directory for each file
        output_dir = f'./output/{BASE_NUM + file_num + 1}'
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Read the original CSV
            df = pd.read_csv(input_path)
            
            # Skip header row and first 180 rows, and exclude last 180 rows
            # Original: 1:3601 (3600 rows)
            # New: 181:3421 (3240 rows)
            data = df.iloc[181:3421]
            
            # Split into chunks of 120 rows
            for i in range(0, 3240, 120):  # Changed to 3240 (3600 - 360)
                chunk = data.iloc[i:i+120]
                
                # Calculate the output file number
                output_file = f'{i//120}_0.csv'
                output_path = os.path.join(output_dir, output_file)
                
                # Save chunk to new CSV file
                chunk.to_csv(output_path, index=False)
                print(f"✅ Created {output_path}")
                
        except Exception as e:
            print(f"❌ Error processing {input_path}: {e}")

# Run the splitting function
split_csv_files()