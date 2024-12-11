import pandas as pd
import os

BASE_NUM = 41

def split_csv_files():
    for file_num in range(4):
        input_path = f'./testData/30/{file_num}.csv'
        
        output_dir = f'./output/{BASE_NUM + file_num + 1}'
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            df = pd.read_csv(input_path)
            
            data = df.iloc[181:3421]
            
            for i in range(0, 3240, 120):
                chunk = data.iloc[i:i+120]
                
                output_file = f'{i//120}_0.csv'
                output_path = os.path.join(output_dir, output_file)
                
                chunk.to_csv(output_path, index=False)
                print(f"✅ Created {output_path}")
                
        except Exception as e:
            print(f"❌ Error processing {input_path}: {e}")

split_csv_files()