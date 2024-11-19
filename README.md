# 2024BELAB_Final

## EEG Data Preprocessor
This tool processes EEG (Electroencephalogram) data by filtering and visualizing alpha, beta, and theta brain waves. The code is in `data_preprocessor.py`.

### Overview

The data preprocessor performs the following operations:
- Reads raw EEG data from CSV files
- Applies bandpass filtering to isolate specific frequency bands
- Generates visualizations of both raw and filtered data
- Processes data into averaged time segments
- Saves visualizations as PNG files
- Reform the data to the format that the model needs

### Configuration (config.json)

The `config.json` file controls the preprocessing parameters, modify once there is a new dataset to be processed.