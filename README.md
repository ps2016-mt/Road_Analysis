# Road_Analysis

# setup instructions
conda env create -f environment.yml
conda activate road_traffic_env

pre-commit install
pre-commit run --all-files

# data setup
Download the accident_data.csv and vehicle_data.csv files from the Kaggle Road Traffic Collision Dataset (https://www.kaggle.com/datasets/salmankhaliq22/road-traffic-collision-dataset?select=accident_data.csv).
Move the downloaded files into the data folder so the structure looks like this:
road_traffic_analysis/
├── data/
│   ├── accident_data.csv
│   ├── vehicle_data.csv

The project’s data-loading script is designed to automatically locate these files in the data/ folder and process them without requiring any changes to the file paths.�