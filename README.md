# Road_Analysis

# setup instructions
conda env create -f environment.yml
conda activate road_traffic_env

pre-commit install
pre-commit run --all-files

pip install -e .

# data setup
Download the accident_data.csv file from the Kaggle Road Traffic Collision Dataset (https://www.kaggle.com/datasets/salmankhaliq22/road-traffic-collision-dataset?select=accident_data.csv).
Move the downloaded files into the data folder so the structure looks like this:
road_traffic_analysis/
├── data/
│   ├── accident_data.csv

The project’s data-loading script is designed to automatically locate these files in the data/ folder and process them without requiring any changes to the file paths.

# data wrangling, cleaning and pre-processing
Run the eda_cleaning.ipynb notebook. Running this will process the raw csv file into a parquet file within your data folder, such that the the structure will resemble this:
├── data/
│   ├── accident_data.csv
│   ├── processed_data.parquet
The notebook also provides context into the feature selection and how the different features vary with the target varaible.

# main model running
Run the model_training_evaluation.ipynb notebook. Running this will load the produced parquet file, and use this as a base to run our GLM and LGBM modelling.�