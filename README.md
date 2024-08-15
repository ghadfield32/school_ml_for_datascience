# Breast Cancer Prediction Project

This project uses machine learning to predict whether a breast mass is benign or malignant based on various features.

## Setup

1. Clone this repository
2. Create the Conda environment:
   ```
   conda env create -f environment.yml
   ```
3. Activate the environment:
   ```
   conda activate breast-cancer-pred
   ```
4. Run the Streamlit app:
   ```
   streamlit run app/app.py
   ```

## Project Structure

- `data/`: Contains raw and processed data
- `notebooks/`: Jupyter notebooks for exploration and analysis
- `src/`: Source code for data processing, feature engineering, and model training
- `app/`: Streamlit app for model deployment
- `tests/`: Unit tests
