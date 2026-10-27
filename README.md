# Renewable Energy Analysis and Prediction

Google Colab notebook + live Streamlit dashboard. Dataset: renewable_energy.csv (1960–2015, 142 location codes).

## Run in Google Colab
Upload `Renewable_Energy_Colab.ipynb` to Colab and run it. When prompted, upload `renewable_energy.csv`.

## Deploy to Streamlit Community Cloud
Upload all four files to a GitHub repository. Visit https://share.streamlit.io and choose the repo, branch `main`, and main file `app.py`. The bundled CSV loads automatically.

## Methods
EDA and Random Forest regression. Next-year target is estimated from the preceding 3 consecutive years, with chronological train/test split. KTOE and percentage share are modeled separately. Source data ends in 2015; location codes can include regional aggregates.
