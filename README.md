# DL_Wine: Predicting Wine Quality from Weather Data

This project leverages deep learning to predict wine quality based on historical meteorological data and wine ratings collected from Vivino. The objective is to provide predictive insights on whether a specific year's climatic conditions are favorable for producing high-quality wine.

---

## Project Overview

This repository includes:
- **Data acquisition and preprocessing scripts**
- **Machine Learning (ML) models**
- **Analytical Jupyter notebooks**

---

## Installation

### Dependencies

Install required packages via `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy matplotlib xarray netcdf4 torch torchvision pytorch-lightning
```

---

## Data

### Weather Data

Weather data is sourced from Météo-France and organized by department:

- Historical data (1950-2023)
- Recent data (2024-2025)

Stored in:
```
data/weather/
data/weather_by_year/
data/weather_by_year_cleaned/
```

### Wine Data

Wine quality ratings scraped from Vivino:

- Raw data: `data/Wine/vivino_wines.csv`
- Regions metadata: `data/Wine/regions.csv`

---

## Quick Start

### Example: Loading Wine & Weather data

```python
import pandas as pd

wine_df = pd.read_csv('data/Wine/vivino_wines.csv')
weather_df = pd.read_parquet('data/weather_features/weather_features_33.parquet')

# Inspect datasets
print(wine_df.head())
print(weather_df.head())
```

---

## Project Structure

```
DL_Wine
├─ data/
│  ├─ weather/                   # Raw weather CSV data
│  ├─ weather_by_year/           # Weather data by year (raw)
│  ├─ weather_by_year_cleaned/   # Cleaned yearly weather data
│  ├─ weather_features/          # Extracted weather features
│  ├─ Wine/                      # Vivino scraped wine data
│  └─ weather_pq/                # Cleaned Parquet weather data
├─ src/
│  ├─ model/                     # ML model scripts and classes
│  ├─ preprocessing/
│  │  ├─ Weather/                # Weather preprocessing notebooks and CSV
│  │  └─ Wine & Weather/         # Scripts/notebooks for merging datasets
│  └─ scrapper/                  # Scraping scripts
├─ requirements.txt
├─ README.md
└─ License
```

---

## Workflow

### Data Preprocessing

Generate weather features:

```bash
cd src/preprocessing/Weather
jupyter notebook features_weather.ipynb
```

Merge Wine and Weather data:

```bash
cd src/preprocessing/"Wine & Weather"
jupyter notebook WQI.ipynb
```

### Model Training

Train and evaluate the predictive model:

```bash
cd src/model
python main_tabular.py
```

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/new-analysis`
3. Commit your changes: `git commit -m 'Add new analysis'`
4. Push your branch: `git push origin feature/new-analysis`
5. Create a Pull Request

---

## License

This project is licensed under the MIT License. See the `License` file for details.

---

Enjoy predicting wine quality!
```