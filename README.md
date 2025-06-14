# DL_Wine: Wine Quality Prediction

This repository contains code and analysis to determine whether a given year is suitable for purchasing a specific wine based on meteorological data and historical wine grades. The goal is to predict wine quality based on climatic factors.

---

## Installation

### Dependencies
Install necessary Python packages:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install xarray netcdf4 pandas numpy matplotlib
```

Alternatively, using `conda`:

```bash
conda install -c conda-forge xarray netcdf4 pandas numpy matplotlib
```

---

## Dataset

### Meteorological Data
Download the meteorological dataset from the official Météo-France repository:

- [Météo-France Dataset (ID: 6569b51ae64326786e4e8e1a)](https://meteo.data.gouv.fr/datasets/6569b51ae64326786e4e8e1a)

Save this data in the `data/weather_csv` directory.

### Wine Quality Data
The wine quality data has been collected via web scraping (see `src/scraper/NB.ipynb`).

---

## Quick Start

Load and inspect weather data quickly:

```python
import pandas as pd

# Example loading data
df = pd.read_csv('data/weather_csv/Q_33_previous-1950-2023_RR-T-Vent.csv')
print(df.head())
```

---

## Project Structure

```
DL_Wine
├─ data/
│  ├─ weather_csv/
│  └─ usable_stations/
├─ src/
│  ├─ model/  # Model training and prediction scripts
│  ├─ preprocessing/  # Data cleaning and feature extraction
│  └─ scraper/  # Web scraping scripts for wine quality data
├─ requirements.txt
├─ README.md
└─ License
```

---

## Workflow

### Data Preprocessing

Preprocess weather data and create feature datasets:

```bash
cd src/preprocessing
jupyter notebook features_weather.ipynb
```

### Model Training

Train and evaluate models:

(Details to be completed)

---

## Analysis & Visualization

Use provided notebooks to visualize and analyze weather features and wine quality relationships:

- `Verification_weather.ipynb`: Checks the consistency of weather data.

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

**Enjoy your wine prediction journey!**
