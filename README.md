# House Price Prediction

A machine learning project that predicts house prices (in INR lakhs) for Srinagar localities from size, location, age and amenities.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=flat-square)

## Objective
Find out which features drive house prices and build a model that predicts the price of a new house.

## Dataset
`house_prices.csv` is a **simulated dataset** (generated with `generate_data.py`) of 1,812 listings. It contains duplicates, missing values and a few unrealistic area values so cleaning and outlier handling are realistic.

| Column | Description |
|---|---|
| `location` | Rajbagh, Jawahar Nagar, Sanat Nagar, Bemina, Hyderpora, Nowgam, Soura, Budgam Road, Pantha Chowk |
| `area_sqft`, `bedrooms`, `bathrooms`, `floors` | Size and layout |
| `age_years`, `distance_to_city_km` | Age of the house and distance from the city centre |
| `parking`, `furnishing` | Parking (Yes/No); Unfurnished, Semi-Furnished or Furnished |
| `price_lakhs` | **Target:** price in INR lakhs |

## Workflow
1. **Clean:** removed 12 duplicates; filled age (median), bathrooms and furnishing (mode).
2. **Outliers:** removed 10 unrealistic area values with the IQR rule. Result: 1,790 rows, saved as `house_prices_cleaned.csv`.
3. **EDA:** price distribution, area vs price, price by location, correlation heatmap (charts are in the notebook).
4. **Modelling:** 80/20 split; Linear Regression, Random Forest and Gradient Boosting.
5. **Evaluation:** R², MAE and RMSE on the test set, plus 5-fold cross-validation.
6. **Prediction:** the best model predicts the price of a new house.

## Results

| Model | R² | MAE (lakhs) | RMSE (lakhs) | CV R² |
|---|---|---|---|---|
| Linear Regression | 0.954 | 4.38 | 5.70 | 0.947 |
| Random Forest | 0.939 | 4.95 | 6.54 | 0.938 |
| Gradient Boosting | **0.958** | **4.23** | **5.46** | **0.961** |

Gradient Boosting is best, with predictions off by about 4.2 lakhs on average. Linear Regression is a strong baseline because price is largely linear in area.

## Key insights
- **Area is the biggest driver:** it has a correlation of about 0.79 with price and makes up about 60% of the Gradient Boosting model's feature importance.
- **Location matters:** median prices range from about 89.5 lakhs in Rajbagh to about 51.3 lakhs in Budgam Road.
- **Age and distance to the city** have a smaller negative effect on price.
- Cross-validation scores are close to the test scores, so the model is not simply memorising the training data.

## Files
| File | Purpose |
|---|---|
| `House_Price_Prediction.ipynb` | Full analysis and modelling |
| `house_prices.csv` / `house_prices_cleaned.csv` | Raw and cleaned data |
| `model_comparison.csv` | Model results table |
| `generate_data.py` | Generates the simulated dataset |

## How to run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook House_Price_Prediction.ipynb
```

## Author
**Aabroo Farooq** · [LinkedIn](https://www.linkedin.com/in/aabroofarooq99) · [Portfolio](https://aabroo-farooq.github.io/)
