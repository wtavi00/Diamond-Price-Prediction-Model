# Diamond Price Prediction Model

## Overview
This project builds a **diamond price prediction model** using machine learning techniques. The dataset consists of 54,000 diamond records, including attributes like carat, cut, color, clarity, and dimensions. The goal is to predict the price of a diamond based on these attributes.

## Dataset
The dataset contains the following features:
- **price**: Diamond price in US dollars ($326 - $18,823)
- **carat**: Weight of the diamond (0.2 - 5.01)
- **cut**: Quality of the cut (Fair, Good, Very Good, Premium, Ideal)
- **color**: Diamond color, from J (worst) to D (best)
- **clarity**: Measurement of how clear the diamond is (I1 (worst) to IF (best))
- **x, y, z**: Dimensions of the diamond in mm
- **depth**: Total depth percentage = z / mean(x, y)
- **table**: Width of the top of the diamond relative to the widest point

## Project Workflow
1. **Data Preprocessing**
   - Load the dataset
   - Handle missing values and unnecessary columns
   - Encode categorical variables (`cut`, `color`, `clarity`)
   - Feature scaling
   
2. **Model Building**
   - Split data into training and testing sets
   - Train a **Random Forest Regressor** model
   - Evaluate model performance using **MAE, MSE, RMSE, and R² Score**

3. **Visualization**
   - Scatter plot of actual vs. predicted diamond prices

## Installation & Usage

### Requirements

Ensure you have the following Python libraries installed:

```bash
pip install pandas numpy seaborn scikit-learn matplotlib
```

### Running the Model

1. Clone this repository:

```bash
cd diamond-price-prediction
```

2. Run the Python script:

```bash
python diamond_price_prediction_model.py
```

## Results

The model evaluates its performance using:
* **Mean Absolute Error (MAE)**
* **Mean Squared Error (MSE)**
* **Root Mean Squared Error (RMSE)**
* **R² Score**

A scatter plot is generated to visualize the accuracy of predictions compared to actual prices.

## Future Enhancements

* Experiment with other regression models (e.g., XGBoost, Gradient Boosting)
* Hyperparameter tuning for better performance
* Feature importance analysis

## Auther
[Avijit Tarafder](https://github.com/wtavi00)

## License
[MIT License](https://github.com/wtavi00/Diamond-Price-Prediction-Model/blob/main/LICENSE)
