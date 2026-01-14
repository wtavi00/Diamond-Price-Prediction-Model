import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

df = pd.read_csv("diamonds.csv")

df = df.drop(columns=['Unnamed: 0'], errors='ignore')  # Drop unnecessary index column if present

if df.isnull().sum().sum() > 0: #
    print("Warning: Missing values found. Dropping rows with missing values.")#
    df = df.dropna()#


le_cut = LabelEncoder()
le_color = LabelEncoder()
le_clarity = LabelEncoder()

cut_map = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
color_map = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
clarity_map = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

df['cut'] = le_cut.fit_transform(df['cut'])
df['color'] = le_color.fit_transform(df['color'])
df['clarity'] = le_clarity.fit_transform(df['clarity'])

X = df.drop(columns=['price'])
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=feature_names)
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae}")
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Diamond Prices")
plt.show()

joblib.dump({
    "model": model,
    "cut_map": cut_map,
    "color_map": color_map,
    "clarity_map": clarity_map
}, "diamond_price_pipeline.pkl")

print("Model saved as diamond_price_model.pkl")

def predict_new_diamond(model, scaler, le_cut, le_color, le_clarity, carat, cut, color, clarity, x, y, z, depth, table):
    data = pd.DataFrame([{
        "carat": carat,
        "cut": le_cut.transform([cut])[0],
        "color": le_color.transform([color])[0],
        "clarity": le_clarity.transform([clarity])[0],
        "x": x,
        "y": y,
        "z": z,
        "depth": depth,
        "table": table
    }])
    data_scaled = scaler.transform(data)
    return model.predict(data_scaled)[0]
  
#----------------------------
# Example Usages
#----------------------------

price = predict_new_diamond(model, scaler, le_cut, le_color, le_clarity, 0.5, "Ideal", "E", "VS2", 5.0, 5.1, 3.2, 61.5, 55)
print(f"Predicted price: {price}")
