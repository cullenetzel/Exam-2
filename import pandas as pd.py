import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# Read the Excel file
file_path = "C:/Users/etzelcr16/Downloads/Restaurant Revenue.xlsx"
restaurant_data = pd.read_excel(file_path)

# Selecting relevant columns for prediction
selected_features = ['Number_of_Customers', 'Menu_Price', 'Marketing_Spend',
                     'Average_Customer_Spending', 'Promotions', 'Reviews']
X = restaurant_data[selected_features]
y = restaurant_data['Monthly_Revenue']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Building the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print()

# Regression statistics
X_train = sm.add_constant(X_train) # Adding a constant for the intercept
model_stats = sm.OLS(y_train, X_train).fit()
print(model_stats.summary())

print("Go Brewers")
