# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset
file_path = '/content/AirBNB-1.csv'  # Change the path if needed
airbnb_data = pd.read_csv(file_path)

# Data Cleaning
# Fill missing numerical values with median
numerical_cols = ['accommodates', 'bathrooms', 'review_scores_rating', 'bedrooms', 'beds']
airbnb_data[numerical_cols] = airbnb_data[numerical_cols].fillna(airbnb_data[numerical_cols].median())

# Fill missing categorical values with mode
categorical_cols = ['room_type', 'cancellation_policy', 'cleaning_fee']
for col in categorical_cols:
    airbnb_data[col] = airbnb_data[col].fillna(airbnb_data[col].mode()[0])

# Convert cleaning_fee and instant_bookable to binary
airbnb_data['cleaning_fee'] = airbnb_data['cleaning_fee'].apply(lambda x: 1 if str(x).lower() in ['true', 't', 'yes', '1'] else 0)
airbnb_data['instant_bookable'] = airbnb_data['instant_bookable'].apply(lambda x: 1 if str(x).lower() in ['true', 't', 'yes', '1'] else 0)

# Data Visualization
# Distribution of log_price
plt.figure(figsize=(8, 6))
sns.histplot(airbnb_data['log_price'], kde=True, bins=30, color='blue')
plt.title('Distribution of Log Price')
plt.xlabel('Log Price')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()

# Market share of cancellation policies
plt.figure(figsize=(8, 6))
cancellation_policy_counts = airbnb_data['cancellation_policy'].value_counts()
cancellation_policy_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title('Market Share of Cancellation Policies')
plt.ylabel('')
plt.show()

# Price variation by bedrooms
plt.figure(figsize=(10, 6))
sns.boxplot(x='bedrooms', y='log_price', data=airbnb_data, palette="viridis")
plt.title('Log Price Variation by Number of Bedrooms')
plt.xlabel('Number of Bedrooms')
plt.ylabel('Log Price')
plt.grid(axis='y')
plt.show()

# Price variation by accommodates
plt.figure(figsize=(12, 6))
sns.boxplot(x='accommodates', y='log_price', data=airbnb_data, palette="coolwarm")
plt.title('Log Price Variation by Number of Accommodates')
plt.xlabel('Number of Accommodates')
plt.ylabel('Log Price')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# Correlation analysis
correlation_matrix = airbnb_data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix[['log_price']].sort_values(by='log_price', ascending=False), 
            annot=True, cmap="coolwarm", cbar=True, fmt=".2f")
plt.title('Correlation with Log Price')
plt.show()

# Linear Regression Model
# Select features and target variable
features = ['accommodates', 'bedrooms', 'beds', 'bathrooms', 'cleaning_fee', 'review_scores_rating']
X = airbnb_data[features]
y = airbnb_data['log_price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display results
print("Linear Regression Model Performance:")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"R^2 Score: {r2:.3f}")

# Display feature coefficients
coefficients = pd.DataFrame({
    'Feature': features,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)
print("\nFeature Coefficients:")
print(coefficients)
