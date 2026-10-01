import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load and clean data
df = pd.read_csv('data/dataset.csv')
df.fillna(df.median(), inplace=True)

# Train basic model
X = df[['SquareFeet', 'Bedrooms', 'Age_Years']]
y = df['Price']
model = LinearRegression()
model.fit(X, y)

# Build Dashboard Interface
st.title("Real Estate Price Predictor Dashboard")
st.write("Enter house details below to predict the market price.")

# User inputs
sqft = st.number_input("Square Feet", min_value=500, max_value=10000, value=1500)
beds = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
age = st.number_input("Age of House (Years)", min_value=0, max_value=100, value=10)

# Predict button
if st.button("Predict Price"):
    prediction = model.predict([[sqft, beds, age]])
    st.success(f"Estimated Price: ${prediction[0]:,.2f}")
    
st.subheader("Market Data Overview")
st.scatter_chart(df, x='SquareFeet', y='Price')