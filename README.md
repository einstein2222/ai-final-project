# Stock Price Movement Predictor

## Project Overview
This project predicts whether a stock price will go up or down the next trading day using historical stock market data and news sentiment.

The app is built with Streamlit and deployed on Hugging Face.

---

## Dataset
Data was collected using the yfinance library from Yahoo Finance.

The dataset included open, high, low, close and volume.


Data from 2020 to now was used for training.

Additional features were added, including moving averages, volatility, returns, and momentum, and news sentiment was an additional feature.


---

## Project Type
This is a classification project.

The model predicts:
- 1 = stock goes up
- 0 = stock goes down

---

## Models Used

### Logistic Regression
Baseline model

### XGBoost
Used as the main model due to better performance

---

## How the App Works
1. The user enters a stock ticker.
2. Stock data is downloaded, and indicators are evaluated
3. Recent news sentiment is analyzed
5. The model predicts whether the stock will go up or down the next day, with confidence scores, sentiment scores, recent news, and a prediction


---

## Evaluation
The models were evaluated with:
- accuracy
- precision
- recall
- F1 Score

---

## Tools Used
- Python
- Streamlit
- XGBoost
- scikit-learn
- pandas
- yfinance
- Hugging Face Spaces

---

## Running the Project Locally

do pip install -r requirements.txt
