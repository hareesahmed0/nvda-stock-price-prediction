# NVIDIA Stock Price Prediction using Deep Learning (LSTM)

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Time%20Series-green)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

A deep learning project that models **NVIDIA (NVDA) stock price movements** using a **Bidirectional LSTM neural network** trained on historical market data.  
The project demonstrates how time-series models can capture underlying trends in **noisy financial data** while applying proper evaluation techniques to **avoid data leakage**.

---

# Project Overview

Financial markets are highly **noisy and influenced by unpredictable external events**, making accurate forecasting difficult.  

This project explores whether deep learning can learn meaningful patterns from historical stock prices and generate **short-term predictions**.

The model analyzes the **previous 30 days of market data** to predict the **next-day closing price of NVIDIA stock**.

---

# Key Features

• Time-series forecasting using **Bidirectional LSTM**  
• Historical stock data retrieved from **Yahoo Finance API**  
• Feature engineering with financial indicators  
• Sliding window sequence modeling  
• Chronological train-test split to **prevent data leakage**  
• Evaluation on **~9 months of unseen test data**  

---

# Data Source

Historical stock data was obtained using the **Yahoo Finance API (yfinance)**.

The dataset includes:

- Open
- High
- Low
- Close
- Volume

Time range used:

```
2022 – 2026
```

---

# Feature Engineering

To improve model performance, several financial indicators were added:

• Moving Average (MA-5, MA-10, MA-20)  
• Rolling Volatility (standard deviation of returns)  
• Volume  
• Closing price  

A **30-day sliding window** converts the time-series into supervised learning sequences.

Example:

```
Past 30 days of features → Predict next day closing price
```

---

# Model Architecture

The model uses a **Bidirectional LSTM network**, which processes sequences in both forward and backward directions during training to learn stronger temporal patterns.

Architecture:

```
Input Sequence (30 days)
        ↓
Bidirectional LSTM Layer
        ↓
Dropout Layer
        ↓
Dense Layer
        ↓
Next-Day Price Prediction
```

Why LSTM?

• Designed for sequential data  
• Captures temporal dependencies  
• Handles long-term patterns better than traditional models  

---

# Avoiding Data Leakage

Time-series models are prone to **look-ahead bias** if data is split incorrectly.

To prevent leakage:

• Data was split **chronologically**, not randomly  
• Training data contains **only past observations**  
• Test data contains **future unseen observations**  
• Feature scaling was fitted **only on training data**  

This ensures a **realistic evaluation of forecasting performance**.

---

# Results

The model was evaluated on approximately:

```
~200 trading days (~9 months) of unseen data
```

Performance:

```
Mean Absolute Error (MAE): ~7 USD
```

Observations:

• Model successfully captures **long-term trends**  
• Predictions are **smoother than actual prices**  
• Sudden spikes caused by news or macro events remain difficult to predict  

This behavior is expected when modeling **noisy financial time series**.

---

# Technologies Used

Python  
TensorFlow / Keras  
NumPy  
Pandas  
Scikit-Learn  
Matplotlib  
Yahoo Finance API (yfinance)

---

# Project Structure

```
nvda-stock-price-prediction/
│
├── Nvidia_stock_prediction.ipynb
├── README.md
└── requirements.txt
```

---

# Installation

Clone the repository:

```
git clone https://github.com/hareesahmed0/nvda-stock-price-prediction.git
cd nvda-stock-price-prediction
```

Install dependencies:

```
pip install -r requirements.txt
```

Run the notebook:

```
jupyter notebook Nvidia_stock_prediction.ipynb
```

---

# Key Learnings

This project highlights several important concepts in machine learning:

• Modeling **noisy financial time series**  
• Importance of **time-based train/test splits**  
• Preventing **data leakage in forecasting models**  
• Applying **deep learning architectures for sequential data**  

It also demonstrates the limitations of ML models in predicting **external market shocks**.

---

# Disclaimer

This project is for **educational and research purposes only**.

It does **not constitute financial advice** and should not be used for real trading decisions.

---

# Author

**Harees Ahmed**

GitHub:  
https://github.com/hareesahmed0
