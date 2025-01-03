# Amazon Stock Forecasting Using LSTM and PyTorch

This project focuses on forecasting Amazon stock prices using time-series data and LSTM (Long Short-Term Memory) neural networks, implemented in PyTorch. The project demonstrates the complete pipeline for stock price prediction, including data preprocessing, model training, and visualization of results.

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Steps and Methodology](#steps-and-methodology)
    - [1. Data Preprocessing](#1-data-preprocessing)
    - [2. Data Preparation for LSTM](#2-data-preparation-for-lstm)
    - [3. Defining the LSTM Model](#3-defining-the-lstm-model)
    - [4. Training the Model](#4-training-the-model)
    - [5. Evaluating and Visualizing Results](#5-evaluating-and-visualizing-results)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Results](#results)


---

## Overview

Stock price forecasting is a challenging problem in financial analysis, requiring accurate models to predict future trends. This project leverages LSTM, a type of recurrent neural network (RNN) well-suited for time-series data, to predict the stock prices of Amazon.

---

## Dataset

The dataset used for this project is sourced from a CSV file named `AMZN.csv`, containing historical stock price data for Amazon from **1997 to 2023**. The dataset includes the following columns:

- `Date`: The date of the stock price.
- `Open`: The opening price of the stock on the given date.
- `High`: The highest price of the stock during the trading session.
- `Low`: The lowest price of the stock during the trading session.
- `Close`: The closing price of the stock on the given date.
- `Adj Close`: The adjusted closing price (accounting for splits and dividends).
- `Volume`: The number of shares traded during the session.

### Columns Used:
For this project, only the `Date` and `Close` columns are utilized:
- **Date**: Serves as the timeline for the time-series analysis.
- **Close**: Represents the target variable for stock price prediction.

This simplified focus allows the model to predict future closing prices based on historical trends while keeping preprocessing straightforward.

---

## Steps and Methodology

### 1. Data Preprocessing

The raw dataset is cleaned and prepared to ensure compatibility with the model. Key steps include:

- Extracting relevant columns: Only the `Date` and `Close` columns are used.
- Converting the `Date` column to a datetime format.
- Plotting the stock price over time for visualization.

### 2. Data Preparation for LSTM

Time-series data requires transformation into sequences suitable for LSTM models. The following steps are applied:

- Normalization: Scaling the `Close` values to a range of [-1, 1] to improve model performance.
- Sequence creation: Splitting the data into sequences of fixed length (`n_steps`), where each sequence represents input features and target labels.

### 3. Defining the LSTM Model

The LSTM model is defined in PyTorch, consisting of the following components:

- Input layer: Accepts sequences of stock prices.
- Hidden layers: Includes one or more LSTM layers to capture temporal dependencies.
- Output layer: Predicts the next stock price in the sequence.

### 4. Training the Model

The model is trained using the following configuration:

- Loss function: Mean Squared Error (MSE) to minimize prediction errors.
- Optimizer: Adam optimizer for efficient gradient descent.
- Device: GPU or CPU, depending on availability.

### 5. Evaluating and Visualizing Results

After training, the model's performance is evaluated:

- Predictions are compared against actual stock prices.
- Results are visualized using matplotlib to display trends and forecast accuracy.

---

## Usage

1. **Load the dataset**:
   Ensure the file `AMZN.csv` is in the same directory as the project notebook.

2. **Run the notebook**:
   Open `time_ser.ipynb` in Jupyter Notebook or any compatible IDE:
   - Execute each cell sequentially to preprocess data, train the LSTM model, and generate predictions.
   - The notebook includes code blocks for each step: data preprocessing, LSTM model training, and evaluation.

3. **Visualize results**:
   The notebook automatically generates plots that compare the predicted stock prices with actual values to assess model performance.

   Example visualization:
   - X-axis: Time (dates)
   - Y-axis: Stock prices
   - Lines: Actual vs. Predicted prices
   ![Stock Price Prediction](Visualization.png)

4. **Adjust configurations**:
   Modify hyperparameters such as `n_steps`, learning rate, or LSTM layers directly in the notebook to experiment with different setups.
## Results

The LSTM model demonstrates the ability to predict Amazon stock prices by capturing general trends over time. However, it also highlights the challenges of forecasting during periods of high volatility.

### Key Observations:

- **Strengths**:
  - Accurately tracks the overall upward or downward trends in stock prices.
  - Performs well during stable periods with gradual changes in price.

- **Limitations**:
  - Struggles to predict sharp, sudden changes caused by market shocks or news events.
  - Predictions may lag during highly volatile periods, reflecting the limitations of time-series models for chaotic data.

### Visualization:

The notebook generates plots that compare the actual and predicted stock prices. A typical visualization includes:

- **X-axis**: Time (dates).
- **Y-axis**: Stock prices.
- **Lines**: 
  - The actual stock prices are displayed as a continuous line.
  - The predicted stock prices are overlaid for direct comparison.

These visualizations provide a clear picture of where the model performs well and where it diverges, helping to identify potential areas for improvement.

### Example Insights:

- The model predicts long-term trends more effectively than short-term fluctuations.
- Training on larger datasets or including additional features could enhance short-term prediction accuracy.

Let me know if you'd like me to expand this further with details on numerical metrics (e.g., RMSE, MAE) or specific plots!```

## Installation

Follow these steps to set up the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/anmit/Amazon-Stock-Forecasting.git
   cd Amazon-Stock-Forecasting
