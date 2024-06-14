# Deep Learning
This repository contains a set of notebooks realised as an introduction to Deep Learning. They describe different techniques and deep learning model architectures. They were a part of [ESIEE Paris](https://www.esiee.fr/)' Deep Learning course for 4th year apprentices in the "Computer science and applications" sector.

# 1. DRL Cela Tutorial Summary

## Introduction
The notebook starts by discussing the basics of Deep Reinforcement Learning (DRL) and its applications, specifically in the context of processing EEG signals.

## Architecture of ADALINE
- A section dedicated to explaining the architecture of an ADALINE (Adaptive Linear Neuron). This includes a visual diagram and a brief description of its operation and utility in adaptive filtering.

## Objective
- The main goal is outlined as creating and training a minimalist neural network to remove noise from EEG signals. The notebook describes the use of a basic gradient descent method for adjusting the weights adaptively to achieve noise cancellation.

## Implementation Details
- **Data Preparation**: Handling and preparing EEG signal data for processing.
- **Model Design**: Steps to design the neural network model including layer setup and parameter initialization.
- **Training**: Details on training the network using gradient descent, including setting learning rates and loss calculation.
- **Results and Evaluation**: Analyzing the performance of the trained model on test data, visualizing results to demonstrate the efficacy of noise removal.

## Conclusion
A summary of results achieved and potential areas for further exploration or improvement in future work.

## Appendix
Additional notes and references for further reading on the topics discussed.

# Chickenpox 2024 Vaccine Efficacy Analysis

## Introduction
The document begins with an introduction to the purpose of the analysis, which is to evaluate the efficacy of the chickenpox vaccine in 2024. It outlines the datasets used and the preliminary steps taken to prepare the data for analysis.

## Data Preparation
- **Data Loading**: Initial steps involve loading the chickenpox incidence data along with relevant demographic and vaccination data.
- **Data Cleaning**: Processes to clean and preprocess the data for analysis, including handling missing values and outliers.

## Data Analysis
- **Descriptive Statistics**: Provides basic statistics to understand the distribution and characteristics of the data.
- **Data Visualization**: Includes plots such as time series of chickenpox cases and vaccination rates to visualize trends and patterns.

## Predictive Modeling
- **Feature Engineering**: Discussion on the creation of new variables that could improve model performance.
- **Model Selection**: Comparison of different statistical and machine learning models to predict vaccine efficacy.
- **Model Training and Validation**: Details on how the models are trained and validated using historical data.

## Results
- **Model Performance**: Evaluation of the predictive accuracy of the models.
- **Interpretation of Results**: Insights drawn from the model outputs, discussing the factors influencing vaccine efficacy.

## Conclusion
Summarizes the findings and suggests implications for public health policy. Also discusses potential improvements for future analyses.

## Appendix
Additional resources and references used for the analysis.


# Ladybug Position Prediction

## Introduction
This notebook presents a study on predicting the future position of a ladybug using various neural network models. The goal is to compare the performance of different architectures in time series forecasting.

## Data Collection
- **Source**: Description of how the data on the ladybug's position was collected, including the tools and methods used for data acquisition.
- **Preprocessing**: Steps to clean and prepare the data for model training. This includes normalization, handling missing values, and splitting the data into training and testing sets.

## Exploratory Data Analysis (EDA)
- **Visualization**: Initial exploration of the dataset using visual tools like plots to understand the trends and patterns in the ladybug's movement.

## Neural Network Models
The notebook explores several neural network architectures for time series forecasting:

### 1. Recurrent Neural Network (RNN)
- An RNN model designed to capture temporal dependencies in the data.
- **Architecture**: Input layer, Simple RNN layer with 50 units, TimeDistributed Dense layer for output.

### 2. Long Short-Term Memory (LSTM)
- An LSTM model that addresses the vanishing and exploding gradient problems in traditional RNNs.
- **Architecture**: Input layer, LSTM layer with 50 units, TimeDistributed Dense layer for output.

### 3. Gated Recurrent Unit (GRU)
- A GRU model, a simplified version of LSTM that merges gates to simplify implementation.
- **Architecture**: Input layer, GRU layer with 50 units, TimeDistributed Dense layer for output.

### 4. Multi-Layer Perceptron (MLP)
- A simple feedforward neural network for comparison.
- **Architecture**: Input layer, Dense layer with 50 units, output Dense layer, Reshape layer to match the output shape.

## Model Building and Compilation
- Code snippets to build and compile each of the neural network models using Keras.

## Training and Evaluation
- **Training**: Details on training the models on the prepared dataset, including parameters like batch size, epochs, and optimization algorithms.
- **Evaluation**: Methods to evaluate the performance of the models, using metrics such as Mean Squared Error (MSE) to compare their effectiveness in predicting the ladybug's position.

## Results
- **Comparative Analysis**: Presentation of the results from different models, highlighting which model performed best in terms of accuracy and generalization.

## Conclusion
- Summarizes the findings and suggests which neural network model is most suitable for the task of predicting the ladybug's position.
- Recommendations for future work and potential improvements in model performance.

## Appendix
- Additional notes, code, or references used throughout the notebook for further reading or replication of the study.

