# Parkinson's Disease - ML Detection
This project aims to develop a machine learning model to predict whether a person has Parkinson's disease based on various biomedical voice measurements. Using a dataset containing features extracted from voice recordings of individuals. 

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Machine Learning Prediction](#machine-learning-prediction)
    - [Install dependencies](#install-dependencies)
    - [Data collection and analysis](#data-collection-and-analysis)
    - [Separating features and target](#separating-features-and-target)
    - [Splitting data into Train and Test data](#splitting-data-into-train-and-test-data)
    - [Data Standardization](#data-standardization)
    - [Model training - Support Vector Machine](#model-training---support-vector-machine)
    - [Model evaluation - Accuracy Score](#model-evaluation---accuracy-score)
    - [Building a predictive system](#building-a-predictive-system)
    - [Saving the trained model](#saving-the-trained-model)

## Project Overview
Parkinson's disease is a progressive neurological disorder that affects movement and can lead to significant disability. Early and accurate diagnosis is crucial for effective management and treatment. This project leverages data-driven approaches to aid in the early detection of the disease.

**Goals:**
- **Develop a Predictive Model:** Create a robust machine learning model capable of accurately predicting Parkinson's disease from voice measurements.
- **Data Exploration and Preprocessing:** Understand the dataset's structure, clean the data, and preprocess it to ensure it is suitable for machine learning algorithms.
- **Model Training and Evaluation:** Train a Support Vector Machine (SVM) classifier and evaluate its performance to ensure it generalizes well to new data.
- **Make Predictions on New Data:** Use the trained model to predict the presence of Parkinson's disease in new, unseen data points.
- **Raise Awareness:** Demonstrate the potential of machine learning in healthcare applications, specifically for the early detection of neurological disorders.

## Dataset
This [dataset](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set) is composed of a range of biomedical voice measurements from 31 people, 23 with Parkinson's disease (PD). Each column in the table is a particular voice measure, and each row corresponds to one of 195 voice recordings from these individuals ("name" column). The main aim of the data is to discriminate healthy people from those with PD, according to the "status" column which is set to 0 for healthy and 1 for PD.

Matrix column entries (attributes):
- **name** - ASCII subject name and recording number
- **MDVP:Fo(Hz)** - Average vocal fundamental frequency
- **MDVP:Fhi(Hz)** - Maximum vocal fundamental frequency
- **MDVP:Flo(Hz)** - Minimum vocal fundamental frequency
- **MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP** - Several measures of variation in fundamental frequency
- **MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA** - Several measures of variation in amplitude
- **NHR, HNR** - Two measures of the ratio of noise to tonal components in the voice
- **status** - The health status of the subject (one) - Parkinson's, (zero) - healthy
- **RPDE, D2** - Two nonlinear dynamical complexity measures
- **DFA** - Signal fractal scaling exponent
- **Spread1,spread2,PPE** - Three nonlinear measures of fundamental frequency variation

## Machine Learning Prediction
### Install dependencies
Outline the essential steps for preparing data, training an SVM model, making predictions, and evaluating the model's accuracy.

### Data collection and analysis
Perform initial data loading, exploration, and preprocessing for a Parkinson's disease dataset.
- Load the dataset into a pandas DataFrame.
- Explore the data by displaying the first few rows, check the shape, summarize the data, list column names, generate descriptive statistics, and check for missing values.
- Examine the distribution of the target variable and drop an unnecessary column(s).
- The data is grouped by the target variable to calculate mean values for each group.
- Rename several columns for clarity.

### Separating features and target
- Separate the features and the target variable from the Parkinson's disease dataset.
- Assign all feature columns (excluding status) to the DataFrame X and the target variable (status) to the Series Y.
- Print both X and Y to inspect their contents.
- Iterate over the feature column names and prints each one, which can be useful for later use, such as in creating user interfaces or web applications.

### Splitting data into Train and Test data
- Split the Parkinson's disease dataset into training and testing sets, with 80% of the data for training and 20% for testing.
- Print the shapes of the original, training, and testing feature sets to verify the split.

### Data Standardization
- Standardize the features by fitting a StandardScaler to the training data and transforming both the training and testing data. This ensures that the features have a mean of 0 and a standard deviation of 1, which can improve the performance of many machine learning algorithms.

### Model training - Support Vector Machine
- Initialize a Support Vector Machine (SVM) classifier with a linear kernel. The SVM classifier is designed to find the best linear separation between the classes in the training dataset.
- Train it using the standardized training data. Once trained, this classifier can be used to make predictions on new, unseen data.

### Model evaluation - Accuracy Score
- Evaluate the performance of the trained SVM classifier by calculating and printing the accuracy scores on both the training and testing datasets.
- Make predictions on the training data and calculate the accuracy by comparing the predicted labels with the true labels.
- Make predictions on the testing data and calculate the accuracy similarly. 
- Print the accuracy scores for both the training and testing datasets, allowing you to assess how well the model performs on known and unseen data.

### Building a predictive system
Build a system that:

- Takes a new set of input features that represent a single data point.
- Standardizes it using the previously fitted scaler,
- Uses the trained SVM classifier to predict whether the person has Parkinson's disease.
- The result of the prediction is then printed, along with an interpretation indicating whether the person is likely to have Parkinson's disease or not.

### Saving the trained model
- Save and load a trained machine learning model and its associated scaler using the pickle module.
- Use these saved objects to make predictions on new data.
    - Convert the new data into the required format
    - Standardize it using the previously saved scaler
    - Make a prediction using the loaded SVM model. 
- Interpret and print whether the prediction indicates the presence of Parkinson's disease.
