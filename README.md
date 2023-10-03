# Sentiment Analysis on Amazon Fine Food Reviews

![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)

## Overview

This repository contains the code for building a sentiment analysis model on the Amazon Fine Food Reviews dataset using logistic regression. The model predicts whether a review is positive or negative based on its content.

### Table of Contents

- [Project Overview](#overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Data Analysis](#data-analysis)
- [Data Preprocessing](#data-preprocessing)
- [Building the Model](#building-the-model)
- [Testing and Evaluation](#testing-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

### Prerequisites

Before running the code, make sure you have the following prerequisites installed:

- Python (>=3.0)
- Jupyter Notebook (optional for interactive analysis)
- Required Python libraries (NumPy, Pandas, Seaborn, Plotly, NLTK, WordCloud, Scikit-Learn)

## Data Analysis

In this section, we explore the dataset and visualize the distribution of product scores using Plotly.

```python
# fig = px.histogram(df, x="Score")
fig.update_traces(marker_color="turquoise",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Product Score')
fig.show()
```

## Data Preprocessing

We preprocess the data by classifying reviews into positive and negative sentiments based on the "Score" column.

```python
# df['sentimentt'] = df['sentiment'].replace({-1 : 'negative'})
df['sentimentt'] = df['sentimentt'].replace({1 : 'positive'})
fig = px.histogram(df, x="sentimentt")
fig.update_traces(marker_color="indianred",marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5)
fig.update_layout(title_text='Product Sentiment')
fig.show() 
```

## Building the Model

We build a sentiment analysis model using logistic regression. This includes data preparation, feature extraction, model training, and making predictions.

```python
# from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs', max_iter=6000)
X_train = train_matrix
X_test = test_matrix
y_train = train['sentiment']
y_test = test['sentiment']
Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

## Testing and Evaluation

We evaluate the model's performance by calculating accuracy, precision, recall, and F1-score on the test data.

```python
# precision    recall  f1-score   support

          -1       0.70      0.80      0.75     15110
           1       0.97      0.95      0.96     96654

    accuracy                           0.93    111764
   macro avg       0.83      0.88      0.85    111764
weighted avg       0.93      0.93      0.93    111764
```

## Results

The overall accuracy of the model on the test data is approximately 93%. We provide more details on the results in the Jupyter Notebook and visualization files.

## Contributing

Contributions are welcome! Please feel free to open issues or pull requests to improve this project.


---

Feel free to customize this README to provide specific details about your project, such as dataset sources, project goals, and any additional information you'd like to include. Make sure to replace the placeholder URLs, paths, and descriptions with your project-specific details.
