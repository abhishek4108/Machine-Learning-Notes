# Logistic Regression

<br>
<!-- toc -->

- [Logistic Regression](#logistic-regression)
  - [What is a logistic regression?](#what-is-a-logistic-regression)
  - [Logistic Regression Predicts Probabilities](#logistic-regression-predicts-probabilities)
  - [Learning the Logistic Regression Model](#learning-the-logistic-regression-model)
  - [Types of Logistic Regression](#types-of-logistic-regression)
  - [Performance of Logistics Regression Model](#performance-of-logistics-regression-model)
    - [1. Confusion Matrix](#1-confusion-matrix)
    - [2. ROC Curve](#2-roc-curve)

<!-- tocstop -->

## What is a logistic regression?
Logistic Regression is a classification algorithm. It is used to predict a binary outcome (1 / 0, Yes / No, True / False) given a set of independent variables.

In simple words, it predicts the probability of occurrence of an event by fitting data to a logit function.

Below is an example logistic regression equation:

y = e^(b0 + b1 * x) / (1 + e^(b0 + b1 * x))

Where y is the predicted output, b0 is the bias or intercept term and b1 is the coefficient for the single input value (x). Each column in your input data has an associated b coefficient (a constant real value) that must be learned from your training data

## Logistic Regression Predicts Probabilities

Logistic regression models the probability of the default class (e.g. the first class).

we are modeling the probability that an input (X) belongs to the default class (Y=1), we can write this formally as:

P(X) = P(Y=1|X)

Logistic regression is a linear method, but the predictions are transformed using the logistic function.

p(X) = e^(b0 + b1 * X) / (1 + e^(b0 + b1 * X))

ln(p(X) / 1 – p(X)) = b0 + b1 * X

Odds are calculated as a ratio of the probability of the event divided by the probability of not the event, e.g. 0.8/(1-0.8) which has the odds of 4. So we could instead write:

ln(odds) = b0 + b1 * X

Because the odds are log transformed, we call this left hand side the log-odds or the probit. We can move the exponent back to the right and write it as:

odds = e^(b0 + b1 * X)

## Learning the Logistic Regression Model

The coefficients (Beta values b) of the logistic regression algorithm must be estimated from your training data. This is done using maximum-likelihood estimation.

Maximum-likelihood estimation is a common learning algorithm used by a variety of machine learning algorithms, although it does make assumptions about the distribution of your data.

The intuition for maximum-likelihood for logistic regression is that a search procedure seeks values for the coefficients (Beta values) that minimize the error in the probabilities predicted by the model to those in the data (e.g. probability of 1 if the data is the primary class)

## Types of Logistic Regression

1. Binary Logistic Regression
   The categorical response has only two 2 possible outcomes. Example: Spam or Not
2. Multinomial Logistic Regression
   Three or more categories without ordering. Example: Predicting which food is preferred more (Veg, Non-Veg, Vegan)
3. Ordinal Logistic Regression
   Three or more categories with ordering. Example: Movie rating from 1 to 5

## Performance of Logistics Regression Model

### 1. Confusion Matrix

<img src="images/confusionmatrix.png">

Accuracy =  (TP + TN) / (TP + FP + FN + TN)

True Positive Rate = Sensitivity = Recall =  TP / (TP + FN)

False Positive Rate = FP / (FP + TN)

True Negative Rate = Specificity = TN / (TN + FP)

Precision = TP / (TP + FP)

### 2. ROC Curve

Receiver Operating Characteristic(ROC) summarizes the model’s performance by evaluating the trade offs between true positive rate (sensitivity) and false positive rate(1- specificity)

The area under curve (AUC), referred to as index of accuracy(A) or concordance index, is a perfect performance metric for ROC curve. Higher the area under curve, better the prediction power of the model. Below is a sample ROC curve. The ROC of a perfect predictive model has TP equals 1 and FP equals 0. This curve will touch the top left corner of the graph.

<img src="images/roc.png" width="50%" height="50%">
