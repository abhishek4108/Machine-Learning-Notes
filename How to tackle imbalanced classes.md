 # How to deal with a dataset having imbalanced classes?

<br>

## **1. Can You Collect More Data?**


## **2. Try Changing Your Performance Metric**

- **Confusion Matrix:** A breakdown of predictions into a table showing correct predictions (the diagonal) and the types of incorrect predictions made (what classes incorrect predictions were assigned).
- **Precision:** A measure of a classifiers exactness.
- **Recall:** A measure of a classifiers completeness
- **F1 Score (or F-score):** A weighted average of precision and recall.

## **3. Try Resampling Your Dataset**

1. You can add copies of instances from the under-represented class called **over-sampling** (or more formally sampling with replacement), or
2. You can delete instances from the over-represented class, called **under-sampling**

Wiki link - [Oversampling and undersampling in data analysis](https://en.wikipedia.org/wiki/Oversampling_and_undersampling_in_data_analysis)

**Some Rules of Thumb**
- Consider testing under-sampling when you have a lot of data (tens- or hundreds of thousands of instances or more)
- Consider testing over-sampling when you don’t have a lot of data (tens of thousands of records or less)
- Consider testing random and non-random (e.g. stratified) sampling schemes.
- Consider testing different resampled ratios (e.g. you don’t have to target a 1:1 ratio in a binary classification problem, try other ratios)

## **4. Try Generate Synthetic Samples**

A simple way to generate synthetic samples is to randomly sample the attributes from instances in the minority class.

You could sample them empirically within your dataset or you could use a method like Naive Bayes that can sample each attribute independently when run in reverse. You will have more and different data, but the non-linear relationships between the attributes may not be preserved.

As its name suggests, **SMOTE** is an oversampling method. It works by creating synthetic samples from the minor class instead of creating copies. The algorithm selects two or more similar instances (using a distance measure) and perturbing an instance one attribute at a time by a random amount within the difference to the neighboring instances.

There are a number of implementations of the SMOTE algorithm, for example:

- In Python, take a look at the “UnbalancedDataset” module. It provides a number of implementations of SMOTE as well as various other resampling techniques that you could try.
- In R, the DMwR package provides an implementation of SMOTE.

## **5. Try Different Algorithms**

## **6. Try Penalized Models**

Penalized classification imposes an additional cost on the model for making classification mistakes on the minority class during training. These penalties can bias the model to pay more attention to the minority class.

## **7. Try a Different Perspective**

Two you might like to consider are **anomaly detection** and **change detection**
