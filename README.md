# Titanic_Survival_Prediction
- This dataset is from [Kaggle](https://www.kaggle.com/competitions/titanic/overview)

1. titanic level 1 file:
    This file builds a machine learning algorithm from scratch 
    by Python codes. We'll be using 'with open' to read in dataset,
    store data into a Python dict, and finally train the model and 
    test it on kaggle. This model is the most flexible one among all
    levels. You should do hyperparameter tuning and find the best model.

2. titanic level 2 file:
    This file builds a machine learning algorithm by pandas and sklearn libraries.
    We'll be using pandas to read in dataset, store data into a DataFrame,
    standardize the data by sklearn, and finally train the model and
    test it on kaggle. Hyperparameters are hidden by the library!
    This abstraction makes it easy to use but less flexible.

### Results:
Applying polynomial (degree = 3) model, the training accuracy: 0.8764044943820225
