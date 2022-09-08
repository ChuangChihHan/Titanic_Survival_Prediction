"""
File: titanic_level2.py
Name: Maggie
----------------------------------
This file builds a machine learning algorithm by pandas and sklearn libraries.
We'll be using pandas to read in dataset, store data into a DataFrame,
standardize the data by sklearn, and finally train the model and
test it on kaggle. Hyperparameters are hidden by the library!
This abstraction makes it easy to use but less flexible.
You should find a good model that surpasses 77% test accuracy on kaggle.
"""

import math
import pandas as pd
from sklearn import preprocessing, linear_model
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'

# Global variable
nan_cache = {}                                    # Cache for test set missing data


def data_preprocess(filename, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be read into pandas
	:param mode: str, indicating the mode we are using (either Train or Test)
	:param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
						  (You will only use this when mode == 'Test')
	:return: Tuple(data, labels), if the mode is 'Train'
			 data, if the mode is 'Test'
	"""
	data = pd.read_csv(filename)
	labels = None
	if mode == 'Train':
		features = ['Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
		data = data[features].dropna()

		labels = data['Survived']

		data = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

		# Sex: Male = 1; Female = 0
		data.loc[data['Sex'] == 'male', 'Sex'] = 1
		data.loc[data['Sex'] == 'female', 'Sex'] = 0

		# Embarked: Changing 'S' to 0, 'C' to 1, 'Q' to 2
		data.loc[data['Embarked'] == 'S', 'Embarked'] = 0
		data.loc[data['Embarked'] == 'C', 'Embarked'] = 1
		data.loc[data['Embarked'] == 'Q', 'Embarked'] = 2

		# Count the number of data
		# data = data.count(axis='rows')

		# Calculate the mean
		# Age
		age_mean = data['Age'].mean()
		age_mean = round(age_mean, 3)
		# Fare
		fare_mean = data['Fare'].mean()
		fare_mean = round(fare_mean, 3)

		# Cache some data for test set (Age and Fare)
		nan_cache['Age'] = age_mean
		nan_cache['Fare'] = fare_mean

		# transpose the data
		# data = data.transpose()

		return data, labels
	elif mode == 'Test':
		features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
		data = data[features]

		# Fill in the NaN cells by the values in nan_cache to make it consistent
		data['Age'].fillna(nan_cache['Age'], inplace=True)
		data['Fare'].fillna(nan_cache['Fare'], inplace=True)

		# Sex: Male = 1; Female = 0
		data.loc[data['Sex'] == 'male', 'Sex'] = 1
		data.loc[data['Sex'] == 'female', 'Sex'] = 0

		# Embarked: Changing 'S' to 0, 'C' to 1, 'Q' to 2
		data.loc[data['Embarked'] == 'S', 'Embarked'] = 0
		data.loc[data['Embarked'] == 'C', 'Embarked'] = 1
		data.loc[data['Embarked'] == 'Q', 'Embarked'] = 2

		# count the number of data
		# data = data.count(axis='rows')

		# transpose the data
		# data = data.transpose()

		return data


def one_hot_encoding(data, feature):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: DataFrame, remove the feature column and add its one-hot encoding features
	"""
	if feature == 'Sex':
		# One hot encoding for new categories
		data['Sex_0'] = 0  # female(sex = 0)  = 1
		data['Sex_1'] = 0  # male(sex = 1) = 1
		data.loc[data['Sex'] == 1, 'Sex_1'] = 1  # Sex_1 = 1 / sex = male(1)
		data.loc[data['Sex'] == 0, 'Sex_0'] = 1  # Sex_0 = 1 / sex = female(0)
		# no need sex anymore
		data.pop('Sex')
	elif feature == 'Pclass':
		data['Pclass_0'] = 0  # Pclass1 (1)
		data['Pclass_1'] = 0  # Pclass2 (2)
		data['Pclass_2'] = 0  # Pclass3 (3)
		data.loc[data['Pclass'] == 1, 'Pclass_0'] = 1
		data.loc[data['Pclass'] == 2, 'Pclass_1'] = 1
		data.loc[data['Pclass'] == 3, 'Pclass_2'] = 1
		# no need Pclass anymore
		data.pop('Pclass')
	elif feature == 'Embarked':
		data['Embarked_0'] = 0  # S(0)
		data['Embarked_1'] = 0  # C(1)
		data['Embarked_2'] = 0  # Q(2)
		data.loc[data['Embarked'] == 0, 'Embarked_0'] = 1
		data.loc[data['Embarked'] == 1, 'Embarked_1'] = 1
		data.loc[data['Embarked'] == 2, 'Embarked_2'] = 1
		# no need Embarked anymore
		data.pop('Embarked')
	return data


def standardization(data, mode='Train'):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param mode: str, indicating the mode we are using (either Train or Test)
	:return data: DataFrame, standardized features
	"""
	standardizer = preprocessing.StandardScaler()
	if mode == 'Train':
		data = standardizer.fit_transform(data)
	elif mode == 'Test':
		data = standardizer.transform(data)
	return data


def main():
	"""
	You should call data_preprocess(), one_hot_encoding(), and
	standardization() on your training data. You should see ~80% accuracy
	on degree1; ~83% on degree2; ~87% on degree3.
	Please write down the accuracy for degree1, 2, and 3 respectively below
	(rounding accuracies to 8 decimals)
	TODO: real accuracy on degree1 ->  0.8019662921348315
	TODO: real accuracy on degree2 ->  0.8370786516853933
	TODO: real accuracy on degree3 ->  0.8764044943820225
	"""
	# data preprocessing
	train_data, labels = data_preprocess(TRAIN_FILE, mode='Train', training_data=None)
	test_data = data_preprocess(TEST_FILE, mode='Test', training_data=train_data)

	# one hot encoding
	# train data
	train_data = one_hot_encoding(train_data, 'Sex')
	train_data = one_hot_encoding(train_data, 'Pclass')
	train_data = one_hot_encoding(train_data, 'Embarked')

	# test data
	test_data = one_hot_encoding(test_data, 'Sex')
	test_data = one_hot_encoding(test_data, 'Pclass')
	test_data = one_hot_encoding(test_data, 'Embarked')

	# standardization
	standardizer = preprocessing.StandardScaler()
	train_data = standardizer.fit_transform(train_data)
	test_data = standardizer.transform(test_data)

	#############################
	# Degree 1 Polynomial Model #
	#############################
	h = linear_model.LogisticRegression(max_iter=1000)
	classifier = h.fit(train_data, labels)
	acc = classifier.score(train_data, labels)
	print('Training Accuracy:', acc)

	# Test dataset
	predictions = classifier.predict(test_data)
	print(f'Predictions: {predictions}')

	#############################
	# Degree 2 Polynomial Model #
	#############################
	poly_phi = preprocessing.PolynomialFeatures(degree=2)
	train_data_poly = poly_phi.fit_transform(train_data)

	# Test dataset
	test_poly = poly_phi.fit_transform(test_data)
	classifier2 = h.fit(train_data_poly, labels)
	acc_poly = classifier2.score(train_data_poly, labels)
	print('Training Accuracy:', acc_poly)
	predictions_poly = classifier2.predict(test_poly)
	print(f'Predictions: {predictions_poly}')

	#############################
	# Degree 3 Polynomial Model #
	#############################
	poly_phi_degree_3 = preprocessing.PolynomialFeatures(degree=3)
	train_data_poly_degree_3 = poly_phi_degree_3.fit_transform(train_data)

	# Test dataset
	test_poly_3 = poly_phi_degree_3.fit_transform(test_data)
	classifier3 = h.fit(train_data_poly_degree_3, labels)
	acc_poly_3 = classifier3.score(train_data_poly_degree_3, labels)
	print('Training Accuracy:', acc_poly_3)
	predictions_poly_2 = classifier3.predict(test_poly_3)
	print(f'Predictions: {predictions_poly_2}')


if __name__ == '__main__':
	main()
