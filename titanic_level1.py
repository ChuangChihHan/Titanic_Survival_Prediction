"""
File: titanic_level1.py
Name: Maggie
----------------------------------
This file builds a machine learning algorithm from scratch 
by Python codes. We'll be using 'with open' to read in dataset,
store data into a Python dict, and finally train the model and 
test it on kaggle. This model is the most flexible one among all
levels. You should do hyperparameter tuning and find the best model.
"""
from util import increment, dotProduct
import math
TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename: str, data: dict, mode='Train', training_data=None):
    """
    :param filename: str, the filename to be processed
    :param data: an empty Python dictionary
    :param mode: str, indicating the mode we are using
    :param training_data: dict[str: list], key is the column name, value is its data
                          (You will only use this when mode == 'Test')
    :return data: dict[str: list], key is the column name, value is its data
    """
    with open(filename, 'r') as f:
        first_line = True  # use to separate 1st line from other lines
        for line in f:
            line = line.strip()
            data_list = line.split(',')
            if mode == 'Train':
                for i in range(len(data_list)):
                    if first_line:
                        # put the title of the column as data's key
                        if i == 1 or i == 2 or i == 4 or i == 5 or i == 6 or i == 7 or i == 9 or i == 11:
                            data[data_list[i]] = []
                # for other lines, put values into their columns accordingly
                if first_line is False:
                    # if value in Age or Embarked is empty
                    if data_list[6] == '' or data_list[12] == '':
                        # delete that line's data
                        data_list.clear()
                    else:
                        # put data into the columns accordingly
                        for i in range(len(data_list)):
                            if i == 1:
                                data['Survived'].append(int(data_list[i]))
                            elif i == 2:
                                data['Pclass'].append(int(data_list[i]))
                            elif i == 5:
                                if data_list[i] == 'male':
                                    data['Sex'].append(1)
                                else:
                                    data['Sex'].append(0)
                            elif i == 6:
                                data['Age'].append(float(data_list[i]))
                            elif i == 7:
                                data['SibSp'].append(int(data_list[i]))
                            elif i == 8:
                                data['Parch'].append(int(data_list[i]))
                            elif i == 10:
                                data['Fare'].append(float(data_list[i]))
                            elif i == 12:
                                if data_list[i] == 'S':
                                    data['Embarked'].append(0)
                                elif data_list[i] == 'C':
                                    data['Embarked'].append(1)
                                elif data_list[i] == 'Q':
                                    data['Embarked'].append(2)
                first_line = False
            else:
                for i in range(len(data_list)):
                    if first_line:
                        # put the title of the column as data's key
                        if i == 1 or i == 3 or i == 4 or i == 5 or i == 6 or i == 8 or i == 10:
                            data[data_list[i]] = []
                # for other lines, put values into their columns accordingly
                if first_line is False:
                    # put data into the columns accordingly
                    for i in range(len(data_list)):
                        if i == 1:
                            data['Pclass'].append(int(data_list[i]))
                        elif i == 4:
                            if data_list[i] == 'male':
                                data['Sex'].append(1)
                            else:
                                data['Sex'].append(0)
                        elif i == 5:
                            if data_list[i] == '':
                                age_mean = sum([num for num in training_data['Age']]) / len(training_data['Age'])
                                data['Age'].append(round(age_mean, 3))
                            else:
                                data['Age'].append(float(data_list[i]))
                        elif i == 6:
                            data['SibSp'].append(int(data_list[i]))
                        elif i == 7:
                            data['Parch'].append(int(data_list[i]))
                        elif i == 9:
                            if data_list[i] == '':
                                fare_mean = sum([num for num in training_data['Fare']]) / len(training_data['Fare'])
                                data['Fare'].append(round(fare_mean, 3))
                            else:
                                data['Fare'].append(float(data_list[i]))
                        elif i == 11:
                            if data_list[i] == 'S':
                                data['Embarked'].append(0)
                            elif data_list[i] == 'C':
                                data['Embarked'].append(1)
                            elif data_list[i] == 'Q':
                                data['Embarked'].append(2)
                first_line = False
    return data


def one_hot_encoding(data: dict, feature: str):
    """
    :param data: dict[str, list], key is the column name, value is its data
    :param feature: str, the column name of interest
    :return data: dict[str, list], remove the feature column and add its one-hot encoding features
    """
    if feature == 'Sex':
        data['Sex_1'] = []
        data['Sex_0'] = []
        for i in range(len(data[feature])):
            if data[feature][i] == 1:  # male
                data['Sex_1'].append(1)  # male
                data['Sex_0'].append(0)
            else:  # female
                data['Sex_1'].append(0)
                data['Sex_0'].append(1)
        data.pop('Sex')
    elif feature == 'Pclass':
        data['Pclass_0'] = []  # Pclass1
        data['Pclass_1'] = []  # Pclass2
        data['Pclass_2'] = []  # Pclass3
        for i in range(len(data[feature])):
            if data[feature][i] == 1:
                data['Pclass_0'].append(1)  # Pclass1
                data['Pclass_1'].append(0)  # Pclass2
                data['Pclass_2'].append(0)  # Pclass3
            elif data[feature][i] == 2:
                data['Pclass_0'].append(0)  # Pclass1
                data['Pclass_1'].append(1)  # Pclass2
                data['Pclass_2'].append(0)  # Pclass3
            else:
                data['Pclass_0'].append(0)  # Pclass1
                data['Pclass_1'].append(0)  # Pclass2
                data['Pclass_2'].append(1)  # Pclass3
        data.pop('Pclass')
    elif feature == 'Embarked':
        data['Embarked_0'] = []  # S
        data['Embarked_1'] = []  # C
        data['Embarked_2'] = []  # Q
        for i in range(len(data[feature])):
            if data[feature][i] == 0:  # S
                data['Embarked_0'].append(1)  # S
                data['Embarked_1'].append(0)  # C
                data['Embarked_2'].append(0)  # Q
            elif data[feature][i] == 1:  # C
                data['Embarked_0'].append(0)  # S
                data['Embarked_1'].append(1)  # C
                data['Embarked_2'].append(0)  # Q
            else:  # Q
                data['Embarked_0'].append(0)  # S
                data['Embarked_1'].append(0)  # C
                data['Embarked_2'].append(1)  # Q
        data.pop('Embarked')
    return data


def normalize(data: dict):
    """
    :param data: dict[str, list], key is the column name, value is its data
    :return data: dict[str, list], key is the column name, value is its normalized data
    """
    for key in data:
        max_num = max([num for num in data[key]])
        min_num = min([num for num in data[key]])
        for i in range(len(data[key])):
            data[key][i] = (data[key][i] - min_num) / (max_num - min_num)
    return data


def learnPredictor(inputs: dict, labels: list, degree: int, num_epochs: int, alpha: float):
    """
    :param inputs: dict[str, list], key is the column name, value is its data
    :param labels: list[int], indicating the true label for each data
    :param degree: int, degree of polynomial features
    :param num_epochs: int, the number of epochs for training
    :param alpha: float, known as step size or learning rate
    :return weights: dict[str, float], feature name and its weight
    """
    # Step 1 : Initialize weights
    weights = {}  # feature => weight
    keys = list(inputs.keys())
    if degree == 1:
        for i in range(len(keys)):
            weights[keys[i]] = 0
    elif degree == 2:
        for i in range(len(keys)):
            weights[keys[i]] = 0  # N
        for i in range(len(keys)):
            for j in range(i, len(keys)):
                weights[keys[i] + keys[j]] = 0  # 各feature的排列組合
    # Step 2 : Start training

    # Step 3 : Feature Extract
    for epoch in range(num_epochs):
        feature = {}
        for i in range(len(labels)):
            if degree == 1:
                for j in range(len(keys)):
                    feature[keys[j]] = inputs[keys[j]][i]
            elif degree == 2:
                for j in range(len(keys)):
                    feature[keys[j]] = inputs[keys[j]][i]
                for j in range(len(keys)):
                    for k in range(j, len(keys)):
                        feature[keys[j] + keys[k]] = inputs[keys[j]][i] * inputs[keys[k]][i]

    # Step 4 : Update weights
            k = dotProduct(feature, weights)
            h = 1 / (1 + math.exp(-k))
            scale = - alpha * (h - labels[i])
            increment(weights, scale, feature)

    return weights

