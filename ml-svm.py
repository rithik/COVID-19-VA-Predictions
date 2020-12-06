
import numpy as np
np.random.seed(42)
import random
import sklearn
from sklearn.svm import SVC
from sklearn import preprocessing
import pandas as pd
# Dataset information
# the column names (names of the features) in the data files
# you can use this information to preprocess the features

col_names_x = ['fips', 'unemployment-percent', 'median-household-income', 'num-days', 
            'age0-14', 'age15-24', 'age25-34', 'age35-44', 'age45-54', 'age55-64', 'age65-85', 'white', 'black',
            'american-indian', 'asian', 'total', 'c2', 'c3',  'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10']

col_names_y = ['greaterC']

column_categories_counts = {}
# 1. Data loading from file and pre-processing.
# Hint: Feel free to use some existing libraries for easier data pre-processing. 
# For example, as a start you can use one hot encoding for the categorical variables and normalization 
# for the continuous variables. Also, look out for missing values. 
def load_data(csv_file_path):
    # your code here
    x = np.array([])
    # y = np.array([])
    for col in range(0,len(col_names_x)):
        colnp = pd.read_csv(csv_file_path, usecols = [col_names_x[col]]).to_numpy()
        transformed = preprocessing.StandardScaler().fit_transform(colnp)
        if x.shape[0] == 0: 
            x = transformed
        else:
            x = np.append(x, transformed, axis = 1)

    y = pd.read_csv(csv_file_path, usecols = ['greaterC']).to_numpy()
    y = y.flatten()
    return x, y

# def fold(x, y, i, nfolds):
#     # your code
#     n = int(y.shape[0])
#     size = n/nfolds
#     x_test = x[int(size * i): int((size*i) + size)]
#     x_train_one = x[0:int(size * i)]
#     x_train_two = x[int((size*i) + size):n]
#     x_train = np.concatenate((x_train_one, x_train_two))

#     y_test = y[int(size * i): int((size*i) + size)]
#     y_train_one = y[0:int(size * i)]
#     y_train_two = y[int((size*i) + size):n]
#     y_train = np.concatenate((y_train_one, y_train_two))
#     # print(x_train.shape)
#     # print(x_test.shape)
#     return x_train, y_train, x_test, y_test


# 2. Select best hyperparameter with cross validation and train model.
# Attention: Write your own hyper-parameter candidates.
def train_and_select_model(x_train, y_train):
    # load data and preprocess from filename training_csv
    # x_train_main, y_train_main = load_data(training_csv)
    x_train_main, y_train_main = x_train, y_train
    best_score = 0
    best_model = None
    # hard code hyperparameter configurations, an example:
    param_set = [
                 {'kernel': 'rbf', 'C': 1, 'degree': 3}
                #  {'kernel': 'rbf', 'C': 1, 'degree': 3, 'gamma':'auto'},
                #  {'kernel': 'linear', 'C': 1, 'degree': 3},
                #  {'kernel': 'poly', 'C': 1, 'degree': 1},
                #  {'kernel': 'poly', 'C': 1, 'degree': 3},
                #  {'kernel': 'poly', 'C': 1, 'degree': 5},
                #  {'kernel': 'poly', 'C': 1, 'degree': 7},
                #  {'kernel': 'poly', 'C': 1, 'degree': 9},
                #  {'kernel': 'poly', 'C': 1, 'degree': 11},
                #  {'kernel': 'poly', 'C': 1, 'degree': 13},
                #  {'kernel': 'sigmoid', 'C': 1, 'degree': 3},
                #  {'kernel': 'sigmoid', 'C': 1, 'degree': 1, 'gamma':'auto'}
    ]
    # your code here
    # iterate over all hyperparameter configurations
    for param in param_set:
        kernel = param['kernel']
        c = param['C']
        degree = param['degree']
        if 'gamma' in param:
            gamma = param['gamma']
        else: 
            gamma = 'scale'
        # shuffle +  split 
        x, y = sklearn.utils.shuffle(x_train_main, y_train_main)
        train_test_split = int(x.shape[0] * 0.9)
        x_train = x[:train_test_split]
        y_train = y[:train_test_split]
        x_test = x[train_test_split:]
        y_test = y[train_test_split:]

        clf = SVC(kernel = kernel,C = c, degree = degree, gamma = gamma)
        clf.fit(x_train, y_train)
        test_accuracy = clf.score(x_test, y_test)
        
        if(test_accuracy > best_score):
            best_model = param 
            best_score = test_accuracy

    # select best hyperparameter from cv scores, retrain model 
    print(best_model, best_score)
    best_model = SVC(kernel = best_model['kernel'], C = best_model['C'], degree = best_model['degree']).fit(x_train_main, y_train_main)
    return best_model, best_score

# predict for data in filename test_csv using trained model
def predict(x_test, y_test, trained_model):
    # x_test, y_test = load_data(test_csv)
    print(trained_model.score(x_test, y_test))
    predictions = trained_model.predict(x_test)
    return predictions

# save predictions on test data in desired format 
def output_results(predictions):
    with open('predictions.txt', 'w') as f:
        for pred in predictions:
            f.write(pred)

if __name__ == '__main__':
    dataset = "datasetbinary.csv"
    
    all_x, all_y = load_data(dataset)
    shuffled_x, shuffled_y = sklearn.utils.shuffle(all_x, all_y)
    train_test_split = int(shuffled_x.shape[0] * 0.9)
    x_train = shuffled_x[:train_test_split]
    y_train = shuffled_y[:train_test_split]
    x_test = shuffled_x[train_test_split:]
    y_test = shuffled_y[train_test_split:]
    # fill in train_and_select_model(training_csv) to 
    # return a trained model with best hyperparameter from 3-FOLD 
    # cross validation to select hyperparameters as well as cross validation score for best hyperparameter. 
    # hardcode hyperparameter configurations as part of train_and_select_model(training_csv)
    trained_model, cv_score = train_and_select_model(x_train, y_train)

    print("The best model was scored : ",cv_score)
    # use trained SVC model to generate predictions
    predictions = predict(x_test, y_test, trained_model)
    print(predictions)
    # Don't archive the files or change the file names for the automated grading.
    # Do not shuffle the test dataset
    # output_results(predictions)
    # 3. Upload your Python code, the predictions.txt as well as a report to Collab.
    df = pd.DataFrame({"test" : y_test, "pred" : predictions})
    df.to_csv("submission.csv", index=False)
