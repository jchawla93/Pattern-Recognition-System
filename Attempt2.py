import pandas as pd
import numpy as np
import csv
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

pd.set_option('max_columns', 999)
pd.set_option('display.width', 1000)



# STEP-1 : READING THE DATA FROM A CSV FILE TO A PANDAS DATAFRAME, DIVIDING IT INTO TRAINING AND TEST SAMPLES AND PREPROCESSING THE TRAINING DATASET
column_list = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount',
               'Duration', 'Purpose', 'Class']
categorical_col_list=['Sex','Housing','Purpose']
# continuous_col_list=['No','Age', 'Job', 'Saving accounts', 'Checking account', 'Credit amount','Duration','Class']


# file_name = inputfile  # filename is argument 1

inputfile = "/Users/jiteshchawla/Desktop/EE-559 Project/Proj_dataset_1.csv"
with open(inputfile, 'rU') as f:  # opens PW fi   le
    reader = csv.reader(f)
    data = list(list(rec) for rec in csv.reader(f, delimiter='\t'))


utility_matrix1 = pd.read_csv(inputfile)

# Feature Scaling - Subsituting the values for 'Checking Account and Saving Account' column as [little,moderate,rich,quite rich] -> [1,2,3,4]
utility_matrix1 = utility_matrix1.replace('little', int(0))
utility_matrix1 = utility_matrix1.replace('moderate', int(1))
utility_matrix1 = utility_matrix1.replace('rich', int(2))
utility_matrix1 = utility_matrix1.replace('quite rich', int(3))

utility_matrix1.drop(utility_matrix1.columns[[0,1]], axis=1, inplace=True)
print(utility_matrix1)

# utility_matrix1, test_data_matrix = train_test_split(utility_matrix1, test_size=0.2)
#
# data = utility_matrix1.values.tolist()
# print(data)

# test_data = test_data_matrix.values.tolist()
# utility_matrix1.to_csv('training.csv')
# test_data_matrix.to_csv('testing.csv')



class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

X = pd.DataFrame(data)
imputed_data = DataFrameImputer().fit_transform(X)
imputed_data.columns=column_list
print(imputed_data)


imputed_data_with_dummies = pd.get_dummies(imputed_data, columns = categorical_col_list )
# print(imputed_data_with_dummies)



# x=imputed_data_with_dummies.ix[:,0:799].values
# y=imputed_data_with_dummies.ix[:,19].values
#

imputed_data_with_dummies_list = imputed_data_with_dummies.values.tolist()
# print(imputed_data_with_dummies_list)



