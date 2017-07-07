import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix



pd.set_option('max_columns', 999)
pd.set_option('display.width', 1000)

# STEP-1 : READING THE DATA FROM A CSV FILE TO A PANDAS DATAFRAME, DIVIDING IT INTO TRAINING AND TEST SAMPLES AND PREPROCESSING THE TRAINING DATASET
column_list = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount',
               'Duration', 'Purpose', 'Class']
new_column_list = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Credit amount',
               'Duration', 'Purpose', 'Class']
categorical_col_list=['Sex','Housing','Purpose']


# file_name = inputfile  # filename is argument 1

inputfile = "Proj_dataset_1.csv"
with open(inputfile, 'rU') as f:  # opens PW fi   le
    reader = csv.reader(f)
    data = list(list(rec) for rec in csv.reader(f, delimiter='\t'))

data=data[1:]
utility_matrix1 = pd.read_csv(inputfile)

# Feature Scaling - Subsituting the values for 'Checking Account and Saving Account' column as [little,moderate,rich,quite rich] -> [1,2,3,4]
utility_matrix1 = utility_matrix1.replace('little', int(0))
utility_matrix1 = utility_matrix1.replace('moderate', int(1))
utility_matrix1 = utility_matrix1.replace('rich', int(2))
utility_matrix1 = utility_matrix1.replace('quite rich', int(3))

utility_matrix1.drop(utility_matrix1.columns[[0]], axis=1, inplace=True)
# print(utility_matrix1)

# Finding the number of missing values for Checking and Saving accounts.
cheking_ac_miss_val = utility_matrix1['Checking account'].value_counts(dropna=False)
saving_ac_miss_val = utility_matrix1['Saving accounts'].value_counts(dropna=False)
# print(cheking_ac_miss_val)
# print(saving_ac_miss_val)

# Dropping the Checking Account feature as it had a lot of missing values i.e 316 which accounted for almost one-third of the dataset.
del utility_matrix1['Checking account']


# utility_matrix1, test_data_matrix = train_test_split(utility_matrix1, test_size=0.2)

train_data = utility_matrix1.values.tolist()
# print(train_data)

# test_data = test_data_matrix.values.tolist()
# utility_matrix1.to_csv('training.csv')
# test_data_matrix.to_csv('testing.csv')


# By using the class DaatFrameImputer function from "http://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn"
# I modified according to my dataset to impute the missing values.
class DataFrameImputer(TransformerMixin):
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

X = pd.DataFrame(train_data)
imputed_train_data = DataFrameImputer().fit_transform(X)
imputed_train_data.columns=new_column_list
# print(imputed_train_data)


imputed_train_data_with_dummies = pd.get_dummies(imputed_train_data, columns = categorical_col_list )
print(imputed_train_data_with_dummies)


imputed_train_data_with_dummies_list = imputed_train_data_with_dummies.values.tolist()
# print(imputed_train_data_with_dummies_list)


# STEP-2 : PERFORMING FEATURE SELECTION VIA PCA

X_std = StandardScaler().fit_transform(imputed_train_data_with_dummies)
# print(X_std)
# print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))

# Eigen decomposition of the standardized data based on the correlation matrix:
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
# print('Eigenvectors \n%s' %eig_vecs)
# print('\nEigenvalues \n%s' %eig_vals)

# Performing SVD
u,s,v = np.linalg.svd(X_std.T)
# print("svd")
# print(u)

# Comparing the results of eigen decomposition of covariance and correlation matrix with SVD we find that it is the same.


# Now finding the number of principal components for PCA.

eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]


# Sort the (eigenvalue, eigenvector) tuples from high to low
# eig_pairs.sort()
# eig_pairs.reverse()

# print('Eigenvalues corresponding to the feature columns are:')
# for i in eig_pairs:
    # print(i[0])


# How to select the number of principal components
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
# print(var_exp)
# print(cum_var_exp)


# Observing the eigen values we can remove the following features like Saving accounts, Purpose_radio/TV  Purpose_repairs  Purpose_vacation/others
# as their eigen values are the least of all and won't contribute much towards the classification.




# Creating the target variable
target= imputed_train_data_with_dummies.as_matrix(columns=['Class'])
# print(target)



imputed_train_data_with_dummies.drop('Class',axis=1,inplace=True)
imputed_train_data_with_dummies.drop('Saving accounts',axis=1,inplace=True)
imputed_train_data_with_dummies.drop('Purpose_radio/TV',axis=1,inplace=True)
imputed_train_data_with_dummies.drop('Purpose_repairs',axis=1,inplace=True)
imputed_train_data_with_dummies.drop('Purpose_vacation/others',axis=1,inplace=True)

# Converting the data as numpy array
data=np.array(imputed_train_data_with_dummies)
# print(data)

# print(data.shape)
# print(target.shape)


# # Splitting the data into training and test sets.
# data_train, data_test, target_train, target_test = train_test_split(data,target,test_size=0.2)
# print(data_train)
#
# np.savetxt("data_train.txt",data_train)
# np.savetxt("data_test.txt", data_test)
# np.savetxt("target_train.txt", target_train)
# np.savetxt("target_test.txt", target_test)


data_train = np.loadtxt("data_train.txt")
data_test = np.loadtxt("data_test.txt")
target_train = np.loadtxt("target_train.txt")
target_test = np.loadtxt("target_test.txt")



print(data_train.shape)
print(target_test.shape)

c, r = target.shape
target = target.reshape(c,)



# Performing Logistic Regression by using the inbuilt functions from sklearn library. I took the function from
# "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html" and modified it according to my needs.
print("Logistic Regression")
logreg =LogisticRegression()
logreg.fit(data_train,target_train)
y_pred_logreg=logreg.predict(data_test)
target_names=['class 1', 'class 2']

scores_cv = cross_val_score(logreg, data, target, cv=5)
print("Cross_validation_scores = ", scores_cv)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_cv.mean(), scores_cv.std() * 2))
print(accuracy_score(target_test, y_pred_logreg))
print(classification_report(target_test, y_pred_logreg, target_names=target_names))

cf_matrix_logreg = confusion_matrix(target_test, y_pred_logreg)
# print(cf_matrix)
# Show confusion matrix in a separate window
plt.matshow(cf_matrix_logreg)
plt.title('Confusion matrix_logreg')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


#########################################################################

# Performing the Ada Boosting Classifier by using "http://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html"
# and make necessary modifications to it and moulded it as per my needs.
print("Ada Boosting")
abc=AdaBoostClassifier(n_estimators=100)
abc.fit(data_train,target_train)
y_pred_abc=abc.predict(data_test)

scores_cv = cross_val_score(abc, data, target, cv=5)
print("Cross_validation_scores = ", scores_cv)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_cv.mean(), scores_cv.std() * 2))
print(accuracy_score(target_test, y_pred_abc))
print(classification_report(target_test, y_pred_abc, target_names=target_names))

##############################################################################

# Performing Naive Bayes Classifier by using "http://scikit-learn.org/stable/modules/naive_bayes.html" and using it in my code to
# check the performance of Bayes MinimumClassifier.
print("Bayes Minimum Classifier")
gnb=GaussianNB()
gnb.fit(data_train,target_train)
y_pred_gnb=gnb.predict(data_test)

scores_cv = cross_val_score(gnb, data, target, cv=5)
print("Cross_validation_scores = ", scores_cv)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_cv.mean(), scores_cv.std() * 2))
print(accuracy_score(target_test, y_pred_gnb))
print(classification_report(target_test, y_pred_gnb, target_names=target_names))
cf_matrix_bayes = confusion_matrix(target_test, y_pred_gnb)
# print(cf_matrix)
# Show confusion matrix in a separate window
plt.matshow(cf_matrix_bayes)
plt.title('Confusion matrix_bayes')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


##############################################################################

# Performing K-nearest neighbours with different values of n_neighbors by using inbuilt KNeighborClassifier() from sklearn and deploying
# it according to the needs of my project.
print("KNN")
knn_acc_list=[]
for neighbor in range(3,12):
    knn=KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(data_train,target_train)
    y_pred_knn=knn.predict(data_test)
    target_names=['class 1', 'class 2']
    knn_acc_list.append(accuracy_score(target_test, y_pred_knn))
    print(classification_report(target_test, y_pred_knn, target_names=target_names))
print(knn_acc_list)
scores_cv = cross_val_score(knn, data, target, cv=5)
print("Cross_validation_scores = ", scores_cv)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_cv.mean(), scores_cv.std() * 2))
cf_matrix_knn = confusion_matrix(target_test, y_pred_gnb)
# print(cf_matrix)
# Show confusion matrix in a separate window
plt.matshow(cf_matrix_knn)
plt.title('Confusion matrix_knn')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()


#######################################################################################

# Perfroming Random Forest technique by using the inbuilt function from "http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"
# and modified it accordingly to our dataset.
print("Random Forest")
rnd_fc=RandomForestClassifier(n_estimators=50,criterion='entropy')
rnd_fc.fit(data_train,target_train)
y_pred_rnd_fc=rnd_fc.predict(data_test)

scores_cv = cross_val_score(rnd_fc, data, target, cv=5)
print("Cross_validation_scores = ", scores_cv)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores_cv.mean(), scores_cv.std() * 2))
print(accuracy_score(target_test, y_pred_rnd_fc))
print(classification_report(target_test, y_pred_rnd_fc, target_names=target_names))

cf_matrix_rndf = confusion_matrix(target_test, y_pred_gnb)
# print(cf_matrix)
# Show confusion matrix in a separate window
plt.matshow(cf_matrix_rndf)
plt.title('Confusion matrix_rndf')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

##############################################################################