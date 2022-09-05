import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Code to see all the columns in output
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',15)


housing_data = pd.read_csv("housing_data.csv")
#print(housing_data.head())
#print("\n")

#print(housing_data.info)
#print(housing_data.columns)
#print("\n")

print(housing_data.describe())
print("\n")

housing_data.hist(bins=70, figsize=(25,20))
#plt.show()

## Train-Test Splitting
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data)* test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing_data, 0.2)
print(f"No. of rows in train set: {len(train_set)} \nNo. of rows in test set: {len(test_set)}\n")

## Train-Test Splitting can easily be done using sklearn library so above func(def split_train_test()) will not be required
from sklearn.model_selection import train_test_split
train_set, test_set= train_test_split(housing_data, test_size=0.2, random_state=42)
#print(test_set["River Neighbouring"].value_counts())
print(f"No. of rows in train set: {len(train_set)} \nNo. of rows in test set: {len(test_set)}\n")

## River Neighbouring Houses are very less so it could be possible that our model's training data set might not get even 1 record of River Neighbouring house.
## River Neighbourinh house could be a major factor for prediction of House Prices, Hence we cannot skip it in our training set
## Hence we will be writing following code to deal with such situation so that random selection should have few records of River Neighbouring houses.

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing_data, housing_data["River Neighbouring"]):
    strat_train_set = housing_data.loc[train_index]
    strat_test_set = housing_data.loc[test_index]

print(strat_train_set['River Neighbouring'].value_counts())
print("\n")
print(strat_test_set['River Neighbouring'].value_counts())
print("\n")

# Update our housing_data dataset with the trained dataset i.e. strat_train_set
housing_data = strat_train_set.copy()

##Looking for Correlations
#corr_matrix = housing_data.corr()
#print(corr_matrix["Median Price of Houses"].sort_values(ascending=False))

from pandas.plotting import scatter_matrix
attributes = ["Median Price of Houses", "Average no. of rooms", "Residential Land", "Lower Status of Population"]
scatter_matrix(housing_data[attributes], figsize=(12,8))
#plt.show()
housing_data.plot(kind="scatter", x="Average no. of rooms", y="Median Price of Houses", alpha=0.8)
#plt.show()

##Trying out attribute combinations
housing_data["Tax Per Room"] = housing_data["Tax Rate"]/housing_data["Average no. of rooms"]
#print(housing_data["Tax Per Room"])
#print("\n")

##Looking for Correlations
corr_matrix = housing_data.corr()
print(corr_matrix["Median Price of Houses"].sort_values(ascending=False))
print("\n")

## Missing Attributes
# To take of missing attributes, you have 3 options
#   1. Get rid of missing data points
#   2. Get rid of entire attribute/column
#   3. Set the empty column's values to some arbitrary values such as - 0/Mean/Median/Mode

a = housing_data.dropna(subset=["Average no. of rooms"]) # Option 1
print(a.shape)

print(housing_data.drop("Average no. of rooms", axis=1).shape) # Option 2

median = housing_data["Average no. of rooms"].median()
housing_data["Average no. of rooms"].fillna(median) # option 3
print(housing_data.shape)
print(("\n"))

# Another method perform option 3 using sklear.impute to fill missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit (housing_data)
print(housing_data.shape)
print(("\n"))
print(imputer.statistics_) # will give median value for all the columns
X = imputer.transform(housing_data)
print(("\n"))
housing_data_transformed = pd.DataFrame(X, columns=housing_data.columns)
print(housing_data_transformed.describe())
print(("\n"))


## Scikit-learn Design
#Primarily 3 types of objects in Scikit Learn
#   1. Estimators - Estimates some parameters based on a dataset eg- Imputer.
#                   It has a fit method & transform method. Fits the dataset and calculates internal parameters.
#   2. Transformers - Transform method takes input and returs output based on the learnings from fit().
#                     It also has a convenience function called fit_transform() which fits & then transforms.
#   3. Predictors - fit() & predict() are 2 common functions. It also gives score() function which will evaluate the predictions.
#                   LinearRegression model is an example of predictor.


## Features Scaling
# 2 Types of Features Scaling methods primarily
#   1. Normalization (Min-Max Scaling) = (Value-min)/(max-min)
#   Sklearn provides a class called MinMaxScaler for this
#   2. Standardization = (Value-mean)/std
#   Sklearn provides a class called StandardScaler for this



## Creating a Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    #     ........ add as many as you want in your pipeline
    ('std_scaler', StandardScaler())
])

housing_data_transformed_pipeline = my_pipeline.fit_transform(housing_data_transformed)
print(housing_data_transformed_pipeline)


