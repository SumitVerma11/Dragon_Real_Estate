import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Code to see all the columns in output
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',15)


housing_data = pd.read_csv("housing_data.csv")
print(housing_data.head())
print("\n")

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
print(housing_data["Tax Per Room"])
print("\n")

##Looking for Correlations
corr_matrix = housing_data.corr()
print(corr_matrix["Median Price of Houses"].sort_values(ascending=False))