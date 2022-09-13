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

housing_data = strat_train_set.drop("Median Price of Houses", axis=1)
housing_data_labels = strat_train_set["Median Price of Houses"].copy()

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
print(housing_data_transformed_pipeline.shape)

# Pickling the scaler output
import pickle
pickle.dump(housing_data_transformed_pipeline,open('scaling.pkl','wb'))

## Selecting a desired model for Dragon Real Estate:-
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(housing_data_transformed_pipeline, housing_data_labels)

some_data = housing_data.iloc[:5]
some_labels = housing_data_labels.iloc[:5]

prepared_data = my_pipeline.transform(some_data)
print(model.predict(prepared_data))
print(list(some_labels))


## Evaluating the model
from sklearn.metrics import mean_squared_error
housing_data_predictions = model.predict(housing_data_transformed_pipeline)
lin_mse = mean_squared_error(housing_data_labels, housing_data_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_mse, " ", lin_rmse)
# Value of Mean square error is 23, which is very high, hence discarding this error & trying DecisionTreeRegressor above by commenting LinearRegression model


## Selecting Decesion Tree Regressor model
from sklearn.tree import DecisionTreeRegressor
model1 = DecisionTreeRegressor()
model1.fit(housing_data_transformed_pipeline, housing_data_labels)

housing_data_predictions = model1.predict(housing_data_transformed_pipeline)
mse = mean_squared_error(housing_data_labels, housing_data_predictions)
rmse = np.sqrt(mse)
print(mse, " ", rmse)
# Hence Mean Square Error is coming as 0, which means that it has overfitted the model, which is also not good


## Using better evaluation technique - Cross Validation
from sklearn.model_selection import cross_val_score

# For Linear Regression Model:-
scores = cross_val_score(model, housing_data_transformed_pipeline, housing_data_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)
#print("Scores:" , rmse_scores)
print("Mean:", rmse_scores.mean())
print("Standard Deviation:", rmse_scores.std())
print("\n")

# For Decision Tree Regressor
scores1 = cross_val_score(model1, housing_data_transformed_pipeline, housing_data_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores1 = np.sqrt(-scores1)
#print("Scores:" , rmse_scores1)
print("Mean:", rmse_scores1.mean())
print("Standard Deviation:", rmse_scores1.std())
print("\n")
# Decision Tree Regressor model's error values(score) is slightly better/lesser than Linear Regression model's
# We now want to check whether another model will give better result or not, hence trying Random Forest Regressor now

# For Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
model2 = RandomForestRegressor()
model2.fit(housing_data_transformed_pipeline, housing_data_labels)
scores2 = cross_val_score(model2, housing_data_transformed_pipeline, housing_data_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores2 = np.sqrt(-scores2)
#print("Scores:" , rmse_scores2)
print("Mean:", rmse_scores2.mean())
print("Standard Deviation:", rmse_scores2.std())
# Random Forest Regressor's scores are even better than Decesion Tree Regressor's model


## Saving the model
from joblib import dump, load
dump(model2, 'Dragon.joblib')


## Testing the model
X_test = strat_test_set.drop("Median Price of Houses", axis=1)
Y_test = strat_test_set["Median Price of Houses"].copy()
X_test_prepared = my_pipeline.transform(X_test)
final_predictions = model2.predict(X_test_prepared)
final_mse = mean_squared_error(Y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print("Final Predictions:", final_predictions , "\n\n", final_rmse)


## Pickling the model file for Deployment
import pickle

pickle.dump(final_predictions,open('housing_price_prediction_model.pkl','wb'))
pickled_model = pickle.load(open('housing_price_prediction_model.pkl','rb'))
#print(pickled_model.predict(X_test_prepared))







