#So we'll begin by loading the dataset
import pandas as pd

# **************************************************************
# Path of the file to read
# This is an example file path
iowa_file_path = '../input/home-data-for-ml-course/train.csv'
# home_data is a path that we can save the path to:
home_data = pd.read_csv(iowa_file_path)

# **************************************************************
# Set up code checking
# I don't know what this actually means so I'll just have it in for now
from learntools.core import binder
binder.bind(globals())
from learntools.machine_learning.ex3 import *

# print the list of columns in the dataset to find the name of the prediction target
# The columns are features of the data
home_data.columns

# **************************************************************
# Why I picked the SalesPrice as the Y value is because this is the outcome i want to predict
# I can select the value of y based on the options given to me when I analyze the columns
y = home_data.SalePrice


# **************************************************************
# Create the list of features that you want to use as determinants below
feature_names = ["LotArea",
"YearBuilt",
"1stFlrSF",
"2ndFlrSF",
"FullBath",
"BedroomAbvGr",
"TotRmsAbvGrd"]

# Select data corresponding to features in feature_names
X = home_data[feature_names]

# **************************************************************
# Review data
# print description or statistics from X
print(X.describe())

# print the top few lines
print(X.head())

# **************************************************************
# You probably want this information at the top of the doc
# Also the DecisionTreeRegressor is a type of model that we can use in the sklearn library
from sklearn.tree import DecisionTreeRegressor

#specify the model. 
#For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state=1)
# Fit the model
iowa_model.fit(X, y)

# **************************************************************
# Make predictions using the predict function
print("The predictions are")
predictions = iowa_model.predict(X)
print(predictions)

# **************************************************************
# Test the predictions
print("Making predictions for the following 5 houses:")
print(y.head())
print("The predictions are")
print(iowa_model.predict(X.head()))

# Model Validation
# *****************************************************************
# Import the train_test_split function from sklearn
from sklearn.model_selection import train_test_split
# Split the training and test data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)

# *****************************************************************
# Specify the model
# Always input the random state when creating the model
iowa_model = DecisionTreeRegressor(random_state = 1)
# Fit iowa_model with the training data.
iowa_model.fit(train_X,train_y)

# *****************************************************************
# Make predictions
# Predict with all validation observations
val_predictions = iowa_model.predict(val_X)
# *****************************************************************
# Validating the data

# *****************************************************************
from sklearn.metrics import mean_absolute_error

val_predictions = iowa_model.predict(val_X)
val_mae = mean_absolute_error(val_y, val_predictions)

print(val_mae)
# *****************************************************************
# Here's a function that I can use iteratively to calculate the error
def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)


# *****************************************************************
# I can use this code to predict the optimum value for a decision tree
# I can find the no. of leaf nodes that generates the smallest error

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
# Write loop to find the ideal tree size from candidate_max_leaf_nodes
leaf_size = []
for i in candidate_max_leaf_nodes:
    leaf_size.append(get_mae(i, train_X, val_X, train_y, val_y))
           
# Store the best value of max_leaf_nodes (it will be either 5, 25, 50, 100, 250 or 500)
position = leaf_size.index(min(leaf_size))
best_tree_size = candidate_max_leaf_nodes[position]

# *****************************************************************
# When you're done, fit the data
# Fill in argument to make optimal size and uncomment
final_model = DecisionTreeRegressor(max_leaf_nodes = best_tree_size, random_state=1)
# fit the final model and uncomment the next two lines
final_model.fit(X,y)

# *****************************************************************
# if we want better accuracy we can use a random forest generator instead of a decision tree
# so we need the appropriate library
from sklearn.ensemble import RandomForestRegressor

# Define the model. Set random_state to 1
rf_model = RandomForestRegressor(random_state=1)

# fit your model
rf_model.fit(train_X, train_y)
iowa_pred = rf_model.predict(val_X)

# Calculate the mean absolute error of your Random Forest model on the validation data
rf_val_mae = mean_absolute_error(val_y, iowa_pred)

print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))


# Here's the code for the final solution: Making predictions and all
# One key thing I cemented is: splitting my data properly and ensuring that I'm testing on my validation data

# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]
test_model = RandomForestRegressor(random_state = 1)
# Split the data correctly
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
# Train the model on the cross_validation set
test_model.fit(train_X, train_y)

# make predictions which we will submit. 
test_preds = test_model.predict(test_X)

# The lines below shows how to save predictions in format used for competition scoring
# Just uncomment them.

output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})
output.to_csv('submission.csv', index=False)
