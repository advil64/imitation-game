import os
import json

# Name of directory for grids
directory_name = os.path.dirname(__file__) + '/../data/small_grids/'
# List that holds all the grids to be used as inputs for training
x_train_grid = []
# List that holds all the positions to be used as inputs for training
x_train_position = []
# List that holds all the directions to be used as outputs for training
y_train = []

# iterate through all grids
for file_name in os.listdir(directory_name):
    f = open(directory_name + file_name)
    data = json.load(f)

    # Iterate through all the data in a given grid and append their input and output values
    for i in data:
        x_train_grid.append(i['gridworld'])
        x_train_position.append(i['position'])
        y_train.append(i['direction'])
print(y_train)

