import os
import json

directory_name = os.path.dirname(__file__) + '/../data/small_grids/'
x_train_grid = []
x_train_position = []
y_train = []

for file_name in os.listdir(directory_name):
    f = open(directory_name + file_name)
    data = json.load(f)

    file_grid = []
    file_position = []
    file_direction = []
    for i in data:
        file_grid.append(i['gridworld'])
        file_position.append(i['position'])
        file_direction.append(i['direction'])
    x_train_grid.append(file_grid)
    x_train_position.append(file_position)
    y_train.append(file_direction)

