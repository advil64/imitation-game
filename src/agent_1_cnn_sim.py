import tensorflow as tf
import numpy as np
from gridworld import Gridworld
from time import sleep, time
import random

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class Agent_1:

  def __init__(self, dim):
    self.dim = dim
    self.discovered_grid = Gridworld(dim)
    self.cg = [[0] * dim for i in range(dim)]
    self.neural_network = tf.keras.models.load_model('/common/users/ac1771/imitation-game/models/agent1_CNN')

  def execute_path(self, complete_grid, timeout_sec):
    starting_time = time()
    time_elapsed = time() - starting_time
    total_time_elapsed = time() - starting_time
    retries = 0
    trajectory_length = 0
    curr = (0,0)
    prev = curr
    random_rounds = 0
    while curr != (self.dim-1, self.dim-1):
      time_elapsed = time() - starting_time
      total_time_elapsed = time() - starting_time
      print("Currently in: (%s, %s)" % (curr[0], curr[1]))
      trajectory_length += 1

      self.update_neighbor_obstacles(curr, complete_grid)
      self.cg[curr[0]][curr[1]] += 1

      in_grid = np.reshape(self.discovered_grid.gridworld, (1, 50, 50)) / 2
      locals_val = self.get_local(self.discovered_grid.gridworld, curr)
      in_local = np.reshape(locals_val, (1, 5, 5))
      print(in_local)
      in_position = np.reshape(self.get_position(curr), (1, 50, 50))
      prob_predict = self.neural_network.predict( [in_grid, in_local, in_position] )
      prediction = np.argmax( prob_predict, axis = 1 )

      print("Taking direction: %s" % prediction[0])

      direction = self.get_direction(prediction[0])
      new_position = (curr[0] + direction[0], curr[1] + direction[1])

      if new_position[0] < 0 or new_position[0] >= self.dim or new_position[1] < 0 or new_position[1] >= self.dim:
        # print("NN Agent 1 Failed: Went out of boundary")
        # return False, trajectory_length, retries
        retries += 1
        self.cg[curr[0]][curr[1]] += 1
      elif complete_grid.gridworld[new_position[0]][new_position[1]] == 1:
        retries += 1
        # if self.discovered_grid.gridworld[new_position[0]][new_position[1]] == 1:
        #   print("NN Agent 1 Failed: Attempted to enter a known blocked cell")
        #   return False, trajectory_length, retries
        self.cg[curr[0]][curr[1]] += 1
        # update our knowledge of blocked nodes
        self.discovered_grid.update_grid_obstacle(new_position, 1)
      else:
        curr = new_position

      # throw an error if we've been in a deadend for two minutes
      if total_time_elapsed > 60:
        raise TimeoutError

      # if we've been in the same place for too long, force the algorithm to take a couple of random steps
      if time_elapsed > timeout_sec:
        print("Take 5 random step")
        random_rounds += 1
        # get options
        for i in range(5):
          options = self.get_open_neighbors(curr, complete_grid.gridworld)
          curr = random.choice(options)
      # reset time elapsed
      time_elapsed = time()

        
    
    return True, trajectory_length, retries, random_rounds
  

  def get_open_neighbors(self, position, grid):
    open = []
    # find all the neighbors for the current cell
    for n in [[-1,-1], [-1,0], [-1,1], [0,-1], [0,0], [0,1], [1,-1], [1,0], [1,1]]:
      # the cordinates of the neighbor
      curr_neighbor = (position[0] + n[0], position[1] + n[1])
      # check bounds
      if curr_neighbor[0] >= 0 and curr_neighbor[0] < self.dim and curr_neighbor[1] >= 0 and curr_neighbor[1] < self.dim and grid[curr_neighbor[0]][curr_neighbor[1]] != 1:
        # add the neighbor cell to our list
        open.append((curr_neighbor[0], curr_neighbor[1]))
    
    return open


  def get_direction(self, prediction):
    # 0 = left, 1 = up, 2 = right, 3 = down
    if prediction == 0:
      return [0, -1]
    elif prediction == 1:
      return [-1, 0]
    elif prediction == 2:
      return [0, 1]
    elif prediction == 3:
      return [1, 0]

  def get_local(self, grid, position):
    locals = []

    # find all the neighbors for the current cell
    for n in [[-2,-2], [-2,-1], [-2,0], [-2,1], [-2,2], [-1,-2], [-1,-1], [-1,0], [-1,1], [-1,2], [0,-2], [0,-1], [0,0], [0,1], [0,2], [1,-2], [1,-1], [1,0], [1,1], [1,2], [2,-2], [2,-1], [2,0], [2,1], [2,2]]:
      # the cordinates of the neighbor
      curr_neighbor = (position[0] + n[0], position[1] + n[1])
      # check bounds
      if curr_neighbor[0] >= 0 and curr_neighbor[0] < self.dim and curr_neighbor[1] >= 0 and curr_neighbor[1] < self.dim and grid[curr_neighbor[0]][curr_neighbor[1]] != 1:
        # add the neighbor cell to our list
        locals.append(self.cg[curr_neighbor[0]][curr_neighbor[1]])
      else:
        locals.append(-1)
    
    max_val = max(locals) + 4
    for i in range(len(locals)):
      if locals[i] == -1:
        locals[i] = max_val
    
    return locals

  def get_position(self, position):
    pos_grid = [[0] * self.dim for i in range(self.dim)]
    pos_grid[position[0]][position[1]] = 1
    return pos_grid
  
  # method for 4-neighbor agent
  def update_neighbor_obstacles(self, curr, complete_grid):
    # check the neighbor above the block
    if curr[0] - 1 >= 0:
      if complete_grid.gridworld[curr[0] - 1][curr[1]] == 1:
        self.discovered_grid.update_grid_obstacle((curr[0] - 1, curr[1]), 1)
      else:
        self.discovered_grid.update_grid_obstacle((curr[0] - 1, curr[1]), 0)
    # check the neighbor below the block
    if curr[0] + 1 < self.dim:
      if complete_grid.gridworld[curr[0] + 1][curr[1]] == 1:
        self.discovered_grid.update_grid_obstacle((curr[0] + 1, curr[1]), 1)
      else:
        self.discovered_grid.update_grid_obstacle((curr[0] + 1, curr[1]), 0)
    # check the neighbor left of the block
    if curr[1] - 1 >= 0:
      if complete_grid.gridworld[curr[0]][curr[1] - 1] == 1:
        self.discovered_grid.update_grid_obstacle((curr[0], curr[1] - 1), 1)
      else:
        self.discovered_grid.update_grid_obstacle((curr[0], curr[1] - 1), 0)
    # check the neighbor right of the block
    if curr[1] + 1 < self.dim:
      if complete_grid.gridworld[curr[0]][curr[1] + 1] == 1:
        self.discovered_grid.update_grid_obstacle((curr[0], curr[1] + 1), 1)
      else:
        self.discovered_grid.update_grid_obstacle((curr[0], curr[1] + 1), 0)