from gridworld import Gridworld
from heuristics import manhattan

class Agent_1:

  def __init__(self, dim, output):
    self.dim = dim
    self.discovered_grid = Gridworld(dim)
    self.output = output

  def execute_path(self, path, complete_grid, path_coord):
    explored = 0
    for index, node in enumerate(path):
      curr = node.curr_block
      # check if path is blocked
      if complete_grid.gridworld[curr[0]][curr[1]] == 1:
        # update our knowledge of blocked nodes
        self.discovered_grid.update_grid_obstacle(curr, 1)
        return node.parent_block, explored

      # Add step info to dataset
      if index < len(path)-1:
        self.add_to_json(self.discovered_grid, curr, self.get_direction(curr, path[index + 1].curr_block))

      self.discovered_grid.update_grid_obstacle(curr, 0)
      # update knowledge of neighbor blocks
      self.update_neighbor_obstacles(curr, complete_grid)
      explored += 1
    return path[-1], explored

  def add_to_json(self, grid, position, direction):

    # first map grid to a json compatible object
    out = {}

    # loop through the grid and convert each row to a string
    # for index,row in enumerate(grid.gridworld):
    #   out['row_{}'.format(index)] = self.copy_row(row)
    out['gridworld'] = self.copy_flatgrid(grid.gridworld)
    
    # add the position and direction into the output as well
    out['distance'] = manhattan(position, (self.dim-1, self.dim-1))
    out['position'] = position
    out['direction'] = direction

    self.output.append(out)
    
  def copy_flatgrid(self, grid):
    return [i for row in grid for i in row]

  def get_direction(self, start, next):
    # 0 = left, 1 = up, 2 = right, 3 = down
    if next[0] - start[0] == 1:
      return 3
    elif next[0] - start[0] == -1:
      return 1
    elif next[1] - start[1] == 1:
      return 2
    else:
      return 0

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