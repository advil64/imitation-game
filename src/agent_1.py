from gridworld import Gridworld
from heuristics import manhattan

class Agent_1:

  def __init__(self, dim, output):
    self.dim = dim
    self.discovered_grid = Gridworld(dim)
    self.cg = [[0] * dim for i in range(dim)]
    self.output = output
    self.path = []

  def execute_path(self, path, complete_grid, path_coord):
    explored = 0
    for index, node in enumerate(path):
      curr = node.curr_block
      # check if path is blocked
      if complete_grid.gridworld[curr[0]][curr[1]] == 1:
        # update our knowledge of blocked nodes
        self.discovered_grid.update_grid_obstacle(curr, 1)
        return node.parent_block, explored

      self.discovered_grid.update_grid_obstacle(curr, 0)
      # update knowledge of neighbor blocks
      self.update_neighbor_obstacles(curr, complete_grid)
      self.cg[curr[0]][curr[1]] += 1

      # add to the path

      # Add step info to dataset
      if index < len(path)-1 and self.discovered_grid.gridworld[path[index + 1].curr_block[0]][path[index + 1].curr_block[1]] != 1:
        self.add_to_json(self.discovered_grid, curr, self.get_direction(curr, path[index + 1].curr_block))

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
    out['position'] = self.copy_flatgrid(self.get_position(position))
    out['direction'] = direction
    out['local'] = self.get_local(grid.gridworld, position)

    self.output.append(out)
    
  def copy_flatgrid(self, grid):
    return [i for row in grid for i in row]
  
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