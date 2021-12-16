import tensorflow as tf
import numpy as np
from gridworld import Gridworld
from time import sleep, time
from cell import Cell
import random

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

class Agent_3:

    def __init__(self, dim):
        self.dim = dim
        # grid that Agent uses to keep track of each cell info
        self.cell_info = []
        for i in range(dim):
            row = []
            for j in range(dim):
                row.append(Cell(i, j, dim))
            self.cell_info.append(row)
        self.discovered_grid = Gridworld(dim)
        self.cg = [[0] * dim for i in range(dim)]
        self.cell_sense_map = [[-1] * dim for i in range(dim)]
        self.neural_network = tf.keras.models.load_model('./models/agent3_NN')

    def execute_path(self, complete_grid, timeout_sec):
        starting_time = time()
        new_start_time = time()
        time_elapsed = time() - new_start_time
        total_time_elapsed = time() - starting_time
        retries = 0
        trajectory_length = 0
        curr = (0,0)
        prev = curr
        while curr != (self.dim-1, self.dim-1):
            time_elapsed = time() - new_start_time
            total_time_elapsed = time() - starting_time
            print("Currently in: (%s, %s)" % (curr[0], curr[1]))
            trajectory_length += 1

            self.cg[curr[0]][curr[1]] += 1
            cell = self.cell_info[curr[0]][curr[1]]
            self.sense_neighbors(cell, complete_grid)
            self.discovered_grid.update_grid_obstacle(curr, 0)

            # mark the cell as visited
            cell.visited = True
            # mark cell as a confirmed value because it was visited
            cell.confirmed = True
            # use the new info to draw conclusions about neighbors
            new_confirmed_cells = self.update_neighbors(cell)

            in_grid = np.reshape(self.discovered_grid.gridworld, (1, 50, 50)) / 2
            locals_val = self.get_local(self.discovered_grid.gridworld, curr)
            in_local = np.reshape(locals_val, (1, 5, 5))
            print(in_local)
            in_position = np.reshape(self.get_position(curr), (1, 50, 50))
            in_sense = np.reshape(self.cell_sense_map, (1, 50, 50)) / 8

            prob_predict = self.neural_network.predict( [in_grid, in_position, in_sense, in_local] )
            prediction = np.argmax( prob_predict, axis = 1 )

            print("Taking direction: %s" % prediction[0])

            direction = self.get_direction(prediction[0])
            new_position = (curr[0] + direction[0], curr[1] + direction[1])

            if new_position[0] < 0 or new_position[0] >= self.dim or new_position[1] < 0 or new_position[1] >= self.dim:
                self.cg[curr[0]][curr[1]] += 1
            elif complete_grid.gridworld[new_position[0]][new_position[1]] == 1:
                retries += 1
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
                # random_rounds += 1
                # get options
                for i in range(5):
                    options = self.get_open_neighbors(curr, complete_grid.gridworld)
                    curr = random.choice(options)
                # reset time elapsed
                new_start_time = time()

            
        
        return True, trajectory_length, retries


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

    def update_neighbors(self, cell):
        # set that contains any cell that's been confirmed
        new_confirmed_cells = set()

        # add the neighbors of the current cell and itself to the list
        neighbors = set(cell.get_neighbors(self.cell_info, self.dim))
        neighbors.add(cell)

        # loop through the cells and keep looping until neighbors is empty
        while neighbors:
            curr_cell = neighbors.pop()
            changed = self.update_cell_info(curr_cell)

            # if the cell was visited and we have the block sense, infer and add to knowledge base
            if curr_cell.visited and curr_cell.block_sense != -1:
                updated_cells = self.update_knowledgebase(curr_cell)
                new_confirmed_cells.update(updated_cells)

                # update all of the neighbors neighbors by adding those to the set
                for n in updated_cells:
                    neighbors.update(n.get_neighbors(self.cell_info, self.dim))
                    neighbors.add(n)

        return new_confirmed_cells

    def sense_neighbors(self, cell, complete_grid):
        num_sensed = 0
        neighbors = cell.get_neighbors(self.cell_info, self.dim)

        # loop through the neighbors to be checked and take the sum (1 is block)
        num_sensed = sum(complete_grid.gridworld[n.x][n.y] for n in neighbors)

        # return the number of obstacles surrounding the current node
        cell.block_sense = num_sensed
        self.cell_sense_map[cell.x][cell.y] = num_sensed
    
    def update_cell_info(self, cell):
        num_hidden = 0
        num_block = 0
        num_empty = 0
        neighbors = cell.get_neighbors(self.cell_info, self.dim)

        # loop through the neighbors to be checked
        for n in neighbors:
            if n.confirmed:
                # check and increment if it is blocked
                if self.discovered_grid.gridworld[n.x][n.y] == 1:
                    num_block += 1
                # otherwise increment the empty counter
                else:
                    num_empty += 1
            # the neighbor cell has not been explored yet
            else:
                num_hidden += 1

        has_changed = (
            (cell.hidden - num_hidden)
            or (cell.confirm_block - num_block)
            or (cell.confirm_empty - num_empty)
        )

        if has_changed:
            cell.hidden = num_hidden
            cell.confirm_block = num_block
            cell.confirm_empty = num_empty

        return has_changed

    def update_knowledgebase(self, cell):

        updated_cells = []

        # if there are hidden not cells, leave
        if cell.hidden == 0:
            return updated_cells

        # get the neighbors and check to see which are blockers
        neighbors = cell.get_neighbors(self.cell_info, self.dim)

        # if we know all block cells, update the other cells to be empty
        if cell.block_sense == cell.confirm_block:
            for n in neighbors:
                if not n.confirmed:
                    self.discovered_grid.update_grid_obstacle((n.x, n.y), 0)
                    n.confirmed = True
                    updated_cells.append(n)
            return updated_cells

        # if we know all empty cells, update the other cells to be blocked
        if cell.neighbors - cell.block_sense == cell.confirm_empty:
            for n in neighbors:
                if not n.confirmed:
                    self.discovered_grid.update_grid_obstacle((n.x, n.y), 1)
                    n.confirmed = True
                    updated_cells.append(n)

        return updated_cells
  