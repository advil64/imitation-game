from gridworld import Gridworld
from cell import Cell


class Agent_3:
    def __init__(self, dim, output):
        self.dim = dim
        self.discovered_grid = Gridworld(dim)
        # grid that Agent uses to keep track of each cell info
        self.cell_info = []
        for i in range(dim):
            row = []
            for j in range(dim):
                row.append(Cell(i, j, dim))
            self.cell_info.append(row)
        self.path_world = Gridworld(dim)
        self.cg = [[0] * dim for i in range(dim)]
        self.cell_sense_map = [[-1] * dim for i in range(dim)]
        self.output = output

    """
    Given a path (from repeated A*) we traverse through the gridworld and gather information
    @param path: given path as a list of nodes
    @param complete_grid: the full gridworld to check for obstacles along the way
    @return list: the last node of the gridworld or the parent node
    """

    def execute_path(self, path, complete_grid, path_coord):
        explored = 0
        for index,node in enumerate(path):
            curr = node.curr_block
            path_coord.remove(curr)
            cell = self.cell_info[curr[0]][curr[1]]
            to_ret = None
            bump = False

            # check if path is open
            if complete_grid.gridworld[cell.x][cell.y] == 0:
                # sense Agent's surroundings to determine number of blocks around
                self.sense_neighbors(cell, complete_grid)
                # update our knowledge of blocked nodes
                self.discovered_grid.update_grid_obstacle(curr, 0)

                self.path_world.update_grid_obstacle(curr,0)
                self.cg[curr[0]][curr[1]] += 1
                # Add step info to dataset
                if index < len(path)-1 and self.discovered_grid.gridworld[path[index + 1].curr_block[0]][path[index + 1].curr_block[1]] != 1:
                    self.add_to_json(self.path_world, curr, self.get_direction(curr, path[index + 1].curr_block))

                # return the last node
                to_ret = node
            else:
                self.discovered_grid.update_grid_obstacle(curr, 1)
                self.path_world.update_grid_obstacle(curr, 1)
                # return the parent and tell the gridworld to find a new path
                to_ret = node.parent_block
                # set bump to true to indicate we bumped into an obstacle
                bump = True

            # mark the cell as visited
            cell.visited = True
            # mark cell as a confirmed value because it was visited
            cell.confirmed = True
            # use the new info to draw conclusions about neighbors
            new_confirmed_cells = self.update_neighbors(cell)

            # if we bumped into an obstacle, then leave the execution
            if bump:
                return to_ret, explored
            explored += 1
            # check if any new confirmed cells introduce block in your path
            if self.check_block_in_path(path_coord, new_confirmed_cells):
                return to_ret, explored

        return path[-1], explored
    
    def add_to_json(self, grid, position, direction):

        # first map grid to a json compatible object
        out = {}

        # loop through the grid and convert each row to a string
        # for index,row in enumerate(grid.gridworld):
        #   out['row_{}'.format(index)] = self.copy_row(row)
        out['gridworld'] = self.copy_flatgrid(grid.gridworld)
        
        # add the position and direction into the output as well
        out['sense'] = self.copy_flatgrid(self.cell_sense_map)
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

    """
    This method returns the number of blocked neighbors a given cell has
    @param cell: the current cell in the path traversal
    @param complete_grid: grid with the obstacles
    """

    def sense_neighbors(self, cell, complete_grid):
        num_sensed = 0
        neighbors = cell.get_neighbors(self.cell_info, self.dim)

        # loop through the neighbors to be checked and take the sum (1 is block)
        num_sensed = sum(complete_grid.gridworld[n.x][n.y] for n in neighbors)

        # return the number of obstacles surrounding the current node
        cell.block_sense = num_sensed
        self.cell_sense_map[cell.x][cell.y] = num_sensed

    """
    This method stores the surrounding information of any given cell based on the discovered grid
    @param cell: cell to calculate surroundings
    @return boolean: indicating if any values have changed
    """

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

    def check_block_in_path(self, path_coord, new_confirmed_cells):
        for cell in new_confirmed_cells:
            if self.discovered_grid.gridworld[cell.x][cell.y] == 1:
                if (cell.x, cell.y) in path_coord:
                    return True

        return False
