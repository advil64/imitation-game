# used to read in command line args (dim p heuristic algo)
import argparse
from time import sleep, time
from gridworld import Gridworld
from agent_1 import Agent_1
from agent_3 import Agent_3
from heuristics import manhattan
from a_star import path_planner
import json
from pprint import pprint
import calendar

"""
  Creates a gridworld and carrys out repeated A* based on the agent
  @param dim: dimension of the grid
  @param prob: probability of having a blocker
  @param agent: the type of visibility we have
  @param complete_grid: optional supplied grid instead of creating one 
"""

def solver(dim, prob, directory, complete_grid=None):

    complete_grid = Gridworld(dim, prob, False)
    while not verify_solvability(dim, complete_grid):
      # keep generating a new grid until we get a solvable one
      complete_grid = Gridworld(dim, prob, False)

    # create agents
    agents = [Agent_1(dim, []), Agent_3(dim)]

    for count, agent_object in enumerate(agents):
        # total number of cells explored
        trajectory_length = 0
        # total number of cells processed
        total_cells_processed = 0
        # final path which points to last node
        final_path = None
        # number of times A* was repeated
        retries = 0
        # status of completion
        complete_status = False

        # perform repeated A* with the agent
        starting_time = time()
        # start planning a path from the starting block
        new_path, cells_processed, path_coord = path_planner(
            (0, 0), final_path, agent_object.discovered_grid, dim, manhattan
        )
        total_cells_processed += cells_processed
        # while A* finds a new path
        while len(new_path) > 0:
            retries += 1
            # execute the path
            last_node, explored = agent_object.execute_path(new_path, complete_grid, path_coord)
            trajectory_length += explored
            final_path = last_node
            # get the last unblocked block
            last_block = (0, 0)
            last_unblock_node = None
            if last_node:
                last_block = last_node.curr_block
                last_unblock_node = last_node.parent_block
            # check if the path made it to the goal node
            if last_block == (dim - 1, dim - 1):
                complete_status = True
                break
            # create a new path from the last unblocked node
            new_path, cells_processed, path_coord = path_planner(
                last_block,
                last_unblock_node,
                agent_object.discovered_grid,
                dim,
                manhattan,
            )
            total_cells_processed += cells_processed

        # Update discovered grid to fill unknowns with 1
        for i in range(dim):
            for j in range(dim):
                curr = agent_object.discovered_grid.gridworld[i][j]
                if curr == 9:
                    agent_object.discovered_grid.update_grid_obstacle((i,j), 1)

        # Get length of final path when running A* on the final discoverd_grid
        shortest_trajectory = grid_solver(dim, agent_object.discovered_grid)

        completion_time = time() - starting_time

        # print("Agent %s Completed in %s seconds" % (agent_counter, completion_time))
        # print("Agent %s Processed %s cells" % (agent_counter, total_cells_processed))
        # print("Agent %s Retried %s times" % (agent_counter, retries))
        # print("Agent %s Has Trajectory Length %s" % (agent_counter, trajectory_length))
    
    # get time and write to file
    with open('{}/agent_1_{}.json'.format(directory, int(starting_time)), 'w') as outfile:
        json.dump(agents[0].output, outfile)

def verify_solvability(dim, complete_grid):
    # start planning a path from the starting block
    new_path, cells_processed, s = path_planner((0,0), None, complete_grid, dim, manhattan)

    # Check if a path was found
    if not new_path:
      return False
    
    return True

def grid_solver(dim, discovered_grid):
    final_path = None

    # start planning a path from the starting block
    new_path, cells_processed, path_coord = path_planner((0,0), final_path, discovered_grid, dim, manhattan)
    
    trajectory = 0

    if new_path:
        final_path = new_path[-1]
        trajectory = get_trajectory(final_path)
    
    return trajectory

def get_trajectory(path):
    trajectory_length = 0
    while path:
        path = path.parent_block
        trajectory_length += 1
    return trajectory_length


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-d", "--dimension", type=int, default=5, help="dimension of gridworld"
    )
    p.add_argument(
        "-p", "--probability", type=float, default=0.33, help="probability of a blocked square"
    )
    p.add_argument(
        "-w", "--directory", type=str, default='data/default', help='directory to store the json in'
    )

    # parse arguments and create the gridworld
    args = p.parse_args()

    # call the solver method with the args
    solver(args.dimension, args.probability, args.directory)


if __name__ == "__main__":
    main()
