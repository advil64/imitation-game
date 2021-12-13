# used to read in command line args (dim p heuristic algo)
import argparse
from time import sleep, time
from gridworld import Gridworld
from agent_1_nn_sim import Agent_1 as agent_1_nn
from agent_1_cnn_sim import Agent_1 as agent_1_cnn
from agent_3 import Agent_3
from heuristics import manhattan
from a_star import path_planner
import json
from pprint import pprint
import calendar
from main import solver as normal_solver

"""
  Creates a gridworld and carrys out repeated A* based on the agent
  @param dim: dimension of the grid
  @param prob: probability of having a blocker
  @param agent: the type of visibility we have
  @param complete_grid: optional supplied grid instead of creating one 
"""

def solver(dim, prob, directory, complete_grid=None):

    nn_res = {}
    cnn_res = {}

    complete_grid = Gridworld(dim, prob, False)
    while not verify_solvability(dim, complete_grid):
      # keep generating a new grid until we get a solvable one
      complete_grid = Gridworld(dim, prob, False)
    
    complete_grid.print()

    # first run the normal agent on the complete grid and collect data
    normal_out = normal_solver(dim, prob, complete_grid)

    # create agents (ignored agent 3)
    agents = [agent_1_nn(dim), agent_1_cnn(dim)]

    try:
        for c, i in enumerate(agents):
            starting_time = time()
            success, trajectory_length, retries = i.execute_path(complete_grid, 20)
            completion_time = time() - starting_time
            if c == 0:
                # write to json
                nn_res['success'] = True
                nn_res['completion_time'] = completion_time
                nn_res['retries'] = retries
                nn_res['trajectory_length'] = trajectory_length
            else:
                # write to json
                cnn_res['success'] = True
                cnn_res['completion_time'] = completion_time
                cnn_res['retries'] = retries
                cnn_res['trajectory_length'] = trajectory_length
    except Exception():
        success = False
        # write to json
        if nn_res == {}:
            nn_res['success'] = False
        else:
            cnn_res['success'] = False
        print("Agent failed")

    # write the jsons to a file
    with open('results/agent_1/{}.json'.format(int(starting_time)), 'w') as outfile:
        json.dump({'agent_1': normal_out, 'agent_1_nn': nn_res, 'agent_1_cnn': cnn_res}, outfile)

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
