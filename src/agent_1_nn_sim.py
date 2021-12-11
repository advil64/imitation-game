import tensorflow as tf
from gridworld import Gridworld

class Agent_1:

  def __init__(self, dim, output):
    self.dim = dim
    self.discovered_grid = Gridworld(dim)
    self.output = output
    self.neural_network = tf.keras.models.load_model('../models/agent1_NN')

  def execute_path(self):
    curr = (0,0)
    while curr != (self.dim-1, self.dim-1):
        prob_predict = self.neural_network.predict( [[self.discovered_grid.gridworld]] )
        predictions = np.argmax( prob_predict, axis = 1 )