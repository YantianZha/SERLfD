# proxyAgents.py
# -----------------
# The proxy agent that takes the action specified in the step function

from pacman_src.game import Agent
from pacman_src.pacman import Directions


# noinspection PyMissingConstructor
class ProxyAgent(Agent):
    """
    An agent that takes the action specified in the step function
    """

    def __init__(self, index=0):
        super().__init__()

        self.next_action = Directions.STOP
        self.utilities = None
        self.index = index

    def getAction(self, state):
        next_action = self.next_action
        if next_action is None:
            return Directions.STOP

        # check if the specified action is legal
        legal = state.getLegalActions(self.index)
        if next_action not in legal:
            return Directions.STOP

        self.next_action = None
        return next_action

    def setAction(self, action):
        self.next_action = action

    def setUtility(self, utilities):
        self.utilities = utilities

    def final(self, state):
        """ let the true agent perform final operation """
        return

