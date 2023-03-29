# eatGhostPacmanGymEnv.py
# -------------------
# Integrate the Eat-Ghost-Pacman game into Open AI Gym

import gym
import numpy as np
from addict import Dict
from pacman_src import game
from eat_ghost_env import eatGhostPacmanGame

ACTION_ID_TO_DIRECTION = {
    0: game.Directions.STOP,
    1: game.Directions.EAST,
    2: game.Directions.WEST,
    3: game.Directions.SOUTH,
    4: game.Directions.NORTH
}

DIRECTION_TO_ACTION_ID = {
    game.Directions.STOP: 0,
    game.Directions.EAST: 1,
    game.Directions.WEST: 2,
    game.Directions.SOUTH: 3,
    game.Directions.NORTH: 4
}


# noinspection DuplicatedCode
class EatGhostPacmanGymEnv(gym.Env):
    """ The eat-ghost-pacman environment for OpenAI gym """
    metadata = {'render.modes': ['human', 'rbg_array']}

    def __init__(self, config_file=None):
        super(EatGhostPacmanGymEnv, self).__init__()
        # define the game configuration file (None means default setting)
        self.config_file = config_file
        # define the reward range
        self.reward_range = EatGhostPacmanGymEnv.reward_range
        # define the acton spaces
        # noinspection PyArgumentList
        self.action_space = gym.spaces.Discrete(len(ACTION_ID_TO_DIRECTION))
        # define the observation spaces
        # noinspection PyArgumentList
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(72, 189, 3), dtype=np.uint8)
        self.name = 'eat_ghost_pacman'
        self.spec = Dict({'name': self.name, 'action_space': self.action_space, 'observation_space': self.observation_space, 'reward_range': self.reward_range})

        # creat the internal game environment
        self.game = None

    def render(self, mode='human'):
        """ To deal with different types of outputs (text or graphics) here in the future """
        if mode not in EatGhostPacmanGymEnv.metadata['render.modes']:
            return
        self.game.render()

    def reset(self):
        """
        reset the environment to initial state
        :return: return the initial observation of the game (numpy array) and the rgb utility map (PIL)
        """
        if self.game is not None and not self.game.game_over:
            self.game.final()
        # restart a new game
        game_config = eatGhostPacmanGame.read_config(self.config_file)
        self.game = eatGhostPacmanGame.run_game(**game_config)
        numpy_obs = self.game.get_current_observation()
        return numpy_obs

    # noinspection PyTypeChecker
    def step(self, action):
        """
        Execute one time step within the environment
        :return: observation, reward, done, info
        """
        # map action id to game action
        game_action = ACTION_ID_TO_DIRECTION[int(action)]
        self.game.set_action(game_action)
        return self.game.update()

    #######################################################
    ########### methods not inherited from gym  ###########
    #######################################################

    def get_predicate_keys(self):
        return [eatGhostPacmanGame.PREDICATES.GHOST_PREDICATE.value, eatGhostPacmanGame.PREDICATES.CAPSULE_PREDICATE.value]

    def get_current_predicate(self):
        return self.game.get_current_predicate_values()

    def get_agent_last_move(self):
        return DIRECTION_TO_ACTION_ID[self.game.get_keyboard_agent_last_move()]

    def update_utility_map(self, utility_values, get_mask=False):
        """ return current utility map given the values: (utility_numpy, utility_img) """
        return self.game.get_current_utility_map(utility_values, get_mask=get_mask)

    def _is_win(self):
        """ Whether the agent won the game """
        return self.game.is_win()

    def _is_lose(self):
        """ Whether the agent lost the game """
        return self.game.is_lose()

    def _is_terminated(self):
        """ Whether the game has terminated """
        return self.game.game_over


if __name__ == '__main__':
    pass
