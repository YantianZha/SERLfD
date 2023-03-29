# run_pacman_gym.py
# ------------------
# script to run the eat-ghost pacman game (integrated with Open AI gym)

import os
from PIL import Image

from eat_ghost_env import eatGhostPacmanGymEnv
from eat_ghost_env.eatGhostPacmanGame import PREDICATES
import gym
import time
from utils.experiment_record_utils import ExperimentLogger

TEMP_RESULT_SAVING_DIR = 'tmp/'
RESULT_SAVING_DIR = '../../experiment_results/'

env = gym.make('CartPole-v1')

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.


human_agent_action = 0
human_wants_restart = False
human_sets_pause = False

def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    human_agent_action = a

def key_release(key, mod):
    global human_agent_action
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0

env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release


# noinspection DuplicatedCode
def run_game():
    global human_agent_action, human_wants_restart, human_sets_pause
    human_wants_restart = False

    experiment_recorder = ExperimentLogger(RESULT_SAVING_DIR, 'cart_pole')
    experiment_recorder.redirect_output_to_logfile_as_well()
    print('Start recording demo of cart_pole')

    n_episodes = 10

    for i_episode in range(n_episodes):
        # get initial observation
        old_obs_numpy = env.reset()

        env.render()
        is_done = False
        skip = 0
        n_steps = 0
        while not is_done and n_steps < 50:
            env.render()

            if not skip:
                # print("taking action {}".format(human_agent_action))
                a = human_agent_action
                skip = SKIP_CONTROL
            else:
                skip -= 1

            action = human_agent_action
            print('action: ', action)

            # interact with the environment
            new_obs_numpy, reward, is_done, info = env.step(action)
            # get the actual action taken by the agent
            agent_last_move = action

            # save transition
            experiment_recorder.add_transition(old_obs_numpy, agent_last_move, reward, new_obs_numpy, is_done,
                                               is_save_utility=False, predicate_values=None,
                                               utility_map=None, utility_values=None)
            old_obs_numpy = new_obs_numpy
            n_steps += 1

            time.sleep(0.5)

            if is_done:
                print('episode end222222222')

    experiment_recorder.save_trajectories(is_save_utility=False)


if __name__ == '__main__':
    run_game()
