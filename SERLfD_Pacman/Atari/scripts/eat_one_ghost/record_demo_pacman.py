# run_pacman_gym.py
# ------------------
# script to run the eat-ghost pacman game (integrated with Open AI gym)
import copy
import os
from PIL import Image

from eat_ghost_env import eatGhostPacmanGymEnv
from eat_ghost_env.eatGhostPacmanGame import PREDICATES
from utils.experiment_record_utils import ExperimentLogger
from utils.gym_atari_env_wrapper import PytorchImage

TEMP_RESULT_SAVING_DIR = 'tmp/'
RESULT_SAVING_DIR = '../../experiment_results/'


# noinspection DuplicatedCode
def run_game():
    experiment_recorder = ExperimentLogger(RESULT_SAVING_DIR, 'eat_2_ghost_5_demo', save_trajectories=True)
    experiment_recorder.redirect_output_to_logfile_as_well()
    config_file = '../../experiment_configs/expr_pacman_record_demo.cnf'
    experiment_recorder.copy_file(config_file)

    env = eatGhostPacmanGymEnv.EatGhostPacmanGymEnv(config_file)
    pred_utility_values = {PREDICATES.GHOST_PREDICATE.value: -0.8, PREDICATES.CAPSULE_PREDICATE.value: 0.8}

    n_episodes = 5
    n_steps = 0
    for i_episode in range(n_episodes):
        print('\n', '#'*50, '\n[INFO] current episode: ', i_episode)
        # get initial observation
        old_obs_numpy = env.reset()
        # get initial utility map
        old_utility_numpy = env.update_utility_map(pred_utility_values)

        if i_episode < 1:
            print('observation shape: ', old_obs_numpy.shape)
            print('utility map shape: ', old_utility_numpy.shape)

        env.render()
        is_done = False
        n_steps = 0
        while not is_done:
            env.render()
            # get current predicate values and get current utility map
            predicate_values = env.get_current_predicate()
            print('[INFO] current predicate: ', predicate_values)

            # since the action is given by human expert, we don't need to query the explainer here
            old_utility_numpy = env.update_utility_map(pred_utility_values)
            if i_episode < 20:
                utility_img = Image.fromarray(old_utility_numpy, 'RGB')
                # save the utility map image
                experiment_recorder.save_rgb_image('utility', utility_img, step=n_steps + 1, episode=i_episode)

            # Sample a random aaction (no effect since we will use keyboard input)
            action = env.action_space.sample()
            # interact with the environment
            new_obs_numpy, reward, is_done, info = env.step(action)
            # get the actual action taken by the agent
            agent_last_move = env.get_agent_last_move()

            # get next predicate
            next_predicate_values = env.get_current_predicate()
            if is_done:
                next_predicate_values = predicate_values
            print('[INFO] next predicate: ', next_predicate_values)

            # save the new observation image
            if i_episode < 20:
                new_obs_img = Image.fromarray(new_obs_numpy, 'RGB')
                experiment_recorder.save_rgb_image('obs', new_obs_img, step=n_steps + 1, episode=i_episode)

            # save transition
            experiment_recorder.add_transition(old_obs_numpy, agent_last_move, reward, new_obs_numpy, is_done,
                                               is_save_utility=True, predicate_values=predicate_values,
                                               next_predicate_values=next_predicate_values,
                                               utility_map=old_utility_numpy, utility_values=pred_utility_values)
            old_obs_numpy = new_obs_numpy
            n_steps += 1

    experiment_recorder.save_trajectories(is_save_utility=True)


if __name__ == '__main__':
    run_game()
