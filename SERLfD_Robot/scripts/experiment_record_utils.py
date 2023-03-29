# experiment_record_utils.py
# -------------------------
# helper functions to record experiment data

from datetime import datetime
import os
import sys
import shutil
import pickle
import numpy as np
import torch
# import wandb
import imageio
import cv2


DEFAULT_PROJECT_NAME = 'default project'
DEFAULT_IS_USE_WANDB = True


class ExperimentLogger:
    def __init__(self, saved_dir, expr_name, is_add_time_to_name=True, save_trajectories=True, mode='rgb'):
        if is_add_time_to_name:
            now = datetime.now()
            d = now.strftime('%Y%m%d_%H%M%S')
            expr_name = expr_name + '_' + d
        self.expr_name = expr_name
        self.project_name = DEFAULT_PROJECT_NAME
        self.is_use_wandb = DEFAULT_IS_USE_WANDB
        self.is_save_trajs = save_trajectories

        # create expr dir
        if not os.path.exists(saved_dir):
            os.mkdir(saved_dir)
        self.saved_dir = os.path.join(saved_dir, expr_name)
        if not os.path.exists(self.saved_dir):
            os.mkdir(self.saved_dir)

        # create image saved dir
        self.img_dir = os.path.join(self.saved_dir, 'images')
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)

        # create snapshot saved dir
        self.snapshot_dir = os.path.join(self.saved_dir, 'snapshots')
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)

        # trajectories holder
        self.basic_trajectories = list()    # traj with basic information
        self.current_basic_trajectory = list()
        self.util_trajectories = list()
        self.current_util_traj = list()

        # create log file
        self.logfile = open(os.path.join(self.saved_dir, 'log.txt'), 'a')

        self.mode = mode

    def add_trajectories(self, trajs):
        """ Save the basic trajectories """
        for traj in trajs:
            for t in range(len(traj)):
                curr_state = traj[t][0]
                action = traj[t][1]
                reward = traj[t][2]
                next_state = traj[t][3]
                done = traj[t][4]
                self.add_transition(curr_state, action, reward, next_state, done, is_save_utility=False)

    def add_transition(self, curr_state, action, reward, next_state, done
                       , is_save_utility=False, predicate_values=None, next_predicate_values=None
                       , utility_map=None, utility_values=None):
        """
        Save the transition and related utility information
        The utility information is stored in the format of [(predicate_values, next_predicate_values, utility_map, utility_values)]
        """
        if not self.is_save_trajs:
            return
        self.current_basic_trajectory.append((curr_state, action, reward, next_state, done))
        if done:
            self.basic_trajectories.append(self.current_basic_trajectory)
            self.current_basic_trajectory = list()
        if is_save_utility:
            if utility_map is None:
                utility_map = np.zeros(shape=(1,))
            if utility_values is None:
                utility_values = {}
            if predicate_values is None:
                predicate_values = {}
            if next_predicate_values is None:
                next_predicate_values = {}
            self.current_util_traj.append((predicate_values, next_predicate_values, utility_map, utility_values))
            if done:
                self.util_trajectories.append(self.current_util_traj)
                self.current_util_traj = list()

    def save_models(self, checkpoint, postfix=None, is_snapshot=True, prefix='model'):
        """
        Save current model
        :param checkpoint: the parameters of the models, see example in pytorch's documentation: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        :param is_snapshot: whether saved in the snapshot directory
        :param prefix: the prefix of the file name
        :param postfix: the postfix of the file name (can be episode number, frame number and so on)
        """
        saved_dir = self.snapshot_dir if is_snapshot else self.saved_dir

        if postfix is not None:
            fname = get_unique_fname(saved_dir, prefix + '_' + postfix + '.tar')
        else:
            fname = get_unique_fname(saved_dir, prefix + '_' + self.expr_name + '.tar')
        torch.save(checkpoint, os.path.join(saved_dir, fname))

    def save_trajectories_snapshot(self, is_save_utility=False):
        """
        save trajectories to file
        is_separated_file: whether to save each trajectory to its own file
        is_save_utility: whether to save utility
        """
        fname = 'traj_' + self.expr_name + '.pickle'
        fname = get_unique_fname(self.snapshot_dir, fname)
        fname = os.path.join(self.snapshot_dir, fname)
        with open(fname, 'wb') as f_traj:
            pickle.dump(self.basic_trajectories, f_traj, protocol=pickle.HIGHEST_PROTOCOL)
        if is_save_utility:
            fname = 'utilities_' + self.expr_name + '.pickle'
            fname = get_unique_fname(self.snapshot_dir, fname)
            fname = os.path.join(self.snapshot_dir, fname)
            with open(fname, 'wb') as f_util:
                pickle.dump(self.util_trajectories, f_util, protocol=pickle.HIGHEST_PROTOCOL)

    def im2predicates(self, img, num_predicates, c=-1):
        num_dims = len(img.shape)
        assert num_dims == 3 or num_dims == 4, "input is not an image"
        return img[:, c, :, :].flatten()[:num_predicates] if num_dims == 4 else img[c, :, :].flatten()[
                                                                                    :num_predicates]

    def save_trajectories(self, num_predicates, fps=None, show_predicates=False, is_separated_file=False, is_save_utility=False):
        """
        save trajectories to file
        is_separated_file: whether to save each trajectory to its own file
        is_save_utility: whether to save utility
        """

        def draw_predicates_on_img(image, num_p, size=None, rwd=None, done=None):
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # org
            org = (00, 415)
            # fontScale
            fontScale = 0.5
            # Red color in BGR
            color = (0, 0, 255)
            # Line thickness of 2 px
            thickness = 1

            msg = self.im2predicates(image, num_p)
            if rwd:
                msg = np.concatenate((msg, [rwd]))
            if done:
                msg = np.concatenate((msg, [np.float(done)]))

            msg = np.around(msg, decimals=1)
            max_words_per_line = 13

            if self.mode == 'rgb':
                # Remove the simple states channel and only keep the RGB channels
                image = np.moveaxis(image[:3], 0, -1).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = np.moveaxis(image[:3], 0, -1)
                image = cv2.normalize(np.expand_dims(image[:,:,0], -1), None, 0, 255, cv2.NORM_MINMAX)

            if size:
                image = cv2.resize(image, size)

            while len(msg) % max_words_per_line > 0:
                text = msg[:max_words_per_line]
                text = str(text)
                image = cv2.putText(image, text, tuple(org), font, fontScale,
                                    color, thickness, cv2.LINE_AA, False)
                msg = msg[max_words_per_line:]
                org = list(org)
                org[1] += 15
                if len(msg) % max_words_per_line == 0:
                    text = str(msg)
                    image = cv2.putText(image, text, tuple(org), font, fontScale,
                                        color, thickness, cv2.LINE_AA, False)
            return image

        if is_separated_file:
            print("UUU", len(self.basic_trajectories[0]))
            for i in range(len(self.basic_trajectories)):
                fname = 'traj_' + str(i) + '_' + self.expr_name + '.pickle'
                fname = os.path.join(self.saved_dir, fname)
                print("YYY", fname)
                with open(fname, 'wb') as f_traj:
                    pickle.dump([self.basic_trajectories[i]], f_traj, protocol=pickle.HIGHEST_PROTOCOL)
                if is_save_utility:
                    fname = 'utilities_' + str(i) + '_' + self.expr_name + '.pickle'
                    fname = os.path.join(self.saved_dir, fname)
                    with open(fname, 'wb') as f_util:
                        pickle.dump([self.util_trajectories[i]], f_util, protocol=pickle.HIGHEST_PROTOCOL)
                if fps:
                    fname = 'traj_video_' + str(i) + '_' + self.expr_name + '.mp4'
                    fname = os.path.join(self.saved_dir, fname)
                    if show_predicates:
                        states = np.array([draw_predicates_on_img(tran[0], num_predicates, (512, 512)) for tran in self.basic_trajectories[i]] + [draw_predicates_on_img(self.basic_trajectories[i][-1][3], num_predicates, (512, 512))])
                    else:
                        states = np.array([tran[0] for tran in self.basic_trajectories[i]])
                        if self.mode == 'depth':
                            states = [255 * (depth - depth.min()) / (depth.max() - depth.min()) for depth in states]


                    for i, s in enumerate(states):
                        self.save_rgb_image(self.img_dir, s, step=i, episode=self.expr_name)
                    self.save_states_to_video(fname, states, fps)
        else:
            fname = 'traj_' + self.expr_name + '.pickle'
            fname = os.path.join(self.saved_dir, fname)
            with open(fname, 'wb') as f_traj:
                pickle.dump(self.basic_trajectories, f_traj, protocol=pickle.HIGHEST_PROTOCOL)
            if is_save_utility:
                fname = 'utilities_' + self.expr_name + '.pickle'
                fname = os.path.join(self.saved_dir, fname)
                with open(fname, 'wb') as f_util:
                    pickle.dump(self.util_trajectories, f_util, protocol=pickle.HIGHEST_PROTOCOL)

    def save_rgb_image(self, fname, img, img_format='png', step=None, episode=None):
        if step is None:
            ts = str(datetime.now().timestamp()).replace('.', '_')
            step = ts
            episode = 'NAN'
        fname = fname + '_' + str(episode) + '_' + str(step) + '.' + img_format
        # fname = os.path.join(self.img_dir, fname)
        # img.save(fname)
        cv2.imwrite(fname, img)

    def save_states_to_video(self, fname, states, fps):
        imageio.mimwrite(fname, states)

    def copy_file(self, source):
        config_fname = os.path.basename(source)
        saved_fname = os.path.join(self.saved_dir, config_fname)
        shutil.copyfile(source, saved_fname)

    def redirect_output_to_logfile_as_well(self):
        class Logger(object):
            def __init__(self, logfile):
                self.stdout = sys.stdout
                self.logfile = logfile

            def write(self, message):
                self.stdout.write(message)
                self.logfile.write(message)

            def flush(self):
                # this flush method is needed for python 3 compatibility.
                # this handles the flush command by doing nothing.
                # you might want to specify some extra behavior here.
                pass
        sys.stdout = Logger(self.logfile)
        sys.stderr = sys.stdout

    def set_is_use_wandb(self, is_use):
        self.is_use_wandb = is_use

    def set_wandb_project_name(self, name):
        self.project_name = name

    def set_wandb(self):
        if self.is_use_wandb:
            wandb.init(project=self.project_name, name=self.expr_name)

    def save_config_wandb(self, config):
        """
        The config is assumed to be a nested dict, e.g. {"hyper param": {"learning rate": 0.1}}
        """
        wandb_config = {}
        for key0 in config:
            for key1 in config[key0]:
                param_name = str(key0) + '_' + str(key1)
                param_value = config[key0][key1]
                wandb_config[param_name] = str(param_value)
        if self.is_use_wandb:
            wandb.config.update(wandb_config)

    def log_wandb(self, log_dict, step=None):
        if self.is_use_wandb:
            if step is not None:
                wandb.log(log_dict, step=step)
            else:
                wandb.log(log_dict)

    def watch_wandb(self, model_list):
        if self.is_use_wandb:
            wandb.watch(model_list, log='parameters')

    def open_virtual_display(self):
        from pyvirtualdisplay import Display
        display = Display(visible=0, size=(1080, 608))
        display.start()


def get_unique_fname(file_dir, fname_base):
    name, extension = os.path.splitext(fname_base)
    post_fix = 0
    while True:
        fname = name + '_' + str(post_fix) + extension
        if not os.path.exists(os.path.join(file_dir, fname)):
            return fname
        post_fix += 1






