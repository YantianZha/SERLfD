# -*- coding: utf-8 -*-
"""Common util functions for all algorithms.

- Author: Curt Park
- Contact: curt.park@medipixel.io
"""

import random
from collections import deque
import cv2
import numpy as np
import torch

np.set_printoptions(suppress=True) #prevent numpy exponential
                                   #notation on print, default False
# https://stackoverflow.com/questions/9777783/suppress-scientific-notation-in-numpy-when-creating-array-from-nested-list

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def identity(x):
    """Return input without any change."""
    return x


def soft_update(local, target, tau):
    """Soft-update: target = tau*local + (1-tau)*target."""
    for t_param, l_param in zip(target.parameters(), local.parameters()):
        t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)


def set_random_seed(seed, env):
    """Set random seed"""
    env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def reverse_action(action, action_space):
    """Change the range (low, high) to (-1, 1)."""
    low = action_space.low
    high = action_space.high

    scale_factor = (high - low) / 2
    reloc_factor = high - scale_factor

    action = (action - reloc_factor) / scale_factor
    action = np.clip(action, -1.0, 1.0)

    return action

# def preprocess_simple_demos(demos, action_space=None, reward_scale=1.0, exe_single_group=True, exe_group_num=0, use_bi_reward=False, goal_reward=0.0):

def preprocess_demos(demos, resz=None, action_space=None, reward_scale=1.0, exe_single_group=True, exe_group_num=0, use_bi_reward=False, goal_reward=0.0):
    for id, transition in enumerate(demos):
        transition = np.array(transition)

        if resz:
            obs1 = np.moveaxis(transition[0], 0, -1)
            obs2 = np.moveaxis(transition[3], 0, -1)

            obs1_p_f = obs1[:, :, -1].flatten()
            obs2_p_f = obs2[:, :, -1].flatten()
            if not exe_single_group:
                obs1_p_f[:exe_group_num] = np.zeros(exe_group_num)
                obs2_p_f[:exe_group_num] = np.zeros(exe_group_num)

            obs1 = np.concatenate((cv2.resize(obs1[:, :, :3], dsize=resz), np.expand_dims(np.resize(obs1_p_f[:resz[0]*resz[1]], (resz[1], resz[0])), -1)), axis=2)
            transition[0] = np.moveaxis(obs1,-1, 0)

            obs2 = np.concatenate((cv2.resize(obs2[:, :, :3], dsize=resz), np.expand_dims(np.resize(obs2_p_f[:resz[0]*resz[1]], (resz[1], resz[0])), -1)), axis=2)
            transition[3] = np.moveaxis(obs2, -1, 0)

        transition[1] = np.array(transition[1])

        if not exe_single_group:
            transition[1] = transition[1][len(transition[1])-action_space.shape[0]:len(transition[1])]

        if action_space:
            transition[1] = reverse_action(transition[1], action_space)

        if use_bi_reward:
            transition[2] = 0.0 if transition[2] != goal_reward else goal_reward

        if reward_scale != 1.0:
            transition[2] = reward_scale * transition[2]

        demos[id] = transition
    return demos


def get_n_step_info_from_demo(demo, n_step, gamma):
    """Return 1 step and n step demos."""
    assert demo
    assert n_step > 1

    demos_1_step = list()
    demos_n_step = list()
    n_step_buffer = deque(maxlen=n_step)

    for transition in demo:
        n_step_buffer.append(transition)

        if len(n_step_buffer) == n_step:
            # add a single step transition
            demos_1_step.append(n_step_buffer[0])

            # add a multi step transition
            curr_state, action = n_step_buffer[0][:2]
            reward, next_state, done = get_n_step_info(n_step_buffer, gamma)
            transition = (curr_state, action, reward, next_state, done)
            demos_n_step.append(transition)

    return demos_1_step, demos_n_step


def get_n_step_info(n_step_buffer, gamma):
    """Return n step reward, next state, and done."""
    # info of the last transition
    reward, next_state, done = n_step_buffer[-1][-3:]

    for transition in reversed(list(n_step_buffer)[:-1]):
        r, n_s, d = transition[-3:]

        reward = r + gamma * reward * (1 - d)
        next_state, done = (n_s, d) if d else (next_state, done)

    return reward, next_state, done


def im2predicates(img, num_predicates, c=-1):
    num_dims = len(img.shape)
    assert num_dims == 3 or num_dims == 4, "input is not an image"
    return img[:, c, :, :].flatten()[:num_predicates] if num_dims == 4 else img[c, :, :].flatten()[
                                                                            :num_predicates]


def draw_predicates_on_img(image, num_p, size=None, rwd=None, done=None, utilv=None):
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

    msg = im2predicates(image, num_p)

    if utilv is not None:
        msg = np.concatenate((msg, utilv))

    if rwd:
        msg = np.concatenate((msg, [rwd]))
    if done:
        msg = np.concatenate((msg, [np.float(done)]))

    msg = np.around(msg, decimals=2)
    max_words_per_line = 12

    # Remove the simple states channel and only keep the RGB channels
    image = np.moveaxis(image[:3], 0, -1).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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