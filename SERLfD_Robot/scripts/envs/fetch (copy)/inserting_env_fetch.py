# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""Simplified grasping environment using PyBullet.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import random
import time
import cv2
from absl import logging
import gin
import gym
from gym import spaces
import numpy as np
from PIL import Image
from six.moves import range
from algorithms.common.load_config_utils import loadYAML
import pybullet
# from dql_grasping import kuka
import envs.fetch.fetch as fetch
from envs.fetch.transformations import quaternion_to_euler_angle_vectorized2

INTERNAL_BULLET_ROOT = None
if INTERNAL_BULLET_ROOT is None:
  import pybullet_data
  OSS_DATA_ROOT = pybullet_data.getDataPath()
else:
  OSS_DATA_ROOT = ''

local_env_data_path = 'envs/fetch/fetch_description'
# OSS_DATA_ROOT = os.getcwd() + '/fetch_simple_simulation/fetch_simple_description/models'

# pylint: enable=bad-import-order
# pylint: enable=g-import-not-at-top


class fetchInsertingEnv(gym.Env):
  """Simplified grasping environment with discrete and continuous actions.
  """

  def __init__(
      self,
      camera_random=0,
      simple_observations=False,
      continuous=False,
      remove_height_hack=False,
      urdf_list=None,
      render_mode='GUI',
      dv=0.06,
      target=False,
      num_resets_per_setup=1,
      render_width=240,
      render_height=240,
      downsample_width=240,
      downsample_height=240,
      test=False,):
    """Creates a fetchGraspingEnv.

    Args:
      block_random: How much randomness to use in positioning blocks.
      camera_random: How much randomness to use in positioning camera.
      simple_observations: If True, observations are the position and
        orientation of end-effector and closest block, rather than images.
      continuous: If True, actions are continuous, else discrete.
      remove_height_hack: If True and continuous is True, add a dz
                          component to action space.
      urdf_list: List of objects to populate the bin with.
      render_mode: GUI, DIRECT, or TCP.
      num_objects: The number of random objects to load.
      dv: Velocity magnitude of cartesian dx, dy, dz actions per time step.
      target: If True, then we receive reward only for grasping one "target"
        object.
      target_filenames: Objects that we want to grasp.
      non_target_filenames: Objects that we dont want to grasp.
      num_resets_per_setup: How many env resets before calling setup again.
      render_width: Width of camera image to render with.
      render_height: Height of camera image to render with.
      downsample_width: Width of image observation.
      downsample_height: Height of image observation.
      test: If True, uses test split of objects.
      allow_duplicate_objects: If True, samples URDFs with replacement.
      max_num_training_models: The number of distinct models to choose from when
        selecting the num_objects placed in the tray for training.
      max_num_test_models: The number of distinct models to choose from when
        selecting the num_objects placed in the tray for testing.
    """
    self.env_name = "fetch_reach_v0"
    self._time_step = 1. / 200.
    self._max_episode_steps = 50

    # Open-source search paths.
    self._urdf_root = OSS_DATA_ROOT
    self._models_dir = os.path.join(self._urdf_root, 'random_urdfs')

    self._action_repeat = 200
    self._env_step = 0
    self._renders = render_mode in ['GUI', 'TCP']
    # Size we render at.
    self._width = render_width
    self._height = render_height
    # Size we downsample to.
    self._downsample_width = downsample_width
    self._downsample_height = downsample_height
    self._target = target
    self._dv = dv
    self._urdf_list = urdf_list
    self._cam_random = camera_random
    self._simple_obs = simple_observations
    self._continuous = continuous
    self._remove_height_hack = remove_height_hack
    self._resets = 0
    self._num_resets_per_setup = num_resets_per_setup
    self._test = test

    if render_mode == 'GUI':
      self.cid = pybullet.connect(pybullet.GUI)
      pybullet.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
    elif render_mode == 'DIRECT':
      self.cid = pybullet.connect(pybullet.DIRECT)
    elif render_mode == 'TCP':
      self.cid = pybullet.connect(pybullet.TCP, 'localhost', 6667)


    conf_str, self.conf_data = loadYAML(
      os.getcwd() + "/../config/fetch_serl_insert_env.yaml")
    self.get_params()
    print("Call get_obs")
    self.merge_simple_states_to_img = self.conf_data['env']['merge_simple_states_to_img']
    self._init_env_variables()
    self.reset()
    obs = self._get_observation()
    # self.gazebo.pauseSim()
    self.gripper_min_dist = self.conf_data['fetch']['gripper_min_dist']
    self.gripper_max_dist = self.conf_data['fetch']['gripper_max_dist']
    self.exe_single_group = self.conf_data['fetch']['exe_single_group']
    # if self.exe_single_group:
    #   self.a_min, self.a_max = np.array([0., 0., 0.25, -0.4, 0.37, -2, -1]), np.array(
    #     [1., 1., 0.75, 0.4, 0.91, 2, 1])
    # else:
    self.a_min, self.a_max = np.array([0.25, -0.4, 0.37, -2, -1]), np.array(
        [0.75, 0.4, 0.91, 2, 1])
    self.action_space = spaces.Box(self.a_min, self.a_max)
    self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')
    self.viewer = None

  def _init_env_variables(self):
      """
      Inits variables needed to be initialized each time we reset at the start
      of an episode.
      :return:
      """
      self.done = False
      self.success = False
      self.prev_relative_pose = {}
      # self._update_simple_states()
      for obj_name in self.target_region:
          self.prev_relative_pose[obj_name] = [0] * 7
      self.to_exe_group_action = [0.] * 2
      self.move_arm_result = True
      self.to_execute_group = -1
      self.previous_predicates_values = [-1.] * len(self.predicates_list)

  def get_params(self):
    self.group_name = self.conf_data['fetch']['group_name']
    self.n_actions = self.conf_data['fetch']['n_actions']
    self.n_max_iterations = self.conf_data['fetch']['max_iterations']
    self.init_pos = self.conf_data['fetch']['init_pos']
    self.n_observations = int(self.conf_data['fetch']['n_observations'])
    self.gripper_max_effort = float(self.conf_data['fetch']['gripper_max_effort'])

    # self.inserted_obj_width = self.conf_data['env']['inserted_obj_width']
    self.reached_goal_reward = self.conf_data['env']['reached_goal_reward']
    self.rendering = self.conf_data['env']['render']
    self.inserted_objs = self.conf_data['env']['inserted_objs']
    self.target_region = self.conf_data['env']['target_region']
    self.predicates = self.conf_data['env']['predicates']
    self.predicates_list = self.conf_data['env']['predicates_list']
    self.done_predicates = self.conf_data['env']['done_predicates']
    self.predicate_true_value = self.conf_data['env']['predicate_true_value']
    self.predicate_false_value = self.conf_data['env']['predicate_false_value']

  def setup(self):
    """Sets up the robot + tray + objects.
    """
    test = self._test
    self._urdf_list = [[local_env_data_path + '/../../models/insert_task_cube2_block4/model.urdf', [0.4, 0.1, 0.43]], [local_env_data_path + '/../../models/insert_task_cylinder2_block3/model.urdf', [0.4, -0.1, 0.43]]]

    if not self._urdf_list:  # Load from procedural random objects.
      if not self._object_filenames:
        self._object_filenames = self._get_random_objects(
            num_objects=self._num_objects,
            test=test,
            replace=self._allow_duplicate_objects,
        )
      self._urdf_list = self._object_filenames
    logging.info('urdf_list %s', self._urdf_list)
    pybullet.resetSimulation(physicsClientId=self.cid)
    pybullet.setPhysicsEngineParameter(
        numSolverIterations=150, physicsClientId=self.cid)
    pybullet.setTimeStep(self._time_step, physicsClientId=self.cid)
    pybullet.setGravity(0, 0, -10, physicsClientId=self.cid)
    plane_path = os.path.join(self._urdf_root, 'plane.urdf')
    pybullet.loadURDF(plane_path, [0, 0, 0], physicsClientId=self.cid)
    # table_path = os.path.join(self._urdf_root, 'table_big.urdf')
    table_path = local_env_data_path + '/../../models/table/table_big.urdf'
    a = pybullet.loadURDF(
        table_path, basePosition=[1.0, 0.0, 0.0], useFixedBase=True,
        physicsClientId=self.cid)
    textureId = pybullet.loadTexture(local_env_data_path + '/../../models/table/textures/Drawing.jpeg', physicsClientId=self.cid)
    pybullet.changeVisualShape(a, -1, textureUniqueId=textureId, physicsClientId=self.cid)

    pybullet.loadURDF(
      local_env_data_path + '/../../models/insert_task_block3/model.urdf', basePosition=[0.55, -0.1, 0.4], useFixedBase=True,
      physicsClientId=self.cid)

    pybullet.loadURDF(
      local_env_data_path + '/../../models/insert_task_block4/model.urdf', basePosition=[0.55, 0.1, 0.4], useFixedBase=True,
      physicsClientId=self.cid)

    # pybullet.loadURDF(
    #   'fetch_description/models/insert_task_cylinder2_block3/model.urdf', basePosition=[0.4, -0.1, 0.43],
    #   physicsClientId=self.cid)
    # textureId = pybullet.loadTexture('/home/yz/Downloads/wood.jpeg', physicsClientId=self.cid)
    # pybullet.changeVisualShape(a, -1, textureUniqueId=textureId, physicsClientId=self.cid)


    self._block_uids = []
    # for urdf_name in self._urdf_list:
      # xpos = self._block_random * random.random()
      # ypos = 0.2 + self._block_random * (random.random() - .5)
      # angle = np.pi / 2 + self._block_random * np.pi * random.random()
      # ori = pybullet.getQuaternionFromEuler([0, 0, angle])
      # uid = pybullet.loadURDF(
      #     urdf_name, [xpos, ypos, .15], [ori[0], ori[1], ori[2], ori[3]],
      #     physicsClientId=self.cid)
      # self._block_uids.append(uid)
    for model in self._urdf_list:
      urdf_name = model[0]
      uid = pybullet.loadURDF(
          urdf_name, basePosition=model[1],
          physicsClientId=self.cid)
      self._block_uids.append(uid)

    self._fetch = fetch.fetch(
        urdfRootPath=local_env_data_path,
        timeStep=self._time_step,
        clientId=self.cid)
    # for _ in range(500):
    #   pybullet.stepSimulation(physicsClientId=self.cid)

    pass

  def reset(self):
    # self._resets += 1
    # if self._resets % self._num_resets_per_setup == 0:
    self.setup()

    self._attempted_grasp = False

    look = [0.65, -0.3, 0.84]
    distance = 3.0
    pitch = -90 #+ self._cam_random * np.random.uniform(-3, 3)
    yaw = 0 #245 + self._cam_random * np.random.uniform(-3, 3)
    roll = 0
    self._view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(
        look, distance, yaw, pitch, roll, 2)
    fov = 20. + self._cam_random * np.random.uniform(-2, 2)
    aspect = self._width / self._height
    near = 0.1
    far = 10
    self._proj_matrix = pybullet.computeProjectionMatrixFOV(
        fov, aspect, near, far)
    self._env_step = 0

    # for i in range(len(self._urdf_list)):
    #   # xpos = self._block_random * random.random()
    #   # ypos = 0.2 + self._block_random * (random.random() - .5)
    #   # # random angle
    #   # angle = np.pi / 2 + self._block_random * np.pi * random.random()
    #   # ori = pybullet.getQuaternionFromEuler([0, 0, angle])
    #   # pybullet.resetBasePositionAndOrientation(
    #   #   self._block_uids[i], [xpos, ypos, .15],
    #   #   [ori[0], ori[1], ori[2], ori[3]],
    #   #   physicsClientId=self.cid)
    #
    #   print("XXX", self._urdf_list[i][1])
    #   pybullet.resetBasePositionAndOrientation(
    #     self._block_uids[i], self._urdf_list[i][1], [0]*4,
    #     physicsClientId=self.cid)
    #   # Let each object fall to the tray individual, to prevent object
    #   # intersection.
    #   for _ in range(500):
    #     pybullet.stepSimulation(physicsClientId=self.cid)

    # Let the blocks settle and move arm down into a closer approach pose.
    self._fetch.reset()
    # note the velocity continues throughout the grasp.
    self._fetch.applyAction([0.8, 0, 0.9, 0, 1])
    for i in range(100):
      pybullet.stepSimulation(physicsClientId=self.cid)
    return self._get_observation()

  def __del__(self):
    pybullet.disconnect(physicsClientId=self.cid)

  def _get_observation(self):
    if self._simple_obs:
      return self._get_simple_observation()
    else:
      return self._get_image_observation()

  def _get_image_observation(self):
    results = pybullet.getCameraImage(width=self._width,
                                      height=self._height,
                                      viewMatrix=self._view_matrix,
                                      projectionMatrix=self._proj_matrix,
                                      physicsClientId=self.cid)
    rgba = results[2]
    np_img_arr = np.reshape(rgba, (self._height, self._width, 4))
    # Extract RGB components only.
    # img = Image.fromarray(np_img_arr[:, :, :3].astype(np.uint8))
    # shape = (self._downsample_width, self._downsample_height)
    # img = img.resize(shape, Image.ANTIALIAS)
    # img = np.array(img)

    obs = np_img_arr[:, :, :3].astype(np.uint8)
    obs, im_h, im_w = cv2.resize(obs, dsize=(240, 180)), 180, 240  # cv resize parameter: width, height
    obs = np.moveaxis(obs, -1, 0)

    if self.merge_simple_states_to_img:
      simple_states = self._get_simple_observation()
      predicates = [-1.] * len(self.predicates_list) # fake
      source = np.zeros(im_h * im_w)

      curr_simple_states = np.concatenate(
        (simple_states, predicates))
      self.curr_simple_states = curr_simple_states
      # if self.exe_single_group:
      #     # 2 + 6 + 7 + 2 + 6 + 3 + 3 + 6 + 6 + 6
      #     curr_simple_states = np.concatenate(
      #         (self.to_exe_group_action, robot_pose, arm_joint_pos, self.gripper_joint_pose,
      #          curr_ee_robot_pose, mugs_poses, inserted_obj_poses, predicates))
      # else:
      #     # 6 + 7 + 2 + 6 + 3 + 3 + 6 + 6 + 6
      #     curr_simple_states = np.concatenate(
      #         (robot_pose, arm_joint_pos, self.gripper_joint_pose,
      #          curr_ee_robot_pose, mugs_poses, inserted_obj_poses, predicates))

      source[:len(curr_simple_states)] = curr_simple_states
      source = np.reshape(source, [im_h, im_w])
      obs = np.concatenate((obs, [source]))

    return obs

  def _update_simple_states(self):
    self.joint_states_now = dict(zip(self.joints.name, self.joints.position))
    self.robot_base_state_now = self.get_target_objects_state(['fetch'])['fetch']  # self.update_base()
    self.ee_state_now = self.get_ee_pose()
    self.obj_world_states_now = self.get_target_objects_state(self.target_region.keys())
    self.inserted_obj_poses = self.get_target_objects_state(self.inserted_objs.keys())
    self.obj_robot_states_now = self.get_target_objects_state(self.target_region.keys(), 'fetch')
    self.gripper_joint_pose = [self.get_joints()[jn][0] for jn in self.gripper_group]

  def _get_simple_observation(self):
    """Observations for simplified observation space.

    Returns:
      Numpy array containing location and orientation of nearest block and
      location of end-effector.
    """
    state = pybullet.getLinkState(
        self._fetch.fetchUid, self._fetch.fetchEndEffectorIndex,
        physicsClientId=self.cid)
    print("RRR", state[0])
    print("EEE", state[4])
    # end_effector_pos = np.array(state[0])
    # end_effector_ori = pybullet.getEulerFromQuaternion(np.array(state[1]))
    end_effector_pos = np.array(state[4])
    end_effector_ori = pybullet.getEulerFromQuaternion(np.array(state[5]))

    distances = []
    pos_and_ori = []
    for uid in self._block_uids:
      pos, ori = pybullet.getBasePositionAndOrientation(
          uid, physicsClientId=self.cid)
      ori = pybullet.getEulerFromQuaternion(ori)
      pos, ori = np.array(pos), np.array(ori)
      pos_and_ori.append((pos, ori))
    #   distances.append(np.linalg.norm(end_effector_pos - pos))
    # pos, ori = pos_and_ori[np.argmin(distances)]
    # return np.concatenate((pos, ori, end_effector_pos, end_effector_ori))
    pos_and_ori = np.concatenate(sum([p for p in pos_and_ori], ()))

    self.curr_simple_states = np.concatenate((pos_and_ori, end_effector_pos, end_effector_ori))
    return self.curr_simple_states

  def step(self, action):
    return self._step_continuous(action)

  def _step_continuous(self, action):
    """Applies a continuous velocity-control action.

    Args:
      action: 5-vector parameterizing XYZ offset, vertical angle offset
      (radians), and grasp angle (radians).
    Returns:
      observation: Next observation.
      reward: Float of the per-step reward as a result of taking the action.
      done: Bool of whether or not the episode has ended.
      debug: Dictionary of extra information provided by environment.
    """
    # Perform commanded action.
    self._env_step += 1
    self._fetch.applyAction(action)

    for _ in range(self._action_repeat):
      pybullet.stepSimulation(physicsClientId=self.cid)
      if self._renders:
        time.sleep(self._time_step)
      if self._termination():
        break

    # # If we are close to the bin, attempt grasp.
    # state = pybullet.getLinkState(self._fetch.fetchUid,
    #                               self._fetch.fetchEndEffectorIndex,
    #                               physicsClientId=self.cid)
    # end_effector_pos = state[0]
    # if end_effector_pos[2] <= 0.05:
    #   finger_angle = 0.3
    #   for _ in range(1000):
    #     grasp_action = [0, 0, 0.001, 0, finger_angle]
    #     self._fetch.applyAction(grasp_action)
    #     pybullet.stepSimulation(physicsClientId=self.cid)
    #     finger_angle -= 0.3/100.
    #     if finger_angle < 0:
    #       finger_angle = 0
    #   self._attempted_grasp = True
    observation = self._get_observation()
    print("SSS ", self.curr_simple_states)
    done = self._termination()
    reward = self._reward()

    debug = {
        # 'grasp_success': self._grasp_success
    }
    return observation, reward, done, debug

  def _render(self, mode='human'):
    return

  def _termination(self):

    return False

  def _reward(self):
    return 0.0
    # def get_continuous_rwd_shapping():
    #   ofs = 0
    #   shaping_rewards = 0.0
    #   sub_task_list = [["is_get_cube1", "is_cube1_insertedTo_block2"],
    #                    ["is_get_cylinder1", "is_cylinder1_insertedTo_block1"]]
    #   simple_states = self.curr_simple_states
    #
    #   targ_robot_states = sum([[h.pose.position.x, h.pose.position.y, h.pose.position.z] for h in
    #                            self.obj_robot_states_now.values()], [])
    #
    #   predicate_value = np.array(self.predicates_values.values())
    #   if 1.0 not in predicate_value:
    #     shaping_rewards = max(
    #       -np.linalg.norm(simple_states[17 - ofs:20 - ofs] - targ_robot_states[:3]),
    #       -np.linalg.norm(simple_states[17 - ofs:20 - ofs] - targ_robot_states[3:6]))
    #   elif 1.0 not in predicate_value[2:]:
    #     for id, st in enumerate(sub_task_list):
    #       if self.predicates_list[np.where(predicate_value == 1.0)[0][0]] == st[0]:
    #         if id == 0:
    #           shaping_rewards = -np.linalg.norm(
    #             simple_states[23 - ofs:26 - ofs] - simple_states[35 - ofs:38 - ofs])
    #         else:
    #           shaping_rewards = -np.linalg.norm(
    #             simple_states[26 - ofs:29 - ofs] - simple_states[29 - ofs:32 - ofs])
    #
    #   elif np.where(predicate_value == 1.0)[0][-1] == 2 or np.where(predicate_value == 1.0)[0][-1] == 4:
    #     shaping_rewards = -np.linalg.norm(
    #       simple_states[17 - ofs:20 - ofs] - targ_robot_states[:3])
    #
    #   elif np.where(predicate_value == 1.0)[0][-1] == 3 or np.where(predicate_value == 1.0)[0][-1] == 5:
    #     shaping_rewards = -np.linalg.norm(
    #       simple_states[17 - ofs:20 - ofs] - targ_robot_states[
    #                                          3:6])  # care about grasping cylinder
    #   return shaping_rewards
    #
    # bi_rs = 0.0
    # if list(self.predicates_values.values()) != self.previous_predicates_values:
    #   bi_rs = 10.0
    # self.previous_predicates_values = list(self.predicates_values.values())
    # if self.success:
    #   return self.reached_goal_reward
    # elif self.to_execute_group == 1 and not self.move_arm_result:
    #   print("arm move failure")
    #   return -0.01 + get_continuous_rwd_shapping() + bi_rs
    # else:
    #   return 0.0 + get_continuous_rwd_shapping() if bi_rs == 0.0 else bi_rs

  def close_display(self):
    pybullet.disconnect()
    self.cid = pybullet.connect(pybullet.DIRECT)
    self._setup()

  def _get_urdf_path(self, filename):
    """Resolve urdf path of filename."""
    d = os.path.splitext(filename)[0]
    return os.path.join(self._models_dir, d, filename)
