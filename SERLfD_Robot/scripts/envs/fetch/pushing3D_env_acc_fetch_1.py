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
import math
import glob
import os
import random
import time
import cv2
from absl import logging
# import gin
import gym
from gym import spaces
import numpy as np
from PIL import Image
from six.moves import range
from algorithms.common.load_config_utils import loadYAML
from collections import OrderedDict
import pybullet
# from dql_grasping import kuka
import envs.fetch.fetch_step_acc_3d as fetch
from envs.fetch.transformations import quaternion_to_euler_angle_vectorized2
from scipy.spatial import distance

from envs.utils_geom import quatMult, quatInverse
from kair_algorithms_draft.scripts.envs.fetch.fetch_step_acc_3d import accurateIK, setMotors

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


class fetchPush3DEnv(gym.Env):
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
      test=False,
      random_init=True,):
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
    self._random_init = random_init


    if render_mode == 'GUI':
      self.cid = pybullet.connect(pybullet.GUI)
      pybullet.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
    elif render_mode == 'DIRECT':
      self.cid = pybullet.connect(pybullet.DIRECT)
    elif render_mode == 'TCP':
      self.cid = pybullet.connect(pybullet.TCP, 'localhost', 6667)


    conf_str, self.conf_data = loadYAML(
      "../config/fetch_serl_push3d_env_acc_1.yaml")
    self.get_params()
    print("Call get_obs")
    self.merge_simple_states_to_img = self.conf_data['env']['merge_simple_states_to_img']
    self._init_env_variables()

    # self.gazebo.pauseSim()
    self.gripper_min_dist = self.conf_data['fetch']['gripper_min_dist']
    self.gripper_max_dist = self.conf_data['fetch']['gripper_max_dist']
    self.exe_single_group = self.conf_data['fetch']['exe_single_group']
    # if self.exe_single_group:
    #   self.a_min, self.a_max = np.array([0., 0., 0.25, -0.4, 0.37, -2, -1]), np.array(
    #     [1., 1., 0.75, 0.4, 0.91, 2, 1])
    # else:
    # self.a_min, self.a_max = np.array([-0.05, -0.05, -math.pi/2]), np.array(
    #     [0.05, 0.05, math.pi/2])
    self.a_min, self.a_max = np.array([-0.1, -0.1, -0.06, -math.pi/2]), np.array(
        [0.1, 0.1, 0.06, math.pi/2])
    self.action_space = spaces.Box(self.a_min, self.a_max)
    self.reset()
    obs = self._get_observation()
    self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype='float32')
    self.viewer = None

    files = glob.glob("./*.bullet")

    for f in files:
      try:
        os.remove(f)
      except OSError as e:
        print("Error: %s : %s" % (f, e.strerror))

  def _init_env_variables(self):
      """
      Inits variables needed to be initialized each time we reset at the start
      of an episode.
      :return:
      """
      self.done = False
      self.success = False
      self.finished_prev = 0
      # self.prev_relative_pose = {}
      self.simple_states_prev = OrderedDict()
      # self._update_simple_states()
      # for obj_name in self.target_region:
      #     self.prev_relative_pose[obj_name] = [0] * 6
      self.to_exe_group_action = [0.] * 2
      self.move_arm_result = True
      self.to_execute_group = -1

  def get_params(self):
    self.group_name = self.conf_data['fetch']['group_name']
    self.n_actions = self.conf_data['fetch']['n_actions']
    self.n_max_iterations = self.conf_data['fetch']['max_iterations']
    self.init_pos = self.conf_data['fetch']['init_pos']
    self.init_pos_b = self.conf_data['fetch']['init_pos_base']
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
    self.bi_mapping = {self.predicate_true_value: True, self.predicate_false_value: False}

    self.use_shaping = self.conf_data['env']['use_shaping']

  def setup(self):
    """Sets up the robot + tray + objects.
    """
    test = self._test
    # self._urdf_list = [[local_env_data_path + '/../../models/insert_task_cube2_block4/model.urdf', [0.6, 0.1, 0.42]], [local_env_data_path + '/../../models/insert_task_cylinder2_block3/model.urdf', [0.6, -0.1, 0.42]]]


    self._urdf_list = [[local_env_data_path + '/../../models/insert_task_cube2_block4/model_push.urdf', [0.61, 0., 0.42]], [local_env_data_path + '/../../models/insert_task_cylinder2_block3/model_push.urdf', [0.61, -0.1, 0.42]]]
    self._urdf_list = [random.choice(self._urdf_list)]

    if self._random_init:
      # Object initial x/y/yaw, randomized
      # x_range = 0.04
      x_range = 0.02
      y_range = 0.1
      # obj_x = np.random.uniform(low=0.56 - x_range, high=0.56 + x_range, size=(1,))
      obj_x = np.random.uniform(low=0.58 - x_range, high=0.58 + x_range, size=(1,))
      # obj_x = np.random.uniform(low=0.54 - x_range, high=0.54 + x_range, size=(1,))

      obj_y = np.random.uniform(low=0 - y_range, high=0 + y_range, size=(1,))
      obj_yaw = np.random.uniform(low=-np.pi, high=np.pi, size=(1,))
      # obj_x, obj_y, obj_yaw = np.array([0.57421037]), np.array([0.08562258]), np.array([-0.72180485])
      obj_orn = pybullet.getQuaternionFromEuler([0, 0, obj_yaw])
      print("XXX ", [obj_x, obj_y, obj_yaw])
      self._urdf_list[0][1] = [obj_x, obj_y, 0.42]

    logging.info('urdf_list %s', self._urdf_list)
    pybullet.resetSimulation(physicsClientId=self.cid)
    pybullet.setPhysicsEngineParameter(
        numSolverIterations=150, physicsClientId=self.cid)
    pybullet.setTimeStep(self._time_step, physicsClientId=self.cid)
    pybullet.setGravity(0, 0, -10, physicsClientId=self.cid)
    plane_path = os.path.join(self._urdf_root, 'plane.urdf')
    pybullet.loadURDF(plane_path, [0, 0, 0], physicsClientId=self.cid)
    # table_path = os.path.join(self._urdf_root, 'table_big.urdf')
    table_path = local_env_data_path + '/../../models/table/table_push.urdf'
    a = pybullet.loadURDF(
        table_path, basePosition=[0.8, 0.0, 0.], useFixedBase=True,
        physicsClientId=self.cid)
    textureId = pybullet.loadTexture(local_env_data_path + '/../../models/table/textures/wood.jpeg', physicsClientId=self.cid)
    pybullet.changeVisualShape(a, -1, textureUniqueId=textureId, physicsClientId=self.cid)

    l = np.random.choice([0, 1])
    self.Locs = [0.42, 0, 0.36501] #[0.42, -0.1, 0.3651, 0.42, 0.1, 0.3651]
    BlockLocs = [self.Locs, [0.37, 0, 0.36501]]

    block3_id = pybullet.loadURDF(
      local_env_data_path + '/../../models/insert_task_block4/model.urdf', basePosition=BlockLocs[0], useFixedBase=True,
      physicsClientId=self.cid)
    pybullet.setCollisionFilterGroupMask(block3_id, -1, 0, 0)

    bound_id = pybullet.loadURDF(
      local_env_data_path + '/../../models/insert_task_block4/model_boundary.urdf', basePosition=BlockLocs[1], useFixedBase=True,
      physicsClientId=self.cid)

    # pybullet.loadURDF(
    #   'fetch_description/models/insert_task_cylinder2_block3/model.urdf', basePosition=[0.4, -0.1, 0.43],
    #   physicsClientId=self.cid)
    # textureId = pybullet.loadTexture('/home/yz/Downloads/wood.jpeg', physicsClientId=self.cid)
    # pybullet.changeVisualShape(a, -1, textureUniqueId=textureId, physicsClientId=self.cid)


    self._block_uids = OrderedDict()
    self._block_uids['insert_task_block_1'] = block3_id
    # for urdf_name in self._urdf_list:
      # xpos = self._block_random * random.random()
      # ypos = 0.2 + self._block_random * (random.random() - .5)
      # angle = np.pi / 2 + self._block_random * np.pi * random.random()
      # ori = pybullet.getQuaternionFromEuler([0, 0, angle])
      # uid = pybullet.loadURDF(
      #     urdf_name, [xpos, ypos, .15], [ori[0], ori[1], ori[2], ori[3]],
      #     physicsClientId=self.cid)
      # self._block_uids.append(uid)

    self._used_objs = []
    for model in self._urdf_list:
      urdf_name = model[0]
      if obj_orn:
        uid = pybullet.loadURDF(
          urdf_name, basePosition=model[1],
          baseOrientation=obj_orn,
          physicsClientId=self.cid)
      else:
        uid = pybullet.loadURDF(
          urdf_name, basePosition=model[1],
          physicsClientId=self.cid)
      if 'cube' in urdf_name:
        self._block_uids['cube1'] = uid
        self._used_objs.append('cube1')
      else:
        self._block_uids['cylinder1'] = uid
        self._used_objs.append('cylinder1')


    if 'cube1' in self._block_uids.keys():
      config = pybullet.getAABB(self._block_uids['cube1'])
      print(local_env_data_path + '/../../models/insert_task_cube2_block4/model_cover.urdf')
      if obj_orn:
        cover_id = pybullet.loadURDF(
          local_env_data_path + '/../../models/insert_task_cube2_block4/model_cover.urdf', basePosition=[self._urdf_list[0][1][0], self._urdf_list[0][1][1], config[1][2]+0.015],
          baseOrientation=obj_orn,
          physicsClientId=self.cid)
      else:
        cover_id = pybullet.loadURDF(
          local_env_data_path + '/../../models/insert_task_cube2_block4/model_cover.urdf',
          basePosition=[self._urdf_list[0][1][0], self._urdf_list[0][1][1], config[1][2] + 0.015],
          physicsClientId=self.cid)
      self._block_uids['cube1_cover'] = cover_id
      pybullet.changeDynamics(cover_id, -1, lateralFriction=1.,
                              spinningFriction=0.01,
                              frictionAnchor=1,
                              )

    if 'cylinder1' in self._block_uids.keys():
      config = pybullet.getAABB(self._block_uids['cylinder1'])
      if obj_orn:
        cover_id = pybullet.loadURDF(
          local_env_data_path + '/../../models/insert_task_cylinder2_block3/model_cover.urdf',
          basePosition=[self._urdf_list[0][1][0], self._urdf_list[0][1][1], config[1][2]+0.015],
          baseOrientation=obj_orn,
          physicsClientId=self.cid)
      else:
        cover_id = pybullet.loadURDF(
          local_env_data_path + '/../../models/insert_task_cylinder2_block3/model_cover.urdf',
          basePosition=[self._urdf_list[0][1][0], self._urdf_list[0][1][1], config[1][2] + 0.015],
          physicsClientId=self.cid)

      self._block_uids['cylinder1_cover'] = cover_id
      pybullet.changeDynamics(cover_id, -1, lateralFriction=1.,
                              spinningFriction=0.01,
                              frictionAnchor=1,
                              )



    self._fetch = fetch.fetch(
        urdfRootPath=local_env_data_path,
        timeStep=self._time_step,
        clientId=self.cid,
        ikFix=True)
    for _ in range(500):
      pybullet.stepSimulation(physicsClientId=self.cid)

    self._history_env_shot = []

  def reset(self):
    self._env_step = 0
    self._init_env_variables()

    # self._resets += 1
    # if self._resets % self._num_resets_per_setup == 0:
    self.setup()
    look = [0.65, -0.3, 0.84]
    distance = 3.0
    pitch = -40 + self._cam_random * np.random.uniform(-3, 3)
    yaw = 0 + self._cam_random * np.random.uniform(-3, 3)
    roll = 0
    self._view_matrix = pybullet.computeViewMatrixFromYawPitchRoll(
        look, distance, yaw, pitch, roll, 2)
    fov = 20. + self._cam_random * np.random.uniform(-2, 2)
    aspect = self._width / self._height
    near = 0.1
    far = 10
    self._proj_matrix = pybullet.computeProjectionMatrixFOV(
        fov, aspect, near, far)


    # Let the blocks settle and move arm down into a closer approach pose.
    self._fetch.reset(base_pos=self.init_pos_b, endeffector_pos=self.init_pos)#, fingerDist=self.gripper_min_dist)

    obs = self._get_observation()

    if 'cube1' in self._used_objs:
      if self.simple_states['predicates']["is_cube1_initially_on_left"] == self.predicate_true_value:
        self.done_case = 0
      if self.simple_states['predicates']["is_cube1_initially_on_right"] == self.predicate_true_value:
        self.done_case = 1
    if 'cylinder1' in self._used_objs:
      if self.simple_states['predicates']["is_cylinder1_initially_on_left"] == self.predicate_true_value:
        self.done_case = 2
      if self.simple_states['predicates']["is_cylinder1_initially_on_right"] == self.predicate_true_value:
        self.done_case = 3

    print("done case: ", self.done_case)

    return obs

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
    obs, im_h, im_w = cv2.resize(obs, dsize=(self._downsample_width, self._downsample_height)), self._downsample_height, self._downsample_width  # cv resize parameter: width, height
    obs = np.moveaxis(obs, -1, 0)

    if self.merge_simple_states_to_img:
      simple_states = self._get_simple_observation()

      source = np.zeros(im_h * im_w)

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

      source[:len(simple_states)] = simple_states
      source = np.reshape(source, [im_h, im_w])
      obs = np.concatenate((obs, [source]))

    return obs

  def _update_simple_states(self):
    self.joint_states_now = OrderedDict(zip(self.joints.name, self.joints.position))
    self.robot_base_state_now = self.get_target_objects_state(['fetch'])['fetch']  # self.update_base()
    self.ee_state_now = self.get_ee_pose()
    self.obj_world_states_now = self.get_target_objects_state(self.target_region.keys())
    self.inserted_obj_poses = self.get_target_objects_state(self.inserted_objs.keys())
    self.obj_robot_states_now = self.get_target_objects_state(self.target_region.keys(), 'fetch')
    self.gripper_joint_pose = [self.get_joints()[jn][0] for jn in self.gripper_group]

  def _get_current_predicate_values(self):
    def nearby(entity_a, entity_b, distance, use_z=False):
      """
      Judge if two entities are close enough within a distance.
      """

      def get_entity_position(entity):
        """
        Query gazebo get_model_state service to get the position of an entity.
        An entity can be the robot, an object, or a list position specified by user.
        """
        if isinstance(entity, list):
          return entity
        return self.simple_states[entity][:3]

      entity_a_pos, entity_b_pos = get_entity_position(entity_a), get_entity_position(entity_b)
      if math.hypot(entity_b_pos[0] - entity_a_pos[0], entity_b_pos[1] - entity_a_pos[1]) < distance:
        return self.predicate_true_value
      return self.predicate_false_value

    def within(entity, region):
      """
      Judge if an entity is in a given region.
      """
      if not region[0][0] < self.simple_states[entity][0] < region[0][1]:
        return self.predicate_false_value
      elif not region[1][0] < self.simple_states[entity][1] < region[1][1]:
        return self.predicate_false_value
      elif not region[2][0] < self.simple_states[entity][2] < region[2][1]:
        return self.predicate_false_value

      return self.predicate_true_value

    def pushed(entity):
      curr_ee_world_pose = self.simple_states['ee']

      curr_obj_world_pose = self.simple_states[entity]
      curr_relative_pose = [abs(x - y) for x, y in zip(curr_ee_world_pose, curr_obj_world_pose)]
      # obj_ee_dist = distance.euclidean(curr_obj_world_pose[:3], curr_ee_world_pose[:3])

      # if not self.simple_states_prev:
      if self._env_step == 0:
        return self.predicate_false_value

      pushed_dist = 0.0293 if self._env_step < 3 else 0.005
      # print(entity, curr_relative_pose, self.prev_relative_pose)
      # if np.allclose(a=curr_relative_pose, b=self.prev_relative_pose[entity],
      #                atol=0.4) and not np.allclose(a=self.simple_states_prev[entity], b=self.simple_states[entity], atol=0.01):
      if not np.allclose(a=self.simple_states_prev[entity], b=self.simple_states[entity], atol=pushed_dist):
        # self.prev_relative_pose[entity] = curr_relative_pose
        return self.predicate_true_value

      # self.prev_relative_pose[entity] = curr_relative_pose
      return self.predicate_false_value

    def inserted(entity, target, offsets):
      if not abs(self.simple_states[target][0] - self.simple_states[entity][0]) < offsets[0]:
        return self.predicate_false_value
      elif not abs(self.simple_states[target][1] - self.simple_states[entity][1]) < offsets[1]:
        return self.predicate_false_value
      elif not self.simple_states[entity][2] - self.simple_states[target][2] < offsets[2]:
        return self.predicate_false_value

      return self.predicate_true_value

    def not_opened(entity):
      not_opened = self.predicate_false_value
      entity_cover_config = self.simple_states[entity+'_cover']
      entity = self._block_uids[entity]
      config = pybullet.getAABB(entity)

      height = pybullet.getAABB(entity)[1][2]
      contacts = pybullet.getContactPoints(entity) # Get all contact points of the entity
      if contacts:
        for p in contacts:
          # print("PPP ", entity_cover_config[2], abs(entity_cover_config[3:5]), p[5][2], height-0.004)
          # print(entity_cover_config[2] > height-0.004, np.allclose(entity_cover_config[3:5], [0, 0], atol=0.785), p[5][2] >= height-0.004)
          if entity_cover_config[2] > height-0.004 and np.allclose(entity_cover_config[3:5], [0, 0], atol=0.785) and p[5][2] >= height-0.004: # If the z pose value is >= the entity's height

            not_opened = self.predicate_true_value
            break
      return not_opened

    def relative_orn(entity, target, expected, predicate): # L, R, F, B # TODO Up, Down, Overloapping


      if 'initial' in predicate and self._env_step > 0:
        return self.simple_states_prev['predicates'][predicate]

      possibilities = {'left': 0, 'right': 1, 'front': 2, 'back': 3}
      expected = possibilities[expected]
      left = True
      front = False

      if isinstance(entity, str):
        entity = self.simple_states[entity]
      if isinstance(target, str):
        target = self.simple_states[target]

      if abs(entity[0]) < abs(target[0]):
        front = True
      if entity[1] > target[1]:
        left = False

      result = [[1, 0] if left == True else [0, 1]][0] + [[1, 0] if front == True else [0, 1]][0]
      return self.predicate_true_value if result[expected] == 1 else self.predicate_false_value

    if self._env_step == 0:
      print("1st round")
      self.simple_states_prev = self.simple_states

    predicates_values = [self.predicate_false_value] * len(self.predicates_list)

    for func in self.predicates:
      for pred in self.predicates[func]:
        skip = True
        for o in self._used_objs:
          if o in pred:
            skip = False
        if skip:
          continue
        args = self.predicates[func][pred]
        pred_value = locals()[func](*args)
        # Call the "func" function with arguments
        # https://stackoverflow.com/questions/28372223/python-call-function-from-string
        # https://stackoverflow.com/questions/3941517/converting-list-to-args-when-calling-function

        predicates_values[self.predicates_list.index(pred)] = pred_value

    # self.predicates_values =
    return OrderedDict(zip(self.predicates_list, predicates_values))

  def _get_simple_observation(self):
    """Observations for simplified observation space.

    Returns:
      Numpy array containing location and orientation of nearest block and
      location of end-effector.
    """
    self.simple_states = OrderedDict()

    self.simple_states['locs'] = np.array(self.Locs)
    self.simple_states['insert_task_block_1'] = np.array([-1000]*6)
    self.simple_states['cube1'] = np.array([-1000]*6)
    self.simple_states['cylinder1'] = np.array([-1000]*6)
    self.simple_states['cube1_cover'] = np.array([-1000]*6)
    self.simple_states['cylinder1_cover'] = np.array([-1000]*6)

    state = pybullet.getLinkState(
        self._fetch.fetchUid, self._fetch.fetchEndEffectorIndex,
        physicsClientId=self.cid)
    # end_effector_pos = np.array(state[0])
    # end_effector_ori = pybullet.getEulerFromQuaternion(np.array(state[1]))
    end_effector_pos = np.array(state[4])
    end_effector_ori = pybullet.getEulerFromQuaternion(np.array(state[5]))
    self.simple_states['ee'] = np.concatenate((end_effector_pos, end_effector_ori))

    pos_and_ori = []
    for obj in self._block_uids:
      pos, ori = pybullet.getBasePositionAndOrientation(
          self._block_uids[obj], physicsClientId=self.cid)
      ori = pybullet.getEulerFromQuaternion(ori)
      pos, ori = np.array(pos), np.array(ori)
      self.simple_states[obj] = np.concatenate((pos, ori))
      pos_and_ori.append((pos, ori))
    #   distances.append(np.linalg.norm(end_effector_pos - pos))
    # pos, ori = pos_and_ori[np.argmin(distances)]
    # return np.concatenate((pos, ori, end_effector_pos, end_effector_ori))
    pos_and_ori = np.concatenate(sum([p for p in pos_and_ori], ()))
    predicates = self._get_current_predicate_values()

    # self.curr_simple_states = np.concatenate((self.simple_states['locs'], pos_and_ori, end_effector_pos, end_effector_ori, list(predicates.values())))
    self.curr_simple_states = np.concatenate([self.simple_states[x] for x in self.simple_states] + [list(predicates.values())])
    self.simple_states['predicates'] = predicates
    print(self.simple_states)

    return self.curr_simple_states

  def step(self, action):
    print("action ", action)
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
    # self._fetch.applyAction(np.concatenate((action, [0.0])))
    # PositionNow = self.simple_states['ee'][:3]
    # targetPosition = PositionNow + np.concatenate((action[:2], [self.init_pos[2]]))
    # OrientationNow = self.simple_states['ee'][-1]
    # targetOrientation = np.concatenate((self.init_pos[4:5], OrientationNow + action[2]))
    # jointPoses = accurateIK(self._fetch.fetchUid, self._fetch.fetchEndEffectorIndex,
    #                         targetPosition, pybullet.getQuaternionFromEuler(targetOrientation),
    #                         self._fetch.ll, self._fetch.ul, self._fetch.jr, self._fetch.rp,
    #                         useNullSpace=True)
    # setMotors(self._fetch.fetchUid, jointPoses)

    # self._fetch.move_pos(relative_pos=np.concatenate((action[:2], [0])), relative_local_euler=[0, 0, action[-1]], numSteps=100)#, gripper_target_pos=self.gripper_min_dist/2)
    # self._fetch.move_pos(relative_pos=action[:3], relative_local_euler=[0, 0, action[-1]], numSteps=100)#, gripper_target_pos=self.gripper_min_dist/2)
    self._fetch.move_pos(relative_pos=action[:3], absolute_global_euler=[self.init_pos[3], self.init_pos[4], action[-1]], numSteps=100, gripper_target_pos=0.)

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
    done = self._termination()
    reward = self._reward()

    debug = {
        # 'grasp_success': self._grasp_success
    }
    self.simple_states_prev = self.simple_states
    return observation, reward, done, debug

  def _render(self, mode='human'):
    return

  def _termination(self):
    if self._env_step >= self._max_episode_steps:
      return True

    # Check if current situation is impossible to success
    # 1) The any of target objects drops down on the ground (it's impossible for robot to pick it up)
    for obj_name in self._used_objs:
      if self.simple_states[obj_name][2] <= 0.35:
        print("A target fell down. Impossible to finish the task. Episode ends.")
        return True
      if not np.allclose(a=self.simple_states[obj_name][3:4], b=[0, 0],
                         atol=0.785398): # pi/4
        return True



    for d in self.done_predicates[self.done_case]:
      if not self.bi_mapping[self.simple_states['predicates'][d]] == self.done_predicates[self.done_case][d]:
        return False

    self.success = True
    return True

  def _reward(self):
    finished = 0
    if self.success:
      return self.reached_goal_reward

    # for d in self.done_predicates[self.done_case]:
    #   if self.simple_states['predicates'][d] == self.done_predicates[self.done_case][d] != self.init_predicates[d]:
    #     finished += 1
    #
    # if finished > self.finished_prev:
    #   self.finished_prev = finished
    #   return 25.0
    # else:

    # neg_dist = -distance.euclidean(self.simple_states['cube1'][:2], self.simple_states['insert_task_block_2'][:2])-distance.euclidean(self.simple_states['cylinder1'][:2], self.simple_states['insert_task_block_1'][:2])
    # if self.use_shaping and self.obs_targets_distance + neg_dist > 0:
    #   shape_rwd = 100 * (self.obs_targets_distance + neg_dist)
    #   self.obs_targets_distance = -neg_dist
    #   return shape_rwd
    # else:
    #   return 0
    return 0

  def close_display(self):
    pybullet.disconnect()
    self.cid = pybullet.connect(pybullet.DIRECT)
    self._setup()

  def _get_urdf_path(self, filename):
    """Resolve urdf path of filename."""
    d = os.path.splitext(filename)[0]
    return os.path.join(self._models_dir, d, filename)


  def mouse_control(self, grasp, back_step=1):
    if back_step == 0:
      back_step = 1
    pybullet.removeAllUserParameters()

    state = pybullet.getLinkState(
      self._fetch.fetchUid, self._fetch.fetchEndEffectorIndex)
    actualEndEffectorPos = state[4]
    # actualEndEffectorOrn = pybullet.getEulerFromQuaternion(state[5])
    actualEndEffectorOrn = state[5]
    actualEndEffectorOrnEuler = pybullet.getEulerFromQuaternion(actualEndEffectorOrn)

    print("SSAA: ", [actualEndEffectorPos, actualEndEffectorOrnEuler])

    # targetPosXId = pybullet.addUserDebugParameter("targetPosX", 0., 1.5, actualEndEffectorPos[0])
    # targetPosYId = pybullet.addUserDebugParameter("targetPosY", -1, 1, actualEndEffectorPos[1])
    # targetPosZId = pybullet.addUserDebugParameter("targetPosZ", 0, 1.15, actualEndEffectorPos[2])
    # targetOriRollId = pybullet.addUserDebugParameter("targetOriRoll", -3.15, 3.15, actualEndEffectorOrnEuler[0])
    # targetOriPitchId = pybullet.addUserDebugParameter("targetOriPitch", -3.15, 3.15, actualEndEffectorOrnEuler[1])
    # targetOriYawId = pybullet.addUserDebugParameter("targetOriYaw", -3.15, 3.15, actualEndEffectorOrnEuler[2])
    targetPosXId = pybullet.addUserDebugParameter("targetPosX", actualEndEffectorPos[0]+self.a_min[0], actualEndEffectorPos[0]+self.a_max[0], actualEndEffectorPos[0])
    targetPosYId = pybullet.addUserDebugParameter("targetPosY", actualEndEffectorPos[1]+self.a_min[1], actualEndEffectorPos[1]+self.a_max[1], actualEndEffectorPos[1])
    targetPosZId = pybullet.addUserDebugParameter("targetPosZ", actualEndEffectorPos[2]+self.a_min[2], actualEndEffectorPos[2]+self.a_max[2], actualEndEffectorPos[2])
    targetOriRollId = pybullet.addUserDebugParameter("targetOriRoll", -3.15, 3.15, actualEndEffectorOrnEuler[0])
    targetOriPitchId = pybullet.addUserDebugParameter("targetOriPitch", -3.15, 3.15, actualEndEffectorOrnEuler[1])
    targetOriYawId = pybullet.addUserDebugParameter("targetOriYaw", actualEndEffectorOrnEuler[2]+self.a_min[3], actualEndEffectorOrnEuler[2]+self.a_max[3], actualEndEffectorOrnEuler[2])
    graspId = pybullet.addUserDebugParameter("grasp", 0, 1, grasp)

    time.sleep(1.)

    try:
      while True:
        targetPosX = pybullet.readUserDebugParameter(targetPosXId)
        targetPosY = pybullet.readUserDebugParameter(targetPosYId)
        targetPosZ = pybullet.readUserDebugParameter(targetPosZId)
        targetOriRoll = pybullet.readUserDebugParameter(targetOriRollId)
        targetOriPitch = pybullet.readUserDebugParameter(targetOriPitchId)
        targetOriYaw = pybullet.readUserDebugParameter(targetOriYawId)
        grasp = pybullet.readUserDebugParameter(graspId)

        targetPosition = [targetPosX, targetPosY, targetPosZ]
        targetOrientation = [targetOriRoll, targetOriPitch, targetOriYaw]

        fingleValue = 0.1 if grasp > 0.5 else 0.0
        self._fetch.move_pos(absolute_pos=targetPosition, absolute_global_euler=targetOrientation,
                               gripper_target_pos=fingleValue, mouse=True)

        # jointPoses = accurateIK(self._fetch.fetchUid, self._fetch.fetchEndEffectorIndex,
        # 															 targetPosition, pybullet.getQuaternionFromEuler(targetOrientation),
        # 															 self._fetch.ll, self._fetch.ul, self._fetch.jr, self._fetch.rp,
        # 															 useNullSpace=True)
        # setMotors(self._fetch.fetchUid, jointPoses)
        pybullet.stepSimulation()

    except KeyboardInterrupt:
      pass

    # print("SSBB: ", [ALLtargetPositions['L'], ALLtargetOrientations['L'], ALLtargetPositions['R'], ALLtargetOrientations['R']])
    print("BBBB ", targetOrientation)
    # Calculate pose offsets
    TransEE = np.array(targetPosition) - actualEndEffectorPos
    RotEE = np.array(targetOrientation) - actualEndEffectorOrnEuler #actualEndEffectorOrn
    # Relative rotation should be computed by subtraction!
    # RotEE = quatMult(quatInverse(actualEndEffectorOrn),
    #                  np.array(pybullet.getQuaternionFromEuler(targetOrientation)))  # use relative_local_euler
    # RotEE = quatMult(quatInverse(np.array(p.getQuaternionFromEuler(targetOrientation))), np.array(actualEndEffectorOrn)) # use relative_global_euler

    # RotEE = pybullet.getEulerFromQuaternion(RotEE)

    self.restore(back_step)
    print("AA: ", np.concatenate((TransEE, RotEE, [grasp])))
    return np.concatenate((TransEE, np.array(targetOrientation), [grasp]))

  def save(self, id):
    # p.removeState(self.stateId)
    # self.stateId = p.saveState()
    # self._history_env_shot.append(self.stateId)

    # id %= self._history_max_len
    # print("BB", id)
    # self._history_env_shot.append(id)

    self._history_env_shot.append("state_" + str(id) + ".bullet")

    print("Saving state_" + str(id) + ".bullet")
    pybullet.saveBullet("state_" + str(id) + ".bullet")

  def restore(self, id):
    print("Taking " + self._history_env_shot[-id])
    pybullet.restoreState(fileName=self._history_env_shot[-id])
    pybullet.restoreState(fileName=self._history_env_shot[-id])
    pybullet.restoreState(fileName=self._history_env_shot[-id])