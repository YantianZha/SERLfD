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
# fork of //third_party/bullet/examples/pybullet/gym/envs/bullet/kuka.py with
# fast reset capability.
# pylint: skip-file

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math
import os

from absl import logging
import numpy as np
from six.moves import range
from envs.utils_geom import log_rot, quat2rot, quatMult, euler2quat
import pybullet as p

exclude_joint_ids = [0, ]
# USE_JOINT_IDS = [19, 11, 12, 14, 15, 16, 17, 13, 9, 1, 2, 4, 5, 6, 3]
USE_JOINT_IDS = [3, 12, 13, 14, 15, 16, 17, 18, 20, 21]
# USE_JOINT_IDS = [10, 11, 12, 13, 14, 15, 16, 18, 19]
USE_JOINT_NAMES = ['torso_lift_joint', 'shoulder_pan_joint', 'shoulder_lift_joint', 'upperarm_roll_joint', 'elbow_flex_joint', 'forearm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint', 'r_gripper_finger_joint', 'l_gripper_finger_joint']
MAX_JOINT_FORCES = [450, 33.82, 131.76, 76.94, 66.18, 29.35, 25.70, 7.36, 60, 60]
# MAX_JOINT_FORCES = [87, 87, 87, 87, 87, 12, 12, 12, 60, 60]
MAX_JOINT_VELS = [0.1, 1.25, 1.45, 1.57, 1.52, 1.57, 2.26, 2.26, 0.05, 0.05]
# https://docs.fetchrobotics.com/robot_hardware.html

def full_jacob_pb(jac_t, jac_r):
  return np.vstack((jac_t[0], jac_t[1], jac_t[2], jac_r[0], jac_r[1], jac_r[2]))

def getJointRanges(bodyId, includeFixed=False):
    """
    Parameters
    ----------
    bodyId : int
    includeFixed : bool

    Returns
    -------
    lowerLimits : [ float ] * numDofs
    upperLimits : [ float ] * numDofs
    jointRanges : [ float ] * numDofs
    restPoses : [ float ] * numDofs
    """

    lowerLimits, upperLimits, jointRanges, restPoses = [], [], [], []

    numJoints = p.getNumJoints(bodyId)

    for i in range(numJoints):
        jointInfo = p.getJointInfo(bodyId, i)

        if includeFixed or jointInfo[3] > -1:
            ll, ul = jointInfo[8:10]
            jr = ul - ll

            # For simplicity, assume resting state == initial state
            rp = p.getJointState(bodyId, i)[0]

            lowerLimits.append(-2)
            upperLimits.append(2)
            jointRanges.append(2)
            restPoses.append(rp)

    return lowerLimits, upperLimits, jointRanges, restPoses

def accurateIK(bodyId, endEffectorId, targetPosition, targetOrientation, lowerLimits, upperLimits, jointRanges, restPoses, useNullSpace=False, maxIter=10, threshold=1e-4, cid=0):
  """
  Parameters
  ----------
  bodyId : int
  endEffectorId : int
  targetPosition : [float, float, float]
  lowerLimits : [float]
  upperLimits : [float]
  jointRanges : [float]
  restPoses : [float]
  useNullSpace : bool
  maxIter : int
  threshold : float

  Returns
  -------
  jointPoses : [float] * numDofs
  """
  closeEnough = False
  iter = 0
  dist2 = 1e30

  numJoints = p.getNumJoints(bodyId)

  stateId = p.saveState()

  while (not closeEnough and iter < maxIter):
    if useNullSpace:
      jointPoses = p.calculateInverseKinematics(bodyId, endEffectorId, targetPosition, targetOrientation,
                                                lowerLimits=lowerLimits, upperLimits=upperLimits,
                                                jointRanges=jointRanges,
                                                restPoses=restPoses, maxNumIterations=1, physicsClientId=cid)
    else:
      jointPoses = p.calculateInverseKinematics(bodyId, endEffectorId, targetPosition, targetOrientation)

    for i in range(numJoints):
      jointInfo = p.getJointInfo(bodyId, i)
      qIndex = jointInfo[3]
      if qIndex > -1:
        p.resetJointState(bodyId, i, jointPoses[qIndex - 7])
    ls = p.getLinkState(bodyId, endEffectorId)
    newPos = ls[4]
    newOri = ls[5]

    diff = [targetPosition[0] - newPos[0], targetPosition[1] - newPos[1], targetPosition[2] - newPos[2],
            targetOrientation[0] - newOri[0], targetOrientation[1] - newOri[1], targetOrientation[2] - newOri[2],
            targetOrientation[3] - newOri[3]]
    dist2 = np.sqrt((diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2] + diff[3] * diff[3] + diff[4] * diff[
      4] + diff[5] * diff[5] + diff[6] * diff[6]))
    # print("dist2=", dist2)
    closeEnough = (dist2 < threshold)
    iter = iter + 1
  # # print("iter=", iter)
  p.restoreState(stateId)
  p.removeState(stateId)
  return jointPoses


def setMotors(bodyId, jointPoses, control=True):
    """
    Parameters
    ----------
    bodyId : int
    jointPoses : [float] * numDofs
    """
    numJoints = p.getNumJoints(bodyId)

    for i in range(numJoints):
        jointInfo = p.getJointInfo(bodyId, i)
        # print(jointInfo)
        qIndex = jointInfo[3]
        if qIndex > -1:
            if control:
                p.setJointMotorControl2(bodyIndex=bodyId, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                        targetPosition=jointPoses[qIndex - 7])
            else:
                p.resetJointState(bodyId, i, jointPoses[qIndex - 7])


class fetch:

  def __init__(self, urdfRootPath='', timeStep=0.01, clientId=0, ikFix=False,
               returnPos=True):
    """Creates a fetch robot.

    Args:
      urdfRootPath: The path to the root URDF directory.
      timeStep: The Pybullet timestep to use for simulation.
      clientId: The Pybullet client's ID.
      ikFix: A boolean for whether to apply the ikFix for control. This includes
        increase IK solver iterations, using intertial frame position instead
        of center of mass, and better tracking actual EEF pose. The old
        experiment results did not have these fixes.
      returnPos: A boolean for whether to return commanded EEF position.
    """
    self.cid = clientId
    self.urdfRootPath = urdfRootPath
    self.timeStep = timeStep
    self.ikFix = ikFix
    self.ikIter = 100
    self.returnPos = returnPos

    self.maxForce = 60.
    self.maxJointVel = 0.2
    self.fingerAForce = 6
    self.fingerBForce = 6
    self.fetchLeftFingerJointIndex = USE_JOINT_IDS[-1]
    self.fetchRightFingerJointIndex = USE_JOINT_IDS[-2]
    self.fingerTipForce = 6
    self.useInverseKinematics = 1
    self.useSimulation = 1
    self.useNullSpace = 1
    self.useOrientation = 1
    self.fetchEndEffectorIndex = 19
    #lower limits for null space
    #joint damping coefficents
    self.jd = [.8] * 24
    #print(self.jd)
    self.fetchUid = p.loadURDF(urdfRootPath + '/fetch.urdf', physicsClientId=self.cid, useFixedBase=1, basePosition=[-0.06, 0.0, 0.0])

    self.ll, self.ul, self.jr, self.rp = getJointRanges(self.fetchUid, includeFixed=False)
    # self.reset()

  def reset(self,
            base_pos=None,
            endeffector_pos=None,
            fingerDist=None):
    """Resets the fetch base and joint positions.

    Args:
      base_pos:  The [x, y, z] position of fetch base.
      endeffector_pos: The [x, y, z] position of the initial endeffector
        position.
    """
    # Default values for the base position and initial endeffector position.
    if base_pos is None:
      base_pos = [-0.1, 0.0, 0.07]

    p.resetBasePositionAndOrientation(self.fetchUid,
                                      base_pos,
                                      [0.000000, 0.000000, 0.000000, 1.000000],
                                      physicsClientId=self.cid)

    for i in range (p.getNumJoints(self.fetchUid,physicsClientId=self.cid)):
      jointInfo = p.getJointInfo(self.fetchUid,i,physicsClientId=self.cid)
      # print(i, jointInfo)

    self.numJoints = p.getNumJoints(self.fetchUid,physicsClientId=self.cid)
    InitjointPoses = [0.0] * self.numJoints

    # for _ in range(5):
    if endeffector_pos is None:
      # InitjointPoses[4] = 0.9
      InitjointPoses[20:21] = [0.] * 2
      InitjointPoses[3] = 0.05
    else:
      orn = p.getQuaternionFromEuler(endeffector_pos[3:]) if len(endeffector_pos) < 7 else endeffector_pos[3:]
      InitjointPoses = list(accurateIK(self.fetchUid,
                                                         self.fetchEndEffectorIndex,
                                                         endeffector_pos[:3],
                                                         orn,
                                                         lowerLimits=self.ll,
                                                         upperLimits=self.ul,
                                                         jointRanges=self.jr,
                                                         restPoses=self.rp,
                                                         useNullSpace=True,
                                                         maxIter=self.ikIter))
      # InitjointPoses = list(p.calculateInverseKinematics(self.fetchUid,
      #                                                self.fetchEndEffectorIndex,
      #                                                endeffector_pos[:3],
      #                                                orn,
      #                                                jointDamping=self.jd,
      #                                                lowerLimits=self.ll,
      #                                                upperLimits=self.ul,
      #                                                jointRanges=self.jr,
      #                                                restPoses=self.rp,
      #                                                residualThreshold=1e-4))

      # InitjointPoses[20:21] = fingerPos

      posNow = p.getLinkState(self.fetchUid, self.fetchEndEffectorIndex, physicsClientId=self.cid)

    setMotors(self.fetchUid, InitjointPoses, control=False)
    # self.applyFingerAngle(0.06)

    # def reset_arm(self, endeffector_pos=None):


    # for jointIndex in USE_JOINT_IDS:
    #   p.resetJointState(self.fetchUid,
    #                     jointIndex,
    #                     InitjointPoses[jointIndex],
    #                     physicsClientId=self.cid)
    #
    #   if self.useSimulation:
    #     p.setJointMotorControl2(self.fetchUid,
    #                             jointIndex,
    #                             p.POSITION_CONTROL,
    #                             targetPosition=InitjointPoses[jointIndex],
    #                             force=self.maxForce,
    #                             physicsClientId=self.cid)

    # Set the endeffector height to endEffectorPos.
    self.endEffectorPos = endeffector_pos

    self.endEffectorAngle = 0

    self.motorNames = []
    self.motorIndices = []

    pass

    for i in USE_JOINT_IDS:
      jointInfo = p.getJointInfo(self.fetchUid, i, physicsClientId=self.cid)
      qIndex = jointInfo[3]
      if qIndex > -1:
        self.motorNames.append(str(jointInfo[1]))
        self.motorIndices.append(i)

    pass

  def getActionDimension(self):
    if (self.useInverseKinematics):
      return len(self.motorIndices)
    return 6 # Position x,y,z and roll/pitch/yaw euler angles of end effector.

  def getObservationDimension(self):
    return len(self.getObservation())

  def getObservation(self):
    observation = []
    state = p.getLinkState(
        self.fetchUid, self.fetchEndEffectorIndex, physicsClientId=self.cid)
    if self.ikFix:
      # state[0] is the linkWorldPosition, the center of mass of the link.
      # However, the IK solver uses localInertialFrameOrientation, the inertial
      # center of the link. So, we should use state[4] and not state[0].
      pos = state[4]
    else:
      pos = state[0]
    orn = state[1]

    observation.extend(list(pos))
    observation.extend(list(orn))

    return observation

  def applyFingerAngle(self, fingerAngle):
    # TODO(ejang) - replace with pybullet.setJointMotorControlArray (more efficient).

    p.setJointMotorControl2(
        self.fetchUid, self.fetchLeftFingerJointIndex, p.POSITION_CONTROL, # 'gripper_l_joint' (finger l)
        targetPosition=-fingerAngle, force=self.fingerAForce,
        physicsClientId=self.cid)
    p.setJointMotorControl2(
        self.fetchUid, self.fetchRightFingerJointIndex, p.POSITION_CONTROL, # 'gripper_l_joint_m' (finger r)
        targetPosition=fingerAngle, force=self.fingerBForce,
        physicsClientId=self.cid)


  def get_arm_joints(self):  # use list
    info = p.getJointStates(self.fetchUid, USE_JOINT_IDS[:-2])
    angles = [x[0] for x in info]
    return angles

  def get_ee(self):
    info = p.getLinkState(self.fetchUid, self.fetchEndEffectorIndex)
    return np.array(info[4]), np.array(info[5])

  def traj_time_scaling(self, startPos, endPos, numSteps):
    trajPos = np.zeros((numSteps, 3))
    for step in range(numSteps):
      s = 3 * (1.0 * step / numSteps) ** 2 - 2 * (1.0 * step / numSteps) ** 3
      trajPos[step] = (endPos - startPos) * s + startPos
    return trajPos

  def traj_tracking_vel(self, targetPos, targetQuat, posGain=20, velGain=5):
    eePos, eeQuat = self.get_ee()

    eePosError = targetPos - eePos
    eeOrnError = log_rot(quat2rot(targetQuat).dot((quat2rot(eeQuat).T)))  # in spatial frame

    jointPoses = [0., 0.] + self.get_arm_joints() + [0., 0.]  # add fingers
    eeState = p.getLinkState(self.fetchUid,
                             self.fetchEndEffectorIndex,
                             computeLinkVelocity=1,
                             computeForwardKinematics=1)
    # Get the Jacobians for the CoM of the end-effector link. Note that in this example com_rot = identity, and we would need to use com_rot.T * com_trn. The localPosition is always defined in terms of the link frame coordinates.
    zero_vec = [0.0] * len(jointPoses)
    jac_t, jac_r = p.calculateJacobian(self.fetchUid,
                                       self.fetchEndEffectorIndex,
                                       eeState[2],
                                       jointPoses,
                                       zero_vec,
                                       zero_vec)  # use localInertialFrameOrientation
    jac_sp = full_jacob_pb(jac_t, jac_r)[:, 2:10]  # 6x12 -> 6x8, ignore last three columns

    try:
      jointDot = np.linalg.pinv(jac_sp).dot(
        (np.hstack((posGain * eePosError, velGain * eeOrnError)).reshape(6, 1)))  # pseudo-inverse
    except np.linalg.LinAlgError:
      jointDot = np.zeros((8, 1))

    return jointDot

  def move_pos(self, absolute_pos=None,
               relative_pos=None,
               absolute_global_euler=None,  # preferred
               relative_global_euler=None,  # preferred
               relative_local_euler=None,  # not using
               absolute_global_quat=None,  # preferred
               relative_azi=None,  # for arm
               #    relative_quat=None,  # never use relative quat
               numSteps=50,
               maxJointVel=0.20,
               relativePos=True,
               globalOrn=True,
               checkContact=False,
               checkPalmContact=False,
               objId=None,
               gripper_target_pos=None,
               timeStep=0,
               mouse=False):

    ikIter = 10 if mouse else self.ikIter

    # Get trajectory
    eePosNow, eeQuatNow = self.get_ee()
    # state = p.getLinkState(
    #   self.fetchUid, self.fetchEndEffectorIndex, physicsClientId=self.cid)
    # eePosNow = np.array(state[0])
    # eeQuatNow = np.array(state[1])

    # Determine target pos
    if absolute_pos is not None:
      targetPos = absolute_pos
    elif relative_pos is not None:
      targetPos = eePosNow + relative_pos
    else:
      targetPos = eePosNow

    # Determine target orn
    if absolute_global_euler is not None:
      targetOrn = p.getQuaternionFromEuler(absolute_global_euler, physicsClientId=self.cid)  # euler2quat(absolute_global_euler)
    elif relative_global_euler is not None:
      targetOrn = quatMult(euler2quat(relative_global_euler), eeQuatNow)
    elif relative_local_euler is not None:
      targetOrn = quatMult(eeQuatNow, np.array(p.getQuaternionFromEuler(relative_local_euler)))
    elif absolute_global_quat is not None:
      targetOrn = absolute_global_quat
    elif relative_azi is not None:
      # Extrinsic yaw
      targetOrn = quatMult(euler2quat([relative_azi[0], 0, 0]), eeQuatNow)
      # Intrinsic pitch
      targetOrn = quatMult(targetOrn, euler2quat([0, relative_azi[1], 0]))
    # elif relative_quat is not None:
    # 	targetOrn = quatMult(eeQuatNow, relative_quat)
    else:
      targetOrn = np.array([1.0, 0., 0., 0.])


    self.endEffectorPos = targetPos

    if self.endEffectorPos[0]>0.7:
      self.endEffectorPos[0]=0.7
    if self.endEffectorPos[0]<0.41:
      self.endEffectorPos[0]=0.41
    self.endEffectorPos[1] = self.endEffectorPos[1]
    if (self.endEffectorPos[1] < -0.24):
      self.endEffectorPos[1]= -0.24
    if (self.endEffectorPos[1] > 0.24):
      self.endEffectorPos[1] = 0.24

    self.endEffectorPos[2] = 0.42


    jointPoses = list(accurateIK(self.fetchUid,
                                     self.fetchEndEffectorIndex,
                                     self.endEffectorPos,
                                     targetOrn,
                                     lowerLimits=self.ll,
                                     upperLimits=self.ul,
                                     jointRanges=self.jr,
                                     restPoses=self.rp,
                                     useNullSpace=True,
                                     maxIter=ikIter))
    setMotors(self.fetchUid, jointPoses, control=True)

    if gripper_target_pos:
      # self.applyFingerAngle(gripper_target_pos)
      p.setJointMotorControl2(self.fetchUid,
                              self.fetchLeftFingerJointIndex,
                              p.POSITION_CONTROL,
                              targetPosition=gripper_target_pos,
                              maxVelocity=0.5) # 0.04 should be larger to make the fingers fully openned
      p.setJointMotorControl2(self.fetchUid,
                              self.fetchRightFingerJointIndex,
                              p.POSITION_CONTROL,
                              targetPosition=gripper_target_pos,
                              maxVelocity=0.5)

    # # Get trajectory
    # trajPos = self.traj_time_scaling(startPos=eePosNow,
    #                                  endPos=targetPos,
    #                                  numSteps=numSteps)
    #
    # # Run steps
    # numSteps = len(trajPos)
    # for step in range(numSteps):
    #
    #   # Get joint velocities from error tracking control
    #   # Be careful at velGain value if it's too high: https://robotics.stackexchange.com/questions/6617/jacobian-based-trajectory-following
    #   jointDot = self.traj_tracking_vel(targetPos=trajPos[step], targetQuat=targetOrn, velGain=2)
    #
    #   # Send velocity commands to joints
    #   for i in range(len(USE_JOINT_IDS)-2):
    #     p.setJointMotorControl2(self.fetchUid,
    #                             USE_JOINT_IDS[i],
    #                             p.VELOCITY_CONTROL,
    #                             targetVelocity=jointDot[i],
    #                             force=MAX_JOINT_FORCES[i],
    #                             maxVelocity=MAX_JOINT_VELS[i])
    #
    #   if gripper_target_pos:
    #     # self.applyFingerAngle(gripper_target_pos)
    #     p.setJointMotorControl2(self.fetchUid,
    #                             self.fetchLeftFingerJointIndex,
    #                             p.POSITION_CONTROL,
    #                             targetPosition=gripper_target_pos,
    #                             maxVelocity=0.1) # 0.04 should be larger to make the fingers fully openned
    #     p.setJointMotorControl2(self.fetchUid,
    #                             self.fetchRightFingerJointIndex,
    #                             p.POSITION_CONTROL,
    #                             targetPosition=gripper_target_pos,
    #                             maxVelocity=0.1)
    #
    #   # Step simulation
    #   p.stepSimulation()
    #   timeStep += 1
    #
    # return timeStep, True