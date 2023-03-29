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
from envs.utils_geom import log_rot, quat2rot
import pybullet as p

exclude_joint_ids = [0, ]
# USE_JOINT_IDS = [19, 11, 12, 14, 15, 16, 17, 13, 9, 1, 2, 4, 5, 6, 3]
USE_JOINT_IDS = [10, 11, 12, 13, 14, 15, 16, 18, 19]


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


def accurateIK(robotID, bodyId, endEffectorId, targetPosition, lowerLimits, upperLimits, jointRanges, restPoses,
               useNullSpace=False, maxIter=10, threshold=1e-4):
    """
    Parameters
    ----------
    robotID: int
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

    numJoints = p.getNumJoints(robotID)

    while (not closeEnough and iter < maxIter):
        if useNullSpace:
            jointPoses = p.calculateInverseKinematics(bodyId, endEffectorId, targetPosition,
                                                      lowerLimits=lowerLimits, upperLimits=upperLimits,
                                                      jointRanges=jointRanges,
                                                      restPoses=restPoses)
        else:
            jointPoses = p.calculateInverseKinematics(bodyId, endEffectorId, targetPosition)

        for i in range(numJoints):
            jointInfo = p.getJointInfo(bodyId, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                p.resetJointState(bodyId, i, jointPoses[qIndex - 7])
        ls = p.getLinkState(bodyId, endEffectorId)
        newPos = ls[4]
        diff = [targetPosition[0] - newPos[0], targetPosition[1] - newPos[1], targetPosition[2] - newPos[2]]
        dist2 = np.sqrt((diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]))
        #print("dist2=", dist2)
        closeEnough = (dist2 < threshold)
        iter = iter + 1
   # print("iter=", iter)
    return jointPoses


def setMotors(bodyId, jointPoses):
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
            p.setJointMotorControl2(bodyIndex=bodyId, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                    targetPosition=jointPoses[qIndex - 7])


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
    self.returnPos = returnPos

    self.maxForce = 60.
    self.fingerAForce = 6
    self.fingerBForce = 5.5
    self.fetchLeftFingerJointIndex = USE_JOINT_IDS[-1]
    self.fetchRightFingerJointIndex = USE_JOINT_IDS[-2]
    self.fingerTipForce = 6
    self.useInverseKinematics = 1
    self.useSimulation = 1
    self.useNullSpace = 1
    self.useOrientation = 1
    self.fetchEndEffectorIndex = 17
    #lower limits for null space
    #joint damping coefficents
    self.jd = [.8] * 24
    #print(self.jd)
    self.fetchUid = p.loadURDF(urdfRootPath + '/fetch.urdf', physicsClientId=self.cid, useFixedBase=1, basePosition=[-0.06, 0.0, 0.0])

    self.ll, self.ul, self.jr, self.rp = getJointRanges(self.fetchUid, includeFixed=False)
    self.reset()

  def reset(self,
            base_pos=None,
            endeffector_pos=None):
    """Resets the fetch base and joint positions.

    Args:
      base_pos:  The [x, y, z] position of fetch base.
      endeffector_pos: The [x, y, z] position of the initial endeffector
        position.
    """
    # Default values for the base position and initial endeffector position.
    if base_pos is None:
      base_pos = [-0.1, 0.0, 0.07]
    if endeffector_pos is None:
      endeffector_pos = [0.537, 0.0, 0.5]

    p.resetBasePositionAndOrientation(self.fetchUid,
                                      base_pos,
                                      [0.000000, 0.000000, 0.000000, 1.000000],
                                      physicsClientId=self.cid)





    for i in range (p.getNumJoints(self.fetchUid,physicsClientId=self.cid)):
      jointInfo = p.getJointInfo(self.fetchUid,i,physicsClientId=self.cid)
      # print(i, jointInfo)

    self.numJoints = p.getNumJoints(self.fetchUid,physicsClientId=self.cid)
    self.jointPositions = [0.0] * self.numJoints
    self.jointPositions[4] = 0.9
    self.jointPositions[18:19] = [0.4] * 2
    self.jointPositions[2] = 0.05

    for jointIndex in USE_JOINT_IDS:
      p.resetJointState(self.fetchUid,
                        jointIndex,
                        self.jointPositions[jointIndex],
                        physicsClientId=self.cid)
      if self.useSimulation:
        p.setJointMotorControl2(self.fetchUid,
                                jointIndex,
                                p.POSITION_CONTROL,
                                targetPosition=self.jointPositions[jointIndex],
                                force=self.maxForce,
                                physicsClientId=self.cid)

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

  def applyAction(self, motorCommands):
    # motorCommands = [0.6, 0.3, 0.9, -1.57, 0.5]
    # motorCommands = [0.6, 0.1, 0.8, 0, 0.5]

    if (self.useInverseKinematics):
      dx = motorCommands[0]
      dy = motorCommands[1]
      dz = motorCommands[2]
      yaw = motorCommands[3]
      #print("VVV", dx, dy, dz, da)
      fingerAngle = motorCommands[4]

      state = p.getLinkState(
          self.fetchUid,self.fetchEndEffectorIndex,physicsClientId=self.cid)

      if self.ikFix:
        actualEndEffectorPos = state[0]
        self.endEffectorPos = list(actualEndEffectorPos)
      else:
        actualEndEffectorPos = state[0]

      self.endEffectorPos[0] = self.endEffectorPos[0]+dx
      if self.endEffectorPos[0]>0.74:
        self.endEffectorPos[0]=0.74
      if self.endEffectorPos[0]<0.05:
        self.endEffectorPos[0]=0.05
      self.endEffectorPos[1] = self.endEffectorPos[1]+dy
      if (self.endEffectorPos[1] < -0.43):
        self.endEffectorPos[1]= -0.43
      if (self.endEffectorPos[1] > 0.43):
        self.endEffectorPos[1] = 0.43

      self.endEffectorPos[2] = self.endEffectorPos[2] + dz
      if actualEndEffectorPos[2] < 0.25:
        self.endEffectorPos[2] = 0.25
      if actualEndEffectorPos[2] > 0.4:
        self.endEffectorPos[2] = 0.4 #self.endEffectorPos[2]+0.0001

      # self.endEffectorAngle = self.endEffectorAngle + da
      pos = self.endEffectorPos
      # orn = p.getQuaternionFromEuler([-math.pi, 0, yaw]) # -math.pi,yaw])
      orn = p.getQuaternionFromEuler([0, math.pi/2, yaw]) # -math.pi,yaw])

      if (self.useNullSpace==1):
        if (self.useOrientation==1):
          jointPoses = p.calculateInverseKinematics(
              self.fetchUid, self.fetchEndEffectorIndex, pos,
              orn, self.ll, self.ul, self.jr, self.rp,
              maxNumIterations=1, physicsClientId=self.cid)
        else:
          jointPoses = p.calculateInverseKinematics(
              self.fetchUid, self.fetchEndEffectorIndex, pos, lowerLimits=self.ll,
              upperLimits=self.ul, jointRanges=self.jr,
              restPoses=self.rp, maxNumIterations=1,
              physicsClientId=self.cid)
      else:
        if (self.useOrientation==1):
          if self.ikFix:
            jointPoses = p.calculateInverseKinematics(
                self.fetchUid,self.fetchEndEffectorIndex,
                pos,orn,jointDamping=self.jd,maxNumIterations=50,
                residualThreshold=.001,
                physicsClientId=self.cid)
          else:
            jointPoses = p.calculateInverseKinematics(
                self.fetchUid,self.fetchEndEffectorIndex,
                pos,orn,jointDamping=self.jd,maxNumIterations=5000,
                physicsClientId=self.cid)
        else:
          jointPoses = p.calculateInverseKinematics(
              self.fetchUid,self.fetchEndEffectorIndex, pos,
              maxNumIterations=1, physicsClientId=self.cid)

      if (self.useSimulation):
        setMotors(self.fetchUid, jointPoses)
      else:
        # Reset the joint state (ignoring all dynamics, not recommended to use
        # during simulation).

        # TODO(b/72742371) Figure out why if useSimulation = 0,
        # len(jointPoses) = 12 and self.numJoints = 14.
        for i in USE_JOINT_IDS[:-1]: #range(len(jointPoses)):
          p.resetJointState(self.fetchUid, i, jointPoses[i],
                            physicsClientId=self.cid)
      # Move fingers.
      self.applyFingerAngle(fingerAngle)


    else:
      for action in range (len(motorCommands)):
        motor = self.motorIndices[action]
        p.setJointMotorControl2(
            self.fetchUid, motor, p.POSITION_CONTROL,
            targetPosition=motorCommands[action], force=self.maxForce,
            physicsClientId=self.cid)
    if self.returnPos:
      # Return the target position for metrics later.
      return pos

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

    jointPoses = self.get_arm_joints() + [0, 0]  # add fingers
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
    jac_sp = full_jacob_pb(jac_t, jac_r)[:, :7]  # 6x10 -> 6x7, ignore last three columns

    try:
      jointDot = np.linalg.pinv(jac_sp).dot(
        (np.hstack((posGain * eePosError, velGain * eeOrnError)).reshape(6, 1)))  # pseudo-inverse
    except np.linalg.LinAlgError:
      jointDot = np.zeros((7, 1))

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
               timeStep=0):

    # Get trajectory
    eePosNow, eeQuatNow = self.get_ee()

    # Determine target pos
    if absolute_pos is not None:
      targetPos = absolute_pos
    elif relative_pos is not None:
      targetPos = eePosNow + relative_pos
    else:
      targetPos = eePosNow

    # Determine target orn
    if absolute_global_euler is not None:
      targetOrn = p.getQuaternionFromEuler(absolute_global_euler)  # euler2quat(absolute_global_euler)
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

    # Get trajectory
    trajPos = self.traj_time_scaling(startPos=eePosNow,
                                     endPos=targetPos,
                                     numSteps=numSteps)

    # Run steps
    numSteps = len(trajPos)
    for step in range(numSteps):

      # Get joint velocities from error tracking control
      jointDot = self.traj_tracking_vel(targetPos=trajPos[step], targetQuat=targetOrn)

      # Send velocity commands to joints
      for i in USE_JOINT_IDS:
        p.setJointMotorControl2(self.fetchUid,
                                i,
                                p.VELOCITY_CONTROL,
                                targetVelocity=jointDot[i],
                                force=self.maxForce,
                                maxVelocity=maxJointVel)

      if gripper_target_pos is None:
        # Keep gripper current velocity
        p.setJointMotorControl2(self.fetchUid,
                                self.fetchLeftFingerJointIndex,
                                p.VELOCITY_CONTROL,
                                targetVelocity=self.fetchUid.fingerCurVel,
                                force=self.maxForce,
                                maxVelocity=0.04)
        p.setJointMotorControl2(self.fetchUid,
                                self.fetchRightFingerJointIndex,
                                p.VELOCITY_CONTROL,
                                targetVelocity=self.fetchUid.fingerCurVel,
                                force=self.maxForce,
                                maxVelocity=0.04)
      else:
        p.setJointMotorControl2(self.fetchUid,
                                self.fetchUid.pandaLeftFingerJointIndex,
                                p.POSITION_CONTROL,
                                targetPosition=gripper_target_pos,
                                maxVelocity=0.04)
        p.setJointMotorControl2(self.fetchUid, 
                                self.fetchUid.pandaRightFingerJointIndex,
                                p.POSITION_CONTROL,
                                targetPosition=gripper_target_pos,
                                maxVelocity=0.04)

      # Step simulation
      p.stepSimulation()
      timeStep += 1

    return timeStep, True