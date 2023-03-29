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

import pybullet as p

exclude_joint_ids = [0, ]
# USE_JOINT_IDS = [19, 11, 12, 14, 15, 16, 17, 13, 9, 1, 2, 4, 5, 6, 3]
USE_JOINT_IDS = [10, 11, 12, 13, 14, 15, 16, 18, 19]


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

    self.maxForce = 200.
    self.fingerAForce = 6
    self.fingerBForce = 5.5
    self.fingerTipForce = 6
    self.useInverseKinematics = 1
    self.useSimulation = 1
    self.useNullSpace = 1
    self.useOrientation = 1
    self.fetchEndEffectorIndex = 16
    #lower limits for null space
    #joint damping coefficents
    self.jd = [.8] * 24
    #print(self.jd)
    self.fetchUid = p.loadURDF(urdfRootPath + '/robots/fetch.urdf', physicsClientId=self.cid, useFixedBase=1, basePosition=[-0.05, 0.0, 0.0])

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
      print(i, jointInfo)

    self.numJoints = p.getNumJoints(self.fetchUid,physicsClientId=self.cid)
    self.jointPositions = [0.0] * self.numJoints
    self.jointPositions[4] = 0.9
    self.jointPositions[18:19] = [0.4] * 2

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
        self.fetchUid, USE_JOINT_IDS[-1], p.POSITION_CONTROL, # 'gripper_l_joint' (finger l)
        targetPosition=fingerAngle, force=self.fingerAForce,
        physicsClientId=self.cid)
    p.setJointMotorControl2(
        self.fetchUid, USE_JOINT_IDS[-2], p.POSITION_CONTROL, # 'gripper_l_joint_m' (finger r)
        targetPosition=fingerAngle, force=self.fingerBForce,
        physicsClientId=self.cid)

  def applyAction(self, motorCommands):
    pos = None
    motorCommands = [0.6, 0.3, 0.9, 0, 1]
    print("XXX", motorCommands)
    if (self.useInverseKinematics):

      # dx = motorCommands[0]
      # dy = motorCommands[1]
      # dz = motorCommands[2]
      # da = motorCommands[3]
      # #print("VVV", dx, dy, dz, da)
      # fingerAngle = motorCommands[4]
      #
      # state = p.getLinkState(
      #     self.fetchUid,self.fetchEndEffectorIndex,physicsClientId=self.cid)
      #
      # if self.ikFix:
      #   actualEndEffectorPos = state[4]
      #   self.endEffectorPos = list(actualEndEffectorPos)
      # else:
      #   actualEndEffectorPos = state[0]

      # self.endEffectorPos[0] = self.endEffectorPos[0]+dx
      # if (self.endEffectorPos[0]>0.35):
      #   self.endEffectorPos[0]=0.35
      # if (self.endEffectorPos[0]<0.05):
      #   self.endEffectorPos[0]=0.05
      # self.endEffectorPos[1] = self.endEffectorPos[1]+dy
      # if (self.endEffectorPos[1]<-0.02):
      #   self.endEffectorPos[1]=-0.02
      # if (self.endEffectorPos[1]>0.42):
      #   self.endEffectorPos[1]=0.42

      # if (dz>0 or actualEndEffectorPos[2]>0.05):
      #   self.endEffectorPos[2] = self.endEffectorPos[2]+dz
      # if (actualEndEffectorPos[2]<0.05):
      #   self.endEffectorPos[2] = self.endEffectorPos[2]+0.0001

      # self.endEffectorAngle = self.endEffectorAngle + da
      pos = motorCommands[:3] #self.endEffectorPos
      orn = p.getQuaternionFromEuler([0, 0, motorCommands[3]]) # -math.pi,yaw])
      fingerAngle = 0.08 if motorCommands[-1] > 0 else 0.03

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
