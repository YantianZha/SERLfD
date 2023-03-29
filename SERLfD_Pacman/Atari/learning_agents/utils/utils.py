import numpy as np
import time

from learning_agents.controller_policy.base_policy import TRAJECTORY_INFO_INDEX
from utils.trajectory_utils import TRAJECTORY_INDEX

from addict import Dict


class ConfigDict(Dict):
    def __missing__(self, name):
        raise KeyError(name)

    def __getattr__(self, name):
        try:
            value = super(ConfigDict, self).__getattr__(name)
        except KeyError:
            ex = AttributeError(
                "'{}' object has no attribute '{}'".format(
                    self.__class__.__name__, name
                )
            )
        except Exception as e:
            ex = e
        else:
            return value
        raise ex

    def __setitem__(self, name, value):
        if isinstance(value, dict):
            value = ConfigDict(value)

        super(ConfigDict, self).__setitem__(name, value)


class UtilityTransition:
    def __init__(self, predicate_value, next_predicate_value):
        self.util_transition = {TRAJECTORY_INDEX.NEXT_PREDICATE_VALUES.value: next_predicate_value,
                                TRAJECTORY_INDEX.PREDICATE_VALUES.value: predicate_value}

    def set_transition(self, predicate_value, next_predicate_value):
        self.util_transition = {TRAJECTORY_INDEX.NEXT_PREDICATE_VALUES.value: next_predicate_value,
                                TRAJECTORY_INDEX.PREDICATE_VALUES.value: predicate_value}


class IndexedTraj:
    def __init__(self):
        self.traj_dict = {TRAJECTORY_INDEX.STATE.value: [],
                          TRAJECTORY_INDEX.ACTION.value: [],
                          TRAJECTORY_INDEX.REWARD.value: [],
                          TRAJECTORY_INDEX.NEXT_STATE.value: [],
                          TRAJECTORY_INDEX.DONE.value: []
                          }

    def add_transition(self, transition, predicate_values=None, next_predicate_values=None,
                       utility_map=None, utility_values=None):
        """
        :param transition: list in the format of (state, action, reward, next_state, done)
        """
        self.traj_dict[TRAJECTORY_INDEX.STATE.value].append(transition[0])
        self.traj_dict[TRAJECTORY_INDEX.ACTION.value].append(transition[1])
        self.traj_dict[TRAJECTORY_INDEX.REWARD.value].append(transition[2])
        self.traj_dict[TRAJECTORY_INDEX.NEXT_STATE.value].append(transition[3])
        self.traj_dict[TRAJECTORY_INDEX.DONE.value].append(transition[4])

        if predicate_values is not None:
            if TRAJECTORY_INDEX.PREDICATE_VALUES.value not in self.traj_dict:
                self.traj_dict[TRAJECTORY_INDEX.PREDICATE_VALUES.value] = []
            self.traj_dict[TRAJECTORY_INDEX.PREDICATE_VALUES.value].append(predicate_values)
        if utility_map is not None:
            if TRAJECTORY_INDEX.UTILITY_MAP.value not in self.traj_dict:
                self.traj_dict[TRAJECTORY_INDEX.UTILITY_MAP.value] = []
            self.traj_dict[TRAJECTORY_INDEX.UTILITY_MAP.value].append(utility_map)
        if utility_values is not None:
            if TRAJECTORY_INDEX.UTILITY_VALUES.value not in self.traj_dict:
                self.traj_dict[TRAJECTORY_INDEX.UTILITY_VALUES.value] = []
            self.traj_dict[TRAJECTORY_INDEX.UTILITY_VALUES.value].append(utility_values)
        if next_predicate_values is not None:
            if TRAJECTORY_INDEX.NEXT_PREDICATE_VALUES.value not in self.traj_dict:
                self.traj_dict[TRAJECTORY_INDEX.NEXT_PREDICATE_VALUES.value] = []
            self.traj_dict[TRAJECTORY_INDEX.NEXT_PREDICATE_VALUES.value].append(next_predicate_values)




