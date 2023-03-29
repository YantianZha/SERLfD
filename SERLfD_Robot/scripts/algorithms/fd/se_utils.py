from enum import Enum
import torch
import numpy as np
import copy

class TRAJECTORY_INDEX(Enum):
    STATE = 'state'
    ACTION = 'action'
    NEXT_STATE = 'next_state'
    REWARD = 'reward'
    DONE = 'done'
    PREDICATE_VALUES = 'predicate_values'
    PREDICATE_VECTOR = 'predicate_vector'
    NEXT_PREDICATE_VALUES = 'next_predicate_values'
    NEXT_PREDICATE_VECTOR = 'next_predicate_vector'
    UTILITY_MAP = 'utility_map'
    UTILITY_VALUES = 'utility_values'
    UTILITY_VECTOR = 'utility_vector'


class UtilityTransition:
    def __init__(self, predicate_value, next_predicate_value):
        self.util_transition = {TRAJECTORY_INDEX.NEXT_PREDICATE_VALUES.value: next_predicate_value,
                                TRAJECTORY_INDEX.PREDICATE_VALUES.value: predicate_value}

    def set_transition(self, predicate_value, next_predicate_value):
        self.util_transition = {TRAJECTORY_INDEX.NEXT_PREDICATE_VALUES.value: next_predicate_value,
                                TRAJECTORY_INDEX.PREDICATE_VALUES.value: predicate_value}

def img2simpleStates(obs, start=0, end=-1, c=-1):
    # assert isinstance(obs, torch.Tensor), "Input must be in torch tensor format"
    if isinstance(obs, torch.Tensor):
        if len(obs.shape) == 3:
            # https://github.com/hill-a/stable-baselines/issues/133#issuecomment-561805417
            # Take last channel as extra augmenting features
            augment_features = torch.flatten(obs[c, ...])
            # Take known amount of direct features, rest are padding zeros
            augment_features = augment_features[start:end]

        elif len(obs.shape) == 4:
            augment_features = obs[:, c, :, :]
            augment_features = torch.flatten(augment_features, start_dim=1)
            augment_features = augment_features[:, start:end]

        else:
            raise ValueError("input shape is not in image format (B x C x H x W or C x H x W).")

    elif isinstance(obs, np.ndarray):
        if len(obs.shape) == 3:
            # Take last channel as extra augmenting features
            augment_features = np.reshape(obs[c, ...], (obs.shape[1]*obs.shape[2]))
            # Take known amount of direct features, rest are padding zeros
            augment_features = augment_features[start:end]

        elif len(obs.shape) == 4:
            augment_features = np.reshape(obs[:, c, :, :], (obs.shape[0], obs.shape[2]*obs.shape[3]))
            augment_features = augment_features[:, start:end]

        else:
            raise ValueError("input shape is not in image format (B x C x H x W or C x H x W).")

    else:
        raise NotImplementedError

    return augment_features

def simpleStates2img(img, one_d_feat, start=0, end=0, c=-1):
    if isinstance(img, torch.Tensor):
        if len(img.size()) == 3:
            assert len(one_d_feat) == end - start, "The size of one-d feature does not equal to end-id minus start-id"
            img_flatten = torch.flatten(img, start_dim=1)
            img_flatten[c, start:end] = one_d_feat
        if len(img.size()) == 4:
            assert one_d_feat.size()[1] == end - start, "The size of one-d feature does not equal to end-id minus start-id"
            img_flatten = torch.flatten(img, start_dim=2)
            img_flatten[:, c, start:end] = one_d_feat
        return torch.reshape(img_flatten, img.size())

    elif isinstance(img, np.ndarray):
        if len(img.shape) == 3:
            assert len(one_d_feat) == end - start, "The size of one-d feature does not equal to end-id minus start-id"
            img_flatten = np.reshape(img, (img.shape[0], img.shape[1]*img.shape[2]))
            img_flatten[c, start:end] = one_d_feat
        if len(img.shape) == 4:
            assert one_d_feat.shape[1] == end - start, "The size of one-d feature does not equal to end-id minus start-id"
            img_flatten = copy.deepcopy(np.reshape(img, (img.shape[0], img.shape[1], img.shape[2]*img.shape[3])))
            img_flatten[:, c, start:end] = one_d_feat
        return np.reshape(img_flatten, img.shape)
