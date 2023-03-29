import gym
import numpy as np
from utils.utils import to_grayscale
import cv2
from skimage.transform import rescale, resize


class PytorchImage(gym.ObservationWrapper):
    def __init__(self, env, is_to_gray=False, is_normalize=False, rescale=None, resize=None, binary_threshold=None):
        super(PytorchImage, self).__init__(env)
        # we check current shape of observations in environment
        current_shape = self.observation_space.shape
        self.is_to_gray = is_to_gray
        self.is_normalize = is_normalize
        self.normal_min = 0.0
        self.normal_max = 255.0
        self.rescale = rescale
        self.resize = resize
        self.binary_threshold = binary_threshold
        self.last_obs = None

        low = 0
        high = 1.0 if is_normalize else 255

        h = self.resize[0] if self.resize is not None else current_shape[0]
        w = self.resize[1] if self.resize is not None else current_shape[1]

        if is_to_gray:
            self.observation_space = gym.spaces.Box(low=low
                                                    , high=high
                                                    , shape=(1, h, w))
        else:
            # we change order of dimensions - so last one (-1) becomes first
            self.observation_space = gym.spaces.Box(low=low
                                                    , high=high
                                                    , shape=(current_shape[-1], h, w))

    def observation(self, observation):
        # and finally we change order of dimensions for every single observation
        # here transpose method could be also used
        self.last_obs = np.copy(observation)

        obs = observation.astype(float)
        if self.is_to_gray:
            obs = np.expand_dims(to_grayscale(obs), axis=2)
        # do binary thresholding
        if self.binary_threshold is not None:
            _, obs = cv2.threshold(obs, self.binary_threshold, 255, cv2.THRESH_BINARY)
            obs = np.expand_dims(obs, axis=2)
        # normalize
        if self.is_normalize:
            for i_channel in range(obs.shape[2]):
                channel_min = self.normal_min
                channel_max = self.normal_max
                delta = channel_max - channel_min
                if delta == 0:
                    obs[:, :, i_channel] = np.zeros_like(obs[:, :, i_channel], dtype=np.float)
                else:
                    obs[:, :, i_channel] = (obs[:, :, i_channel] - channel_min) / (channel_max - channel_min)
        # rescale the image
        if self.rescale is not None:
            obs = rescale(obs, self.rescale, multichannel=True, anti_aliasing=False)
        # resize the image
        if self.resize is not None:
            obs = resize(obs, self.resize, anti_aliasing=False)

        # to Pytorch channel format
        obs = np.moveaxis(obs, -1, 0)
        return obs

    def get_rgb_obs(self):
        return self.last_obs

    def process_utility_map(self, utility_map):
        utility_map = utility_map.astype(float)
        # rescale the image
        if self.rescale is not None:
            # the utility map should be a 2d array
            utility_map = rescale(utility_map, self.rescale, multichannel=False, anti_aliasing=False)
        # resize the utility map
        if self.resize is not None:
            utility_map = resize(utility_map, self.resize, anti_aliasing=False)
        return utility_map


class GymRenderWrapper(gym.Wrapper):
    """ If used with PytorchImage, must be used inside PytorchImage """
    def __init__(self, env):
        super(GymRenderWrapper, self).__init__(env)
        self.env = env

    def reset(self):
        self.env.reset()
        return self.env.render(mode='rgb_array')

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        next_state = self.env.render(mode='rgb_array')
        return next_state, reward, done, info



