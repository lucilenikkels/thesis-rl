import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from scipy import stats

from recsim import user


class LTSUserState(user.AbstractUserState):
    def __init__(self, memory_discount, sensitivity, innovation_stddev,
                 choc_mean, choc_stddev, kale_mean, kale_stddev,
                 net_kaleness_exposure, time_budget, observation_noise_stddev=0.1
                 ):
        ## Transition model parameters
        ##############################
        self.memory_discount = memory_discount
        self.sensitivity = sensitivity
        self.innovation_stddev = innovation_stddev

        ## Engagement parameters
        self.choc_mean = choc_mean
        self.choc_stddev = choc_stddev
        self.kale_mean = kale_mean
        self.kale_stddev = kale_stddev

        ## State variables
        ##############################
        self.net_kaleness_exposure = net_kaleness_exposure
        self.satisfaction = 1 / (1 + np.exp(-sensitivity * net_kaleness_exposure))
        self.time_budget = time_budget

        # Noise
        self._observation_noise = observation_noise_stddev

    def create_observation(self):
        """User's state is not observable."""
        clip_low, clip_high = (-1.0 / (1.0 * self._observation_noise),
                               1.0 / (1.0 * self._observation_noise))
        noise = stats.truncnorm(
            clip_low, clip_high, loc=0.0, scale=self._observation_noise).rvs()
        noisy_sat = self.satisfaction + noise
        return np.array([noisy_sat, ])

    @staticmethod
    def observation_space():
        return spaces.Box(shape=(1,), dtype=np.float32, low=-2.0, high=2.0)

    # scoring function for use in the choice model -- the user is more likely to
    # click on more chocolatey content.
    def score_document(self, doc_obs):
        return 1 - doc_obs