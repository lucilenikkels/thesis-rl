import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from scipy import stats

from recsim import document

class LTSDocument(document.AbstractDocument):
    def __init__(self, doc_id, kaleness):
        self.kaleness = kaleness
        # doc_id is an integer representing the unique ID of this document
        super(LTSDocument, self).__init__(doc_id)

    def create_observation(self):
        return np.array([self.kaleness])

    @staticmethod
    def observation_space():
        return spaces.Box(shape=(1,), dtype=np.float32, low=0.0, high=1.0)

    def __str__(self):
        return "Document {} with kaleness {}.".format(self._doc_id, self.kaleness)