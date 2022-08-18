import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from scipy import stats

from recsim import document
from recsim import user
from documents.Document import LTSDocument


class LTSDocumentSampler(document.AbstractDocumentSampler):
  def __init__(self, doc_ctor=LTSDocument, **kwargs):
    super(LTSDocumentSampler, self).__init__(doc_ctor, **kwargs)
    self._doc_count = 0

  def sample_document(self):
      doc_features = {}
      doc_features['doc_id'] = self._doc_count
      doc_features['kaleness'] = self._rng.random_sample()
      self._doc_count += 1
      return self._doc_ctor(**doc_features)
