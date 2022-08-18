import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from scipy import stats

from recsim import user
from recsim.choice_model import MultinomialLogitChoiceModel
from user.UserResponse import LTSResponse
from user.UserState import LTSUserState
from user.UserSampler import LTSStaticUserSampler


def user_init(self, slate_size, seed=0):
    super(LTSUserModel, self).__init__(LTSResponse,
                                       LTSStaticUserSampler(LTSUserState, seed=seed),
                                       slate_size)
    self.choice_model = MultinomialLogitChoiceModel({})


def simulate_response(self, slate_documents):
    # List of empty responses
    responses = [self._response_model_ctor() for _ in slate_documents]
    # Get click from of choice model.
    self.choice_model.score_documents(
        self._user_state, [doc.create_observation() for doc in slate_documents])
    scores = self.choice_model.scores
    selected_index = self.choice_model.choose_item()
    # Populate clicked item.
    self._generate_response(slate_documents[selected_index],
                            responses[selected_index])
    return responses


def generate_response(self, doc, response):
    response.clicked = True
    # linear interpolation between choc and kale.
    engagement_loc = (doc.kaleness * self._user_state.choc_mean
                      + (1 - doc.kaleness) * self._user_state.kale_mean)
    engagement_loc *= self._user_state.satisfaction
    engagement_scale = (doc.kaleness * self._user_state.choc_stddev
                        + ((1 - doc.kaleness)
                           * self._user_state.kale_stddev))
    log_engagement = np.random.normal(loc=engagement_loc,
                                      scale=engagement_scale)
    response.engagement = np.exp(log_engagement)


def update_state(self, slate_documents, responses):
    for doc, response in zip(slate_documents, responses):
        if response.clicked:
            innovation = np.random.normal(scale=self._user_state.innovation_stddev)
            net_kaleness_exposure = (self._user_state.memory_discount
                                     * self._user_state.net_kaleness_exposure
                                     - 2.0 * (doc.kaleness - 0.5)
                                     + innovation
                                     )
            self._user_state.net_kaleness_exposure = net_kaleness_exposure
            satisfaction = 1 / (1.0 + np.exp(-self._user_state.sensitivity
                                             * net_kaleness_exposure)
                                )
            self._user_state.satisfaction = satisfaction
            self._user_state.time_budget -= 1
            return


def is_terminal(self):
    """Returns a boolean indicating if the session is over."""
    return self._user_state.time_budget <= 0


LTSUserModel = type("LTSUserModel", (user.AbstractUserModel,),
                    {"__init__": user_init,
                     "is_terminal": is_terminal,
                     "update_state": update_state,
                     "simulate_response": simulate_response,
                     "_generate_response": generate_response})
