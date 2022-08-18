import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from scipy import stats

from recsim.simulator import environment
from recsim.simulator import recsim_gym
from documents.Document import LTSDocument
from documents.DocumentSampler import LTSDocumentSampler
from user.UserSampler import LTSStaticUserSampler
from user.UserModel import LTSUserModel


def clicked_engagement_reward(responses):
  reward = 0.0
  for response in responses:
    if response.clicked:
      reward += response.engagement
  return reward


if __name__ == '__main__':
    sampler = LTSDocumentSampler()
    for i in range(5): print(sampler.sample_document())
    d = sampler.sample_document()
    print("Documents have observation space:", d.observation_space(), "\n"
                                                                      "An example realization is: ",
          d.create_observation())

    sampler = LTSStaticUserSampler()
    starting_nke = []
    for i in range(1000):
        sampled_user = sampler.sample_user()
        starting_nke.append(sampled_user.net_kaleness_exposure)
    _ = plt.hist(starting_nke)

    slate_size = 3
    num_candidates = 10
    ltsenv = environment.Environment(
        LTSUserModel(slate_size),
        LTSDocumentSampler(),
        num_candidates,
        slate_size,
        resample_documents=True)

    lts_gym_env = recsim_gym.RecSimGymEnv(ltsenv, clicked_engagement_reward)

    observation_0 = lts_gym_env.reset()
    print('Observation 0')
    print('Available documents')
    doc_strings = ['doc_id ' + key + " kaleness " + str(value) for key, value
                   in observation_0['doc'].items()]
    print('\n'.join(doc_strings))
    print('Noisy user state observation')
    print(observation_0['user'])
    # Agent recommends the first three documents.
    recommendation_slate_0 = [0, 1, 2]
    observation_1, reward, done, _ = lts_gym_env.step(recommendation_slate_0)
    print('Observation 1')
    print('Available documents')
    doc_strings = ['doc_id ' + key + " kaleness " + str(value) for key, value
                   in observation_1['doc'].items()]
    print('\n'.join(doc_strings))
    rsp_strings = [str(response) for response in observation_1['response']]
    print('User responses to documents in the slate')
    print('\n'.join(rsp_strings))
    print('Noisy user state observation')
    print(observation_1['user'])

    plt.show()
