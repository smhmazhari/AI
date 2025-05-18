# from hmmlearn import hmm
import numpy as np

# observations = np.array([[0.1, 0.2, 0.3],
#                          [0.4, 0.5, 0.6],
#                          [0.7, 0.8, 0.9]])
# print (observations)
# model = hmm.GaussianHMM(n_components=3, covariance_type='diag', n_iter=100)

# train_data = np.random.rand(10, 3)  # Example training data
# print(train_data)
# model.fit(train_data)
# # print(model)
# hidden_states = model.predict(observations)

# print(hidden_states)
# print(f"Predicted hidden states: {hidden_states}")

observations = np.array([[0.1, 0.2, 0.3],
                         [0.4, 0.5, 0.6],
                         [0.7, 0.8, 0.9],
                         [0.2, 0.8, 0.9]])
print (observations.shape[1])