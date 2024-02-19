import numpy as np
import json
import codecs

class HMM:
    def __init__(self, n_hidden, n_obs, n_iter=100, tol=1e-6):
        self.n_hidden = n_hidden
        self.n_obs = n_obs
        self.n_iter = n_iter
        self.tol = tol
        
        self.A = None
        self.B = None
        self.pi = None
        self.log_likelihood = None

    def fit(self, X):
        pass

    def predict(self, X):
        pass