import numpy as np
import json
import codecs

class HMM:
    def __init__(self, n_hidden, n_obs, n_iter=100, tol=1e-6, verbose=False):
        self.n_hidden = n_hidden
        self.n_obs = n_obs
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose
        
        self.A = np.zeros((n_hidden, n_hidden))
        self.B = np.zeros((n_hidden, n_obs))
        self.pi = np.zeros(n_hidden)
        self.log_likelihood = None

    def fit(self, X):
        X = X.astype(int).flatten() 
        loss = []

        # TODO: add guards on n_obs v.s. data range

        # initialize parameters
        self.A = np.eye(self.n_hidden, self.n_hidden) / 2
        for i in range(self.n_hidden):
            self.A[i, (i+1) % self.n_hidden] = 0.5

        self.B = np.ones((self.n_hidden, self.n_obs)) / self.n_obs

        self.pi = np.zeros(self.n_hidden)
        self.pi[0] = 1

        # train model
        for i in range(self.n_iter):
            T = len(X)
            alpha = np.zeros((T, self.n_hidden))
            alpha[0] = self.pi * self.B[:, X[0]]

            # forward
            for t in range(1, T):
                alpha[t] = alpha[t-1].dot(self.A.dot(self.B[:, X[t]]))
            # backward
            beta = np.zeros((T, self.n_hidden))
            beta[-1] = 1
            for t in range(T-2, -1, -1):
                beta[t] = beta[t+1].dot(self.A.dot(self.B[:, X[t+1]]))

            ll = np.log(alpha[-1]).sum()
            loss.append(ll)

            # EM
            gamma, xi = self._e_step(X, alpha, beta)
            self._m_step(X, gamma, xi)

            if self.verbose:
                print(f'iteration {i+1}, log likelihood: {ll}')
            if self.log_likelihood is None:
                self.log_likelihood = ll
            else:
                if np.abs(ll - self.log_likelihood) < self.tol:
                    return loss
            self.log_likelihood = ll

        return loss

    def predict(self, X):
        X = X.astype(int).flatten() 
        pass
    

    def _e_step(self, X, alpha, beta):
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi = np.zeros((len(X)-1, self.n_hidden, self.n_hidden))
        for t in range(len(X)-1):
            xi[t] = (alpha[t] * self.A).dot(self.B[:, X[t+1]] * beta[t+1]) 
        xi /= xi.sum(axis=(1, 2), keepdims=True)

        return gamma, xi


    def _m_step(self, X, gamma, xi):
        self.A = xi.sum(axis=0) 
        self.A /= self.A.sum(axis=1, keepdims=True)
        assert self.A.sum(axis=1).all() == 1

        bmap = np.eye(self.n_obs)[X]
        self.B = gamma.T.dot(bmap) / gamma.sum(axis=0, keepdims=True).T
        assert self.B.sum(axis=1).all() == 1


if __name__ == '__main__':
    np.random.seed(0)
    model = HMM(2, 3, n_iter=10, verbose=True)
    X = np.random.randint(0, 3, 100)
    model.fit(X)