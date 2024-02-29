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
        T = len(X)

        # TODO: add guards on n_obs v.s. data range

        # initialize parameters
        self.A = np.eye(self.n_hidden, self.n_hidden) * 0.9
        for i in range(self.n_hidden):
            self.A[(i+1) % self.n_hidden, i] = 0.1

        self.B = np.random.rand(self.n_hidden, self.n_obs)
        self.B /= self.B.sum(axis=1, keepdims=True)

        self.pi = np.zeros(self.n_hidden)
        self.pi[0] = 1

        loss = []
        # train model
        for i in range(self.n_iter):
            alpha, scale = self.__forward(X)
            beta = self.__backward(X, scale)

            ll = np.log(scale).sum()
            loss.append(ll)

            # print(alpha)
            # print(beta)

            # EM
            gamma, xi = self._e_step(X, alpha, beta)

            # print(gamma)
            # print(xi)   

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

    def __forward(self, X):
        T = len(X)
        scale = []
        alpha = np.zeros((T, self.n_hidden))

        alpha[0,:] = self.pi * self.B[:,X[0]]
        scale.append(alpha[0].sum())
        alpha[0] /= scale[0]

        # forward
        for t in range(1, T):
            alpha[t,:] = (alpha[t-1,:] @ self.A) * self.B[:, X[t]]
            scale.append(alpha[t].sum())
            alpha[t,:] /= scale[-1]

        return alpha, scale
    
    
    def __backward(self, X, scale):
        T = len(X)
        beta = np.ones((T, self.n_hidden))
        for t in range(T-2, -1, -1):
            # TODO: is this wrong?
            beta[t,:] = (self.A  * self.B[:, X[t+1]]) @ beta[t+1,:]
            beta[t,:] /= scale[t]

        return beta

    def _e_step(self, X, alpha, beta):
        T = len(X)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)

        xi = np.zeros((T-1, self.n_hidden, self.n_hidden))
        for t in range(T-1):
            # FIXME:
            # print(self.B[:, X[t+1]].shape)
            # print(alpha[t,:, None].shape)
            xi[t,:,:] = alpha[t,:, None] * self.A * self.B[:, X[t+1]] * beta[t+1,:]
        xi /= xi.sum(axis=(1, 2), keepdims=True)

        return gamma, xi


    def _m_step(self, X, gamma, xi):
        self.A = xi.sum(axis=0) 
        self.A /= self.A.sum(axis=1, keepdims=True)
        assert self.A.sum(axis=1).all() == 1

        bmap = np.eye(self.n_obs)[X]
        self.B = gamma.T.dot(bmap) / gamma.sum(axis=0, keepdims=True).T
        assert self.B.sum(axis=1).all() == 1


    def predict(self, X):
        X = X.astype(int).flatten() 
        pass
    
    def save(self, path):
        pass

    def load(self, path):
        pass


if __name__ == '__main__':
    np.random.seed(0)
    model = HMM(2, 3, n_iter=10, verbose=True)
    X = np.random.randint(0, 3, 100)
    model.fit(X)