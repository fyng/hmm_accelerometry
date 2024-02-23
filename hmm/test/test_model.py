from hmm import HMM 
import numpy as np
import pytest

@pytest.fixture
def hmm_data_fixture():
    # The data is of a process that stays in hidden state 0 for 1000 steps, 
    # then in hidden state 1 for 1000 steps, 
    # then in hidden state 0 for 1000 steps

    # In state 0: always emit observation with label 0
    # In state 1: emit observation with label 0 half the time and 1 half the time.
    X = np.concatenate((
        np.zeros((1000,1)),
        np.random.randint(0,2,size=(1000,1)),
        np.zeros((1000,1))
    ))
    return X

def test_hmm(hmm_data_fixture):
    X = hmm_data_fixture
    print(len(X))
    model = HMM(2, 2, verbose=True)
    model.fit(X)
    assert model.A.shape == (2, 2)
    assert model.B.shape == (2, 2)
    assert np.isclose(model.A, 
                      np.array([[0.999, 0.001], [0.999, 0.001]]))    
    assert np.isclose(model.B, 
                      np.array([[1.0, 0.0], [0.5, 0.5]]))    