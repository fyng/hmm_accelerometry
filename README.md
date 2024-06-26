# hmm_accemerometry
Motion classification from IMU accelerometry using Hidden Markov Model.

# Usage
Install micromamba or mamba as the package manager. To install micromamba, refer to the [installation guide](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)

To install the classifier:
1. Clone the repo
```
git clone https://github.com/fyng/hmm_accemerometry.git
```
```
cd hmm_accelerometry
```

2. Create virtual environment
```
micromamba env create -f environment.yml
```
```
micromamba activate hmm
```

3. Install the model
```
pip install .
```

4. Run the model
In `demo.ipynb`, update the directory of the test folder. Then run all cells.

# Acknowledgement
ECE 5242 - Intelligent Autonomous Systems taught by Prof Daniel Lee & Travers Rhodes 