from setuptools import setup

setup(
    name='hmm',
    version='0.0.1',
    description='Motion classification from accelerometry using Hidden Markov Model',
    author='Feiyang Huang',
    author_email='feh4005@med.cornell.edu',
    url='https://github.com/fyng/hmm_accelerometry',
    python_requires='>=3.10',
    install_requires=[
        'numpy',
        'pytest',
        'scikit-image',
        'scikit-learn',
        'setuptools',
        'matplotlib',
        'opencv-python',
    ],
    packages=['hmm'],
    license='Apache License, Version 2.0',
)