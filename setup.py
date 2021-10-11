from setuptools import find_packages
from setuptools import setup

setup(
    name='caciolai-NER',
    version='1.0.0',
    description='An end-to-end RNN model for Named Entity Recognition.',
    author='caciolai',
    license='MIT License',
    url='https://github.com/caciolai/End-to-end-RNN-Model-for-Named-Entity-Recognition',
    packages=find_packages(where="src"),
    install_requires=[
        'torch',
        'jsonpickle',
        'matplotlib',
        'nltk',
        'numpy',
        'pandas',
        'scikit-learn',
        'scipy',
        'seaborn',
        'torch',
        'tqdm'
    ]
)