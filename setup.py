from setuptools import find_packages
from setuptools import setup

setup(
    name='caciolai-NER',
    version='1.0.0',
    description='An end-to-end RNN model for Named Entity Recognition.',
    author='caciolai',
    license='MIT License',
    url='https://github.com/caciolai/End-to-end-RNN-Model-for-Named-Entity-Recognition',
    packages=find_packages("src"),  # include all packages under src
    package_dir={"": "src"},   # tell distutils packages are under src
    python_requires='>=3.7.9',
    install_requires=[
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