from setuptools import setup, find_packages

setup(
    name='mcts',
    version='0.4',
    packages=find_packages(),
    install_requires=[
        'sortedcontainers>=1.5.9',
        'logwood>=3.1.0',
        'numpy>=1.13.3',
        'keras>=2.1.4',
        'xxhash>=1.0.1'
    ]
)
