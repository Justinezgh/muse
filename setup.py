from setuptools import setup, find_packages

setup(
  name='muse_exp',
  version='0.0.1',
  url='https://github.com/Justinezgh/muse',
  description='MUSE experiments',
  packages=find_packages(),
  package_dir={'muse_exp': 'muse_exp'},
  package_data={
      'muse_exp': ['data/*.csv', 'data/*.npy', 'data/*.pkl'],
   },
  include_package_data=True,
  install_requires=[
    'numpy>=1.19.2',
    'jax>=0.2.0',
    'tensorflow_probability>=0.14.1',
    'numpyro==0.10.1',
    'jax-cosmo>=0.1.0', 
    'typing_extensions>=4.4.0'
  ],
)