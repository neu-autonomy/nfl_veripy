from setuptools import setup, find_packages

setup(name='nn_partition',
      version='0.0.1',
      install_requires=[
          'torch',
          'alphashape',
          'sklearn',
          'scipy',
          'matplotlib',
          'imageio',
          'keras',
          'tensorflow',
          'pypoman',
          'tqdm',
          'pyclipper',
          ],
      packages=find_packages()
)
