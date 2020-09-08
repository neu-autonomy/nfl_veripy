from setuptools import setup, find_packages

setup(name='reach_lp',
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
          ],
      packages=find_packages()
)
