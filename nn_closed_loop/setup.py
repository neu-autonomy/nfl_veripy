from setuptools import setup, find_packages

setup(name='closed_loop',
      version='0.0.1',
      install_requires=[
          'torch',
          'matplotlib',
          'pandas',
          ],
      packages=find_packages()
)
