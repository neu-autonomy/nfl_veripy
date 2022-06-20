from setuptools import setup, find_packages

setup(
    name="nn_closed_loop",
    version="0.0.1",
    install_requires=[
        "torch",
        "matplotlib",
        "pandas",
        "nn_partition",
        "tabulate",
        "colour",
    ],
    packages=find_packages(),
)
