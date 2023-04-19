from setuptools import setup

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
        "jax",
        "jax_verify",
        "parameterized",
    ],
    packages=["nn_closed_loop"],
    # package_data={"nn_closed_loop": ["py.typed", "**/py.typed"]},
    # zip_safe=False,
)
