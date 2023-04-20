from setuptools import find_packages, setup

print("******")
print(find_packages())
print("******")


setup(
    name="nfl_veripy",
    version="0.0.1",
    install_requires=[
        "torch",
        # "matplotlib",
        # "pandas",
        # "tabulate",
        # "colour",
        # "jax",
        # "jax_verify @ git+https://gitlab.com/mit-acl/ford_ugvs/jax_verify.git",
        # "crown_ibp @ git+https://gitlab.com/mit-acl/ford_ugvs/crown_ibp.git",
        # "auto_LiRPA @ git+https://github.com/KaidiXu/auto_LiRPA.git",
        # "parameterized",
        # "pypoman",
        # "alphashape",
        # "scikit-learn",
        # "scipy",
        # "imageio",
        # "tqdm",
        # "pyclipper",
        # "pygifsicle",
    ],
    # package_dir={"": "src"},
    packages=[find_packages()],
    # package_data={"nfl_veripy": ["py.typed", "**/py.typed"]},
    # zip_safe=False,
)
