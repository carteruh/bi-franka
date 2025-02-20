from setuptools import setup
from setuptools import setup, find_packages


INSTALL_REQUIRES = [
        "numpy",
        "scipy",
        "matplotlib",
        "jax",           # or "jax[cpu]" / "jax[gpu]" depending on your system
        "mujoco",
        "mediapy",
    ]

setup(
    name='franka',
    version='0.1.0',
    description='franka',
    author='carter',
    install_requires=INSTALL_REQUIRES,
    python_requires='>=3.10',
)