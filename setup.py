from setuptools import setup
from setuptools import setup, find_packages


INSTALL_REQUIRES = [
        "pycollimator[safe]"
    ]

setup(
    name='franka',
    version='0.1.0',
    description='franka',
    author='carter',
    install_requires=INSTALL_REQUIRES,
    python_requires='>=3.10',
)