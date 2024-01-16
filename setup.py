from setuptools import setup, find_packages
setup(
    name = "MOT_pedestrians",
    version="1.0",
    packages=find_packages(),
    package_data = {"MOT_pedestrians": ["data/*"]}
)