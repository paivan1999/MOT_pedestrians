from setuptools import setup, find_packages
with open('requirements.txt') as f:
    required = f.read().splitlines()
setup(
    name = "MOT_pedestrians",
    version="1.0",
    packages=find_packages(),
    install_requires=required,
    package_data = {"MOT_pedestrians": ["data/*"]}
)