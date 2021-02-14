from setuptools import setup, find_packages

packages = find_packages(
    where='.',
    include=['anomaly_detection*']
)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="anomaly_detection",
    version="0.0.1",
    author="Jitesh Gosar",
    author_email="gosar95@gmail.com",
    description="Training and inference scripts for anomaly_detection using AutoEncoders with CNN",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Jitesh17/anomaly_detection",
    py_modules=["anomaly_detection"],
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch",
        "torchvision",
        "albumentations",
        "numpy",
        "matplotlib",
        "Pillow",
    ],
    python_requires='>=3.6',
)
