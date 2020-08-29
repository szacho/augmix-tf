import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="augmix-tf",
    version="1.0.0",
    description="An implementation of novel data augmentation AugMix (2020) in TensorFlow. It runs on TPU.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/szacho/augmix-tf",
    author="Michal Szachniewicz",
    author_email="mszachniewicz@outlook.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["augmix"],
    include_package_data=True,
    install_requires=["tensorflow", "tensorflow-probability"],
)