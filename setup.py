#!/usr/bin/env python

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="decimer_image_classifier",
    version="1.0.1",
    author="Isabel Agea",
    author_email="Maria.Isabel.Agea.Lorente@vscht.cz",
    maintainer="Isabel Agea",
    maintainer_email="Maria.Isabel.Agea.Lorente@vscht.cz",
    description="DECIMER Image Classifier is a classifier based on EfficientNetB0 that tells images of chemical structures and other types of images apart.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Iagea/DECIMER-Image-Classifier",
    packages=setuptools.find_packages(),
    license="MIT",
    install_requires=["tensorflow==2.12.0", "ipyplot", "pillow>=8.2.0", "attrs>=19.2.0"],
    package_data={"decimer_image_classifier": ["*.*", "model/*.*", "model/*/*.*"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3 :: Only",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
