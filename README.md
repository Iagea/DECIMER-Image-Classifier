# DECIMER Image Classifier
[![License](https://img.shields.io/badge/License-MIT%202.0-blue.svg)](https://opensource.org/licenses/MIt)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-blue.svg)](https://GitHub.com/iagea/DECIMER-Image-Classifier/graphs/commit-activity)
[![GitHub issues](https://img.shields.io/github/issues/iagea/DECIMER-Image-Classifier.svg)](https://GitHub.com/iagea/DECIMER-Image-Classifier/issues/)
[![GitHub contributors](https://img.shields.io/github/contributors/iagea/DECIMER-Image-Classifier.svg)](https://GitHub.com/iagea/DECIMER-Image-Classifier/graphs/contributors/)

This model's aim is to classify whether or not an image is a chemical structure or not. It was built using EfficientNetB0 as a base model, using transfer learning and fine tuning it.

It gives a prediction between 0 and 1 where 0 means it is a chemical structure and 1 means it is not. The data to constract the model will be available in Zenodo. 

## Model construction

The model construction script is available on the **model_construction** folder with comments for every step taken.

## Example

An example with 10 images from the test set that will be available in Zenodo as well is present in the form of a Jupyter notebook on the folder **examples**.

