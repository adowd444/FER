## Introduction
This repository is based off a model outlined in the paper here: http://cs230-fer.firebaseapp.com/ <br> The model used in the paper has an accuracy of 75.8% which outperforms almost all FER models that are currently published.
The model is meant to be applied to real-time facial expression recognition, as used in this project.

## Getting Started
1. Refer to the requirements.txt to install required packages
2. Download the FER2013 and AffectNet datasets
3. Use each preprocessing code for each dataset in the datasets folder (this makes the datasets usable for the model)
4. After data is preprocessed, use the main.py script to create the model
5. Finally, insert model into webcam.py to use real-time facial expression recognition
