import os


'''Configuration file for the Labeler'''

TRAINER_IP = "http://localhost:3333"
## Image directory needs to be inside the repo root dir.
IMAGE_DIRECTORY = "images/"
## Number of labelled images to buffer before sending them to the trainer
BUFFER_SIZE = 10
## Fraction of the dataset to use as test set
TEST_SET_FRAC = 0.10
## Path to the annotations to reload, should be the same as the Trainer config.
ANNOTATIONS_SAVE_PATH = "./annotations"