import os


TRAINER_IP = "http://localhost:3333"
## Image directory needs to be inside the labeler folder
IMAGE_DIRECTORY = "images/"
## Number of labelled images to buffer before sending them to the trainer
BUFFER_SIZE = 4
TEST_SET_FRAC = 0.002
ANNOTATIONS_SAVE_PATH = "./annotations"
