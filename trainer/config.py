import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from functools import partial
import ee_strats.uncertainty_sampling as uncertainty_sampling


'''Configuration file for the Trainer'''

LABELER_IP = "http://127.0.0.1:3334"
BATCH_SIZE = 4 # Batch size for the training and test datasets
SHUFFLE_BUFFER_SIZE = 1000 # Buffer for the train and test datasets shuffling
TRAIN_CONTINUOUSLY = False # Whether or not the trainer should wait for new training data between each training loop
QUERY_EVERY = 1 # Make query every x batches sent by the labeler
EVAL_EVERY = 2 # Evaluate every x batches sent by the labeler
INPUT_SHAPE = (224, 224) # Model input shape
NUM_EPOCHS_PER_LOOP = 5 # Number of epochs to do per training loop
TRAINING_VERBOSITY = 2 # Keras model.fit verbosity level
ANNOTATIONS_SAVE_PATH = "./annotations" # Save directory for the annotations, should be the same as the one from the Labeler configuration.
SAVED_MODEL_PATH = "./saved_model" # Save directory for the keras model
FORCE_ON_CPU = True # Force tensorflow to use the CPU

## Set an early stopping treshold per metric. Dict of metrics as keys and thresholds as values.
## You can skip some metrics if you want to. We can also define a threshold for the "loss".
## If you don't want early stopping logic, set the variable to an empty dict
EARLY_STOPPING_METRICS_THRESHOLDS = {"accuracy": [0.95, "upper_bound"],
                                        "loss": [0, "lower_bound"]}


def setup_model(num_classes, input_shape=INPUT_SHAPE+(3,)):
    '''Defines a basic test model.'''
    baseModel = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape,
        input_tensor=layers.Input(shape=input_shape))
    headModel = baseModel.output
    headModel = layers.AveragePooling2D(pool_size=(3, 3))(headModel)
    headModel = layers.Flatten(name="flatten")(headModel)
    headModel = layers.Dense(128, activation="relu")(headModel)
    headModel = layers.Dropout(0.5)(headModel)
    headModel = layers.Dense(num_classes, activation="softmax")(headModel)
    baseModel.trainable = False
    model = keras.Model(inputs=baseModel.input, outputs=headModel)
    model.compile(loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(),
        metrics=['accuracy'])
    return model

def reload_model(*_):
    '''Use this if you want to reload a model saved from a previous run'''
    return keras.models.load_model(SAVED_MODEL_PATH)


## Select your model function here, num_classes has to be the only argument. Must return a compiled model.
model_fn = setup_model
# model_fn = reload_model  # Use this if you want to reload the previous model.



## Choose your Exploration/Exploitation strat from the ee_strats module
## This base strat use a random entropy sampling with a probability of 0.8.
ee_strat = partial(uncertainty_sampling.randomize_sampling, 
                    sampling=uncertainty_sampling.entropy, p=0.8)
