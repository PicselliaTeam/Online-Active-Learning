import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import uncertainty_measures
import ee_strats


LABELER_IP = "http://127.0.0.1:3334"
BATCH_SIZE = 4
SHUFFLE_BUFFER_SIZE = 1000

## Set an early stopping treshold per metric. Dict of metrics as keys and thresholds as values.
## You can skip some metrics if you want to. We can also define a threshold for the "loss".
## If you don't want early stopping logic, set the variable to empty dict
EARLY_STOPPING_METRICS_TRESHOLDS = {"accuracy": 2}

## Evaluate and make query every x batches sent by the labeler
EVAL_AND_QUERY_EVERY = 15
## Model input shape
INPUT_SHAPE = (224, 224)

NUM_EPOCHS_PER_LOOP = 5
TRAINING_VERBOSITY = 0
ANNOTATIONS_SAVE_PATH = "./annotations"
SAVED_MODEL_PATH = "./saved_model"

def setup_model(num_classes, input_shape=INPUT_SHAPE+(3,)):
    '''Define your model here, num_classes has to be the only argument.
        Must return a compiled model.'''
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


## Define your model function here
model_fn = setup_model
# model_fn = reload_model


## Choose your uncertainty measure from the uncertainty_measure module
uncertainty_measure = uncertainty_measures.SumEntropy

## Choose your EE strat from the ee_strats module
ee_strat = ee_strats.sort_decreasingly
