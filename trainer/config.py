from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
import uncertainty_measures
import ee_strats



BATCH_SIZE = 4

## Set an early stopping treshold per metric. Dict of metrics as keys and thresholds as values.
## You can skip some metrics if you want to. We can also define a threshold for the "loss".
EARLY_STOPPING_METRICS_TRESHOLDS = {"accuracy":0.95}

## Evaluate and make query every x batches sent by the labeler
EVAL_AND_QUERY_EVERY = 5

## Model input shape
INPUT_SHAPE = (224, 224)

NUM_EPOCHS_PER_LOOP = 5
TRAINING_VERBOSITY = 0
ANNOTATIONS_PATH = "./annotations"

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


## Choose your uncertainty measure from the uncertainty_measure module
uncertainty_measure = uncertainty_measures.SumEntropy

## Choose your EE strat from the ee_strats module
ee_strat = ee_strats.sort_decreasingly