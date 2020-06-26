# Online-Active Learning framework with Tensorflow for Computer Vision
<p>
    <img src="Picsellia.png" width=140 height=195>
</p>

![TensorFlow Version: 2.2](https://img.shields.io/badge/Tensorflow%20Version-2.2-brightgreen)
![Support classification](https://img.shields.io/badge/Task-Classification-brightgreen)
![Support classification](https://img.shields.io/badge/Task-Object%20Detection-critical)

Active learning has been proven numerous times to be efficient at reducing the quantity of data required to obtain a good accuracy. However the way it is implemented can totally negate the theoritical benefits. Indeed, reducing how much data we have to annotate is nice, but if we need to wait for the model to train then make predictions on the unlabelled dataset to make the queries.... It's not worth it.

That's why this framework proposes a way to do online learning combined with active learning with Tensorflow 2.2.
A little annotation interface has been developed so you can play with the framework for a classification task. (Don't worry the Object Detection part of the framework will be released soon)

To use our framework with a really optimized annotation interface, both for classification and detection/segmentation, you can check out our platform : [Picsell.ia](https://bit.ly/3g24i5n) and the best of all, it's free to use ! 🚀

## The structure
Two servers run in parallel. One is the Labeling Interface (Labeler), the other one is the Training server (Trainer).
The Labeler sends annotated data to the trainer every n images.
Meanwhile the Trainer is continuously training. It periodically evaluates itself and send queries back to the Labeler with a certain Active Learning strategy.
When a query is received by the Labeler it will automatically re-order the reamining unlabelled images according to the Active Learning strategy.

The Trainer will stop the training and save the model when you tell him to, or when all images are annotated, or when the model metrics reach specific tresholds.

From there the user can run inference with his model and has only annotated a sufficient amount of images.


## Getting Started

### Installation
- git clone the repo
- pip install -r requirements.txt


### How to
- Put your images inside the ``images/`` directory.
- Make changes to the ``config.py`` files if needed (see the Advanced Configuration).
- Launch both ``./trainer/app.py`` and ``./labeler/app.py`` from the root directory of the repository.
- Go to localhost:3334 in your favorite browser (replace localhost with the host ip if used remotely).
- Fill in your labels then start annotating. The Trainer will automatically create a test set, start the training, and make queries.
- Annotate everyting or stop the training with the button if wanted. The annotations and model will be automatically saved (wait until "Trainer safely shut down" is printed in your terminal before killing the servers).
- The Labeler will automatically load previous annotations and send them to the Trainer when you launch it.
- We can easily set up the Trainer to reload the previous model by setting up ``model_fn`` to ``reload_model`` in ``/trainer/config.py``.

## Advanced configuration

There are two config.py files, one for the Labeler and one for the Trainer. You can modify the settings of the apps there.

### Labeler configuration
Inside the labeler/config.py the most important settings to set up are :
- BUFFER_SIZE, the number of data to buffer before sending them to the trainer.
- TEST_SET_FRAC, fraction of the dataset to use as test data (necessary to perform validation).

Other settings are available and described inside the file.

### Trainer configuration
Set up parameters for the model training, active learning strategy...
#### Constants
The most important variables to set up are :
- BATCH_SIZE, heavily depends on your model and hardware configuration.
- EARLY_STOPPING_METRICS_THRESHOLDS, stop the training when reaching one of the thresholds for the specific metrics. More details inside the file.
- EVAL_AND_QUERY_EVERY, number of batches sent by the Labeler to wait before evaluating and making active learning queries.
- INPUT_SHAPE, the input shape of your model.

Other settings are available and described inside the file.

#### Model, E/E strategy and query method
- ``model_fn`` A function returning a compiled keras model. By default you are given a basic test model and a reload model function.
- ``ee_strat`` The exploration/exploitation strategy. You can select one from the trainer/ee_strats.py module or define your own.

## Roadmap

### Core updates
- [x] Initial release
- [ ] Bug hunting
- [ ] Add more E/E strategies and query methods
- [ ] Add support for object-detection when the Tensorflow Object Detection API is updated to tensorflow 2 !!
- [ ] Any other requested feature

### Front end enhancements
Those features are less important since the Labeler isn't the core aspect of this module but are still nice to have.
- [ ] Seeing metrics evolution and other information such as labels repartition while annotating
- [ ] "Stop page" showing a nice dashboard
- [ ] Pause button to temporarly stop the training.
- [ ] Any other requested feature

## Getting Help

You can come talk to us on the following channels :
- [Discord](https://discord.gg/fY5cjvJ)
- [Facebook](https://www.facebook.com/Picsellia-397087527546284)
- [Reddit](https://www.reddit.com/r/picsellia/)

## Contributing, feature requests and bug reporting
- If you want to request a feature or report a bug you can open an issue.
- If you want to contribute to the project, you can fork the project and make a pull request.
