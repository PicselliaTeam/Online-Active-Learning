# Online-Active Learning framework with Tensorflow for Computer Vision
<p>
    <img src="Picsellia.png" width=140 height=195>
</p>

![TensorFlow Version: 2.2](https://img.shields.io/badge/Tensorflow%20Version-2.2-brightgreen)
![Support classification](https://img.shields.io/badge/Task-Classification-brightgreen)
![Support classification](https://img.shields.io/badge/Task-Object%20Detection-critical)

Active learning has been proven numerous times to be efficient at reducing the number of required data to reach a specific performance however the way it is implemented can totally negate the theoritical benefits. Indeed, reducing the number of data to annotate is nice, but if we need to wait for the model to start its training, make predictions on the unlabelled dataset, then make the queries.... We can easily loose time.

That's why this framework proposes a way to do online learning combined with active learning with tensorflow. 

## The structure
Two servers are run in parallel. One is the Labeler client, the other the Trainer server. 
The Labeler sends data to the trainer every n images annotated.
Meanwhile the Trainer is continuously training. It periodically evaluates itself and send queries back to the Labeler from a certain active learning strategy.
When a query is received by the Labeler it will automatically re-order the following unlabelled images according to the active learning strategy.

The Trainer will stop the training and save the model when the Labeler wants to, or when all images are annotated, or when the model metrics reach specific tresholds.
From there the user can use the model and annotations however he wants.


## Getting Started

### Installation
- git clone the repo
- pip install -r requirements.txt 


### How to 

- Put your images inside the ``images/`` directory.
- Make changes to the config.py files if needed (see the Advanced Configuration).
- Launch both ./trainer/app.py abd ./labeler/app.py from the root directory of the repository.
- Go to localhost:3334 in your favorite browser (replace localhost with the host local ip if used inside a local network).
- Fill in your labels then start annotating. The Trainer will automatically create a test set, start the training, and make queries.
- Annotate everyting or stop the training with the button if wanted. The annotations and model will be automatically saved.
- The Labeler will automatically load previous annotations and send them to the Trainer when launching it. We can easily set up the Trainer to reload the previous model.
  
## Advanced configuration

There are two config.py files, one for the Labeler and one for the Trainer. Inside them you can set up different settings.

### Labeler configuration
Inside the labeler/config.py the most important settings to set up are :
- BUFFER_SIZE, the number of data to buffer before sending them to the trainer.
- TEST_SET_FRAC, fraction of the dataset to use as test data.

Other settings are available and described inside the file.

### Trainer configuration
Here you can set up parameters for the model, training, active learning strategy...
#### Constants
The most important constants to set up are :
- BATCH_SIZE, heavily depends on your model and hardware configuration.
- EARLY_STOPPING_METRICS_THRESHOLDS, stop the training when reaching one the thresholds for the specific metrics. More details inside the file.
- EVAL_AND_QUERY_EVERY, number of batches sent by the Labeler to wait before evaluating and making active learning queries.
- INPUT_SHAPE, the input shape of your model.

Other settings are available and described inside the file.

#### Model, E/E strategy and query method
- model_fn, a function returning a compiled keras model, should have num_classes as sole needed argument.
- uncertainty_measure, the query method used. You can select one from the trainer/uncertainty_measures or define your own.
- ee_strat, the exploration/exploitation strategy. You can select one from the trainer/ee_strats model or defien your own.

## Roadmap

### Core updates
- [x] Initial release
- [ ] Bug hunting
- [ ] Add more E/E strategies and query methods
- [ ] Add support for object-detection when the Tensorflow API is updated to tensorflow 2 !!
- [ ] Any other requested feature

### Front end enhancements
Those features are less important since the Labeler isn't the core aspect of this module but are still nice to have.
- [ ] Seeing metrics evolution and other information such as labels repartition while annotating
- [ ] "Stop page" showing a nice dashboard
- [ ] Pause button to temporarly stop the training.
- [ ] Any other requested feature

## Getting Help


## Contributing, feature requests and bug reporting
- If you want to request a feature or report a bug you can open an issue.
- If you want to contribute to the project, you can fork the project and make a pull request.

