# Online-Active Learning framework with Tensorflow for Computer Vision

![TensorFlow Version: 2.2](https://img.shields.io/badge/Tensorflow%20Version-2.2-brightgreen)
![Support classification](https://img.shields.io/badge/Task-Classification-brightgreen)
![Support classification](https://img.shields.io/badge/Task-Object%20Detection-critical)

<!-- ## What's  -->
Active learning has been proven numerous times to be efficient at reducing the number of required data to reach a specific performance however the way it is implemented can totally negate the theoritical benefits. Indeed, reducing the number of data to annotate is nice, but if we need to wait for the model to start its training, make predictions on the unlabelled dataset, then make the queries.... We can easily loose time.

That's why this framework proposes a way to do online learning combined with active learning with tensorflow. 

## The structure
Two servers are run in parallel. One is the Labeler client, the other the Trainer server. 
The Labeler sends data to the trainer every n images annotated.
Meanwhile the Trainer is continuously training. It periodically evaluates itself and send queries back to the Labeler from a certain active learning strategy.
When a query is received by the Labeler it will automatically re-order the following unlabelled images according to the active learning strategy.

The Trainer will stop the training and save the model when the Labeler wants to, or when all images are annotated, or when the model metrics reach specific tresholds.
From there the user can use the model however he wants.

## How to use it

- Put images inside the ``./labeler/images`` directory
- Launch both servers blabla
- Go to dash url
- Fill labels then annotate
  
## Advanced configuration

config.py blabla