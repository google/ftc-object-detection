# Training Scripts

This directory contains an assortment of scripts which are meant to simplify the
process of training a new model for object detection. There are scripts to
preprocess, annotate, and convert data, as well as scripts to visualize trained
models. The `experimental` subdirectory contains a few other scripts for one-off
tasks, as well as other implementations and experiments. Make sure to use the
`--help` command line options for all scripts mentioned, and to check the source
code if the tips don't make sense. Then, post an issue or a pull request to get
it fixed!

## Prerequisites

In order for the training scripts to work, you'll need to install the following
libraries and packages.

### OpenCV 3.4+

The scripts in this directory rely on a relatively modern version of [OpenCV](https://opencv.org/),
at least 3.4. You should be able to install the most recent versions through
`pip`:

```
python3 -m pip install opencv-python
python3 -m pip install opencv-contrib-python
```

### TensorFlow Object Detection API

All of our model configuration and training has been done with the [TensorFlow
Object Detection
API](https://github.com/tensorflow/models/tree/master/research/object_detection)
. The linked repository contains a number of documents for installation,
training (locally and in the cloud), configuring your own models, and more. Once
you've installed the API, we recommend working through one of the quick start
notebooks to understand more about the training process. Afterwards, you may
want to spend more time exploring the other documents. You should then take a
look at [this tutorial](
https://medium.com/tensorflow/training-and-serving-a-realtime-mobile-object-detector-in-30-minutes-with-cloud-tpus-b78971cf1193),
which walks you through training on TPUs in the cloud,
and deploying your trained model to an Android device with TensorFlow Lite,
which is the same underlying technology powering this library.

Once you are comfortable with the API and have picked your favorite training
method (e.g. locally, cloud, TPUs), the scripts in this directory start to
become useful.

Before continuing, you should double check that you've followed all of the
installation steps for the TensorFlow Object Detection API. In particular, make
sure the python scripts in the API are visible in your path. We added the
following lines to our `.bashrc` files. You may need to use a different path.

```
export MODEL_RESEARCH_DIR=/opt/tensorflow_models/research
export PYTHONPATH=${PYTHONPATH}:${MODEL_RESEARCH_DIR}
export PYTHONPATH=${PYTHONPATH}:${MODEL_RESEARCH_DIR}/slim
```

## Data Acquisition

As the first step to training your own model, you'll need to gather some data
for the specific task you're trying to solve. This is often the most time
consuming part of developing new machine learning algorithms, so we dedicated a
good bit of time to attempting to optimize the pipieline. Below is the solution
that we converged to after quite a bit of thought.

### Theory

When machine learning models are trained in industry or academia, they often
rely on datasets with millions of unique images, each of which has been
individually labeled by a human expert. The labeling process costs huge amounts
of time and money, both of which are far outside the budget of individuals
looking to train their own models.

Rather than trying to label individual images, we've found great success
labeling individual frames from videos. If frames are selected infrequently
enough, the results of this process are the same as above -- that is, a large
set of images, each individually annotated with object labels. However, the use
of video enables the use of an object tracker to automatically well formed data,
rather than requiring a human to manually annotate each frame. This makes it
possible for even a single person to gather sufficiently large datasets to train
robust models.

### Video Collection

There are a couple of objectives which should be met when trying to gather video
data. To achieve the best performance from the tracker (ensuring high quality
labels), as well as the best performance from the model (robustness, accuracy,
precision, etc.), you should try to ensure the following criteria (separated by
why they imapact more). A sample video which meets these criteria is available
at `train_data/train_example_vid.mp4`.

**Tracker:**

* Use relatively few objects in each video (1-5 is a good limit). This
  implicitly helps meet the remaining objectives.
* Ensure objects don't occlude each other for too long or too often. If they do,
  you may need to manually correct labels.
* Keep all objects in frame for the duration of the video. Objects leaving the
  frame will cause the tracker to lose objects, and force manual correction.

**Network:**

* Generally, try to collect data which is similar to the cases you expect to see
  when you're using the trained model for real. Also, use similar (the same, if
  possible) devices to collect your data (e.g. the same cell phone) as what you
  will be evaluating on.
* Collect videos of your objects from a variety of viewpoints and lighting
  conditions.
* Collect video from a variety of distances. That is, make sure you've been both
  close to and far from your objects in your videos.
* (optional) Use both portrait and landscape to record videos, ensuring maximum
  generalization. Not necessary if you're only ever using one of the two
  formats.
* (optional) Plant "similar looking" objects in the background, forcing the
  model to learn your objects better.

### Video Preprocessing

1. Sometimes (e.g. when collecting portrait videos), videos will have a rotation
   metadata tag set, rather than actually being rotated. For our videos, we've
   been able to use the `rotate_videos.py` script to rotate the videos into
   their actual orientations.
1. You may have some videos which break the above rules (e.g. severe occlusion
   or objects out of frame). Rather than throw the video away, you should slice
   the good sections and keep them. `ffmpeg` works great here.
1. The tracker seems to work better when initialized around larger objects (more
   features to track). If you're starting far away frmo an object but getting
   closer, you may wish to reverse the video to help the tracker along. `ffmpeg`
   works great here as well.

### Video Labeling

Now that you've acquired your perfect videos, you can label them as follows;

1. Use `find_bb.py` to label the first frame of a video. Make sure to keep
   bounding boxes as tight around the object as possible. Example invocation:

   ```
   python3 find_bb.py --help
   python3 find_bb.py train_data/train_example_vid.mp4
   ```

1. Use `tracking.py` to update the initial labels through the entire video,
   saving each individual data pair as a new set of files. Example invocation:

   ```
   python3 tracking.py --help
   python3 tracking.py train_data/train_example_vid.mp4 -s 1.2
   ```

In some cases, such as with real world videos (e.g. a robot's perspective during
a practice match), the above rules won't be followed, making it difficult for
the tracker to work. Instead, you can use `labeler.py` to step through the video
and label individual frames manually. Example invocation:

```
python3 labeler.py --help
python3 labeler.py train_data/train_example_vid.mp4
```

### Label Postprocessing

Even with careful video curation and bounding box annotation, there may be a few
frames where objects go out of scope, or the tracker fails to perfectly track
the objects. In these cases, you should find the resultant bad labels, and
remove the corresponding image (`.png`) and label (`.txt`) files. This helps
ensure the network is only receiving clean data.

### Conversion to TFRecord

For each saved frame, the above scripts will yield an image and a text file
containing the annotations for that frame. While this is a very easy format to
modify and visualize, it needs to be converted into something else to be used
with the Object Detection API. The `convert_labels_to_records.py` script
recursively walks a tree to find all such pairs, and converts them into a new
TFRecord. It can also automatically generate eval splits, or use an entirely
separate eval folder. Example invocation:

```
python3 convert_labels_to_records.py train_data -n 8 --eval
```

You should now have a set of train and eval `.record` files in the same
directory as your data (e.g. `train_data`).

## Model Selection

Choosing the correct model is often as important as having proper data
available. For efficient inference on a mobile device, we recommend using
something from the MobileNet + SSD family of detectors. The provided model was
trained on a MobileNet V1 + SSD architecture, with a 0.5 depthwise multiplier.
You can find the pipeline configuration for the model in the
`models/sample_mobilenet_v1_0.5_ssd_quantized` directory, as well as the 
pretrained checkpoint (trained on FTC game objects) which you can use to 
bootstrap your own training.

Some things to keep in mind when selecting a model:

* You'll almost certainly want to pick a model which makes pretrained
  checkpoints available. Pretraining networks on large, diverse datasets (e.g.
  ImageNet, MSCOCO), and then finetuning on a specific task (such as detecting
  cubes and balls) has been shown to significantly improve performance over
  training directly on the specific task.
* The inference time is independent of the weights. This means that if you have
  a specific timing requirement that you're trying to satisfy on your device
  (say, 400 ms inference), you can train a large set of models for only a few
  steps (1000 or so), and use that to find one which meets your timing
  requirements. Once you've found one, you can train that model for more steps.
* Processor clock speed has a close to linear impact on inference time. This
  means that a model which runs at a given speed on a 1.2 GHz processor phone
  (e.g. Moto G4 Play) will run nearly twice as fast on a 2.3 GHz phone (e.g.
  Nexus 5). Thus, make sure you perform any measurements on your own devices,
  since numbers you find online may not be directly applicable.

There is a lot more information about model selection in the Object Detection
API documentation, which you should certainly read through.

### Model Restrictions

The TFOD library currently expects output from the network in a specific format,
which matches the output of `export_tflite_ssd_graph.py` with the postprocessing
op enabled. This should work fine with all models which use a SSD head, but may
not work with different detection heads (e.g. FasterRCNN). If you choose to use
a completely different model, you will likely need to modify the postprocessing
function inside `RecognizeImageRunnable`.

## Training

You can now take the `.record` files you generated and use them in the same
training pipeline you were using earlier in the tutorials. As before, you'll
almost certainly want to fine tune an existing model which has already been
trained on a larger dataset (e.g. COCO), rather than training completely from
scratch with just your highly specific dataset.

This process can take anywhere from a few hours to a few days, depending on the
complexity of the model, hardware compute capacity, and quality of the data.
Make sure to allocate plenty of time for iteration!

## Checkpoint Conversion

The training process will result in a series of checkpoints recording the model
parameters at different times. You'll need to convert these checkpoints into a
more useful format, as discussed in the above tutorials. You can use the
`export_inference_graph.py` script to yield a model which can be used on the
desktop. If you are using SSD + Mobilenet, you can use the
`export_tflite_ssd_graph.py` and TOCO to generate a `.tflite` file. If you are
using a different model, you can do this process manually with other provided 
scripts in the Object Detection API. Both of the `export` scripts are already 
present in the Object Detection API (you may have used them in the object 
detection tutorial linked above).

Example invocation for `export_inference_graph.py`:

```
python3 $MODEL_RESEARCH_DIR/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path models/sample_mobilenet_v1_0.5_ssd_quantized/pipeline.config \
    --trained_checkpoint_prefix models/sample_mobilenet_v1_0.5_ssd_quantized/model.ckpt-200007 \
    --output_directory models/sample_mobilenet_v1_0.5_ssd_quantized/output_inference_graph
```

Example invocation for `export_tflite_ssd_graph.py`:

```
python3 $MODEL_RESEARCH_DIR/object_detection/export_tflite_ssd_graph.py \
    --pipeline_config_path models/sample_mobilenet_v1_0.5_ssd_quantized/pipeline.config \
    --trained_checkpoint_prefix models/sample_mobilenet_v1_0.5_ssd_quantized/model.ckpt-200007 \
    --output_directory models/sample_mobilenet_v1_0.5_ssd_quantized/tflite \
    --add_postprocessing_op=true
```

You'll then need to call the following, from the tensorflow directory:

```
bazel run -c opt tensorflow/contrib/lite/toco:toco -- \
    --input_file=[PATH TO THIS REPO]/training/models/sample_mobilenet_v1_0.5_ssd_quantized/tflite/tflite_graph.pb \
    --output_file=[PATH TO THIS REPO]/training/models/sample_mobilenet_v1_0.5_ssd_quantized/tflite/detect.tflite \
    --input_shapes=1,300,300,3 \
    --input_arrays='normalized_input_image_tenor' \
    --ouptut_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
    --inference_type=QUANTIZED_UINT8 \
    --mean_values=128 \
    --std_values=128 \
    --change_concat_input_ranges=false \
    --allow_custom_ops
```

For reference, the sample `detect.tflite` file is provided in
`models/sample_mobilenet_v1_0.5_ssd_quantized/detect.tflite`.

## Visualization

Before deploying your model on a phone, it's a good idea to get a rough sense
for its per frame performance. The `camera_cv.py` script executes your model
against every single frame in a video and displays the results to the screen in
real time. It also makes for slightly more portable demonstrations! Example
invocation:

```
python3 camera_cv.py \
    --movie validation_data/validation_example_vid.mp4 \
    --path_to_model models/sample_mobilenet_v1_0.5_ssd_quantized/output_inference_graph/frozen_inference_graph.pb
```

## Deploy Model

Congratulations! You've been able to train a new model successfully. For steps
on how to use this new model of yours with the TFOD library, please see the
README file in the top level `/TFObjectDetector` directory.

## Miscellaneous Tips

Throughout the development cycle, we picked up a number of tips on improving
object detector model performance. We've presented some of them below, in no
particular order:

* While you can successfully fine tune a network with as little as 100 positive
  examples from each class, a better target is approximately 1000 positive
  examples of each class. However, since your data will be more correlated than
  normal (we normally assume an iid dataset), you should ideally target
  approximately 10k positive examples of each class. Our network was trained
  with approximately 3500 positive examples of each class.
* Try using negative examples, which are images from your domain (e.g. still
  from the robot), but which do not have any of the objects of interest. An
  optimistic target is approximately 90% negative examples and 10% positive
  exmples.
* Not every class needs to be present in every single video. Thus, you may find
  it easier to collect a series of videos of individual objects, to
  automatically avoid any occlusion problems.
