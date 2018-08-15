# TensorFlowObjectDetector (TFOD) for FTC

TODO(vasuagrawal): [insert a pretty video / gif here]

## What is this?

This repository contains an Android library which enables FTC teams to use
machine learning in their OpModes. Specifically, this library makes it possible
to use neural networks to do object detection on camera frames. This library
requires very little setup, and once running will update recognitions in the
background without user interaction, enabling the user to focus on other tasks.
Furthermore, this repository contains scripts to enable teams to collect and
annotate their own datasets, should they wish to learn more or experiment with
different models.

## Why did you do this?

Perhaps a better question is what we're hoping to accomplish by releasing this
library. In no particular order, here are some of our goals.

1. **Democratize machine learning.** We believe machine learning is becoming a
   fundamental technology that everyone (e.g. all FTC teams) should be able to
   benefit from and have access to, and this is another step in that direction.
1. **Enable higher levels of autonomy.** It's no secret that the reason for
   machine learning's success has been its unparalleled performance and accuracy
   across a wide range of real world tasks. By bringing some of this technology
   to FTC, we hope to push the limits of what autonomous robots can do.
1. **Educate people about ML.** We want students to learn about the whole
   process of using and developing machine learning models, to help them better
   understand the true extents of what is possible.

## How does it work?

The library's operation can be thought of as being broken into a few distinct
steps:

1. Initialize a source for image frames
1. Load the labels and models into memmory
1. Start background jobs and other objects

Then, the following runs indefinitely:

1. Get a new image frame
1. If possible, send the frame to the neural network
1. Feed the frame through an object tracker
1. If the neural network is done, feed its results into the tracker
1. Make the most recent recognitions available to the user

A more detailed understanding of each of the steps involved can be found in the
documentation and comments for each of the parts of the system.

## How do I get started?

Depending on when you are reading this, this library may or may not already be
prepackaged with the robot controller app. If that is the case, then you can
jump directly to the usage guidelines. If not, or if you're trying to use this
library outisde of the FTC codebase (which is supported, acceptable, and
encouraged!), you'll first want to follow the steps in the Installation section.
You can then continue to the usage section, as above.

## Acknowledgements

We would like to thank Aakanksha Chowdhery, Vivek Rathod, and Ronny Votel for
their help and support with TensorFlow Lite and the TensorFlow Object Detection
API. We would also like to think David Konerding and Liz Looney for their
mentorship throughout the development cycle. Finally, we would like to thank
everyone involved in coordinating and participating in our data collection
event, including Patricia Cruz, Aaron Cunningham, Nathan Mulcahey, Calvin
Johnson, and FTC teams 8381, 11039, 12454, 12635, 12869, 13799.
