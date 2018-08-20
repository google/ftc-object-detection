# TensorFlow Object Detector (TFOD) Library

This directory contains the TFOD library, implemented as a module as a part of a
demo application. All of the code in this library and application is completely
independent of any FTC source code, and thus can be used for non-FTC projects,
or projects outside the FTC SDK, if desired.

TODO(vasuagrawal): Insert screenshot of the app running here

## Installation

If you're developing with the FTC robot controller app, there's a very good
chance that this library is already included. In that case, you should just be
able to start using this library, as described in the Getting Started section.
If not, follow these steps to install the library in your own app.

1. Ensure that you are able to build the app contained in this folder. To do so,
   you should import the app into Android Studio and build module `app` from
   there. You may need to install a specific version of the Android API and SDK.
1. Test the app by installing it on your Android device, or in an emulator.
   While this step isn't strictly necessary, it's a good way to get an intuition
   for the rough expected performance on your device.
1. Build the `assemble` task for module `tfod`, either through Android Studio,
   or by running `./gradlew :tfod:asssemble` from this directory.
1. The `assemble` task creates a `tfod-release.aar` and `tfod-sources.jar`. The
   release aar is the only one you actually need to use this library, but your
   IDE may be able to use the sources jar for better code completion. Copy the
   relevant files into the `libs` folder of your app. You may first need to make
   a `libs` folder.

   ```
   cp tfod/build/outputs/aar/tfod-release.jar <YOUR APPLICATION ROOT>/libs
   cp tfod/build/libs/tfod-sources.jar <YOUR APPLICATION ROOT>/libs
   ```

1. In your top level `build.gradle` file, you'll want to make sure you have your
   local `libs` folder included as a repository:

   ```
   repositories {
       flatDir {
           dirs 'libs'
       }
   }
   ```

At this point, TFOD should be available as a library for you to use within your
application. However, you will need to make one additional change in order to
prevent a runtime crash. For technical reasons, model files (`.tflite`) must be
stored as uncompressed. To do this, add the following lines to your module level
`build.gradle`:

```
aaptOptions {
    noCompress "tflite"
}
```

You should now have everything necessary to start using the library!

## Getting Started

The basic usage of the library is very simple:

1. Create and initialize a `FrameGenerator` (see below for more information).
1. Construct a `TFObjectDetector` with the newly created `FrameGenerator`, and
   an empty `TfodParameters` object (made with `new
   TfodParameters.Builder().build()`).
1. Initialize the `TFObjectDetector`.
1. Contintually check for recognitions from `takeAnnotatedFrame()` from the
   `TFObjectDetector`.
1. Clean up the `TFObjectDetector` and `FrameGenerator` upon shutdown, via
   `shutdown()` and `onDestroy()` respectively.

This will look something like this:

```
FrameGenerator frameGenerator = new MovingImageFrameGenerator(bitmap);
TFObjectDetector tfod = new TFObjectDetector(
    new TfodParameters.Builder().build(), frameGenerator);
tfod.initialize(this); // where "this" is a Context

while (true) {
    AnnotatedYuvRgbFrame annotatedFrame = tfod.takeAnnotatedFrame();

    // Do something interesting with the frame
}

tfod.shutdown();
frameGenerator.onDestroy();
```

The above loop will get annotated frames at some predetermined rate, and "do
something interesting" with the results. Each `AnnotatedYuvRgbFrame` contains
the image (frame) itself, the recognitions on that frame, if any, and the time
stamp of the frame.

For a more complete example, please take a look at the example app at
`app/src/main/java/com/google/ftcresearch/MainActivity.java`.

## Customization and Tuning

While the TFOD library is designed to be as simple as possible to start using,
careful consideration was also placed on extensibility, customization, and
tuning. This section summarizes some of the ways you can personalize this
library. Of course, since the library is open source, you can make any changes
you'd like. If you have new features, please submit a pull request!

### Custom `FrameGenerator`s

A `FrameGenerator` is used internally by the library to get frames to be
processed. While there are a couple provided `FrameGenerator`s, you may have a
use case which doesn't neatly fit into one of the provided ones. You can simply
create a new object implementing the `FrameGenrator` interface, and the library
will happily use your new image source. Here are some ideas:

* Generate frames from a video stored as a raw resource
* Stream an internet video (e.g. YouTube) into frames
* Use an external camera, rather than the built in cameras

For more details on how to implement your own `FrameGenerator`, please take a
look at the documentation for the interface, as well as the provided example
`FrameGenerator`s.

### Different `TfodParameter`s

There are quite a few parameters used throughout the library to control all
aspects of performance and behavior. Reasonable efforts have been made to
provide sane defaults for all of these parameters, but all parameters are tuned
relatively conservatively. You may find that you can improve the performance of
the library on your specific device with a different set of parameters.
Alternatively, the defaults may enable some functionality which you would prefer
to turn off. In most cases, you are able to do this.

All tunable parameters in the library can be configured through the
`TfodParameter.Builder` class. Some examples of tunable parameters and features:

* Enable / disable object tracker
* Number of parallel interpreters
* Number of threads per interpreter

Each available parameter has been documented in the source, and you are
encouraged to look at the source code to learn about all possible options.

### New object detection model

The provided model was trained on approximately 4000 images (3500 examples of
each of the 2 classes). Advanced users may wish to further refine the provided
model with additional data they gather themselves. Users may also want to tune
for different performance and latency characteristics (the provided model tried
to optimize performance at an inference time of 300 ms). Users may further wish
to train new models on different objects entirely, perhaps for future FTC games,
or simply for their own exploration.

Using a new model is as simple as adding the correct `.tflite` and label files
as raw resources, and then passing the proper identifiers and model parameters
to the constructor for `TFObjectDetector` (via `TfodParameters`).

For more information on training your own model, please see the top level
`/training` folder.

## Gotchas

When developing against this library, there are some things that you might run
into that you should be careful about. In no particular order, here are some of
the most common items:

* **UI Threads** -- Setup for some FrameGenerators might require running on a UI
  thread rather than in a background thread. Additionally, drawing recognitions
  (if desired) to the screen must also be done in a UI thread.
* **Permissions** -- Newer versions of Android require more care when handling
  permissions (e.g. runtime permissions), which you may need to deal with if
  using non standard permissions.
* **Cleanup** -- It is imperative that the library is shut down through its
  `shutdown()` method, so that it has ample time to free up resources.
  Furthermore, you must remember that you, the user, are responsible for
  separately closing the `FrameGenerator` (with `onDestroy()`) if necessary.
