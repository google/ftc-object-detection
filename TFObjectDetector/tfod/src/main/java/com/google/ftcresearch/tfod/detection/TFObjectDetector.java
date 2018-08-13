/*
 * Copyright (C) 2018 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ftcresearch.tfod.detection;

import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.content.res.Resources;
import android.graphics.Canvas;
import android.os.Build;
import android.support.annotation.NonNull;
import android.util.Log;

import com.google.ftcresearch.tfod.generators.FrameGenerator;
import com.google.ftcresearch.tfod.util.AnnotatedYuvRgbFrame;
import com.google.ftcresearch.tfod.util.Rate;
import com.google.ftcresearch.tfod.util.YuvRgbFrame;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;

/**
 * Class to convert object detection and tracking system into a simple interface.
 *
 * <p>TFObjectDetector makes it easy to detect and track objects in real time. After initialization,
 * clients simply call getRecognitions() as often as they wish to get the recognitions corresponding
 * to the most recent frame which has been processed. Recognitions contain information about the
 * location, class, and detection confidence of each particular object.
 *
 * <p>Advanced users may wish to tune the performance of the TFObjectDetector by changing parameters
 * away from the defaults in {@link TfodParameters}. Not all parameters will make a measurable
 * impact on performance.
 */
// TODO(vasuagrawal): Update this class to provide statistics and latency information.
public class TFObjectDetector {

  private static final String TAG = "TFObjectDetector";

  // Parameters passed in through the constructor.
  private final TfodParameters params;
  private final FrameGenerator frameGenerator;
  private final AnnotatedFrameCallback clientCallback;

  // Parameters created when loading the models.
  // TODO(vasuagrawal): Modify loading so that these fields can be final.
  private List<Interpreter> interpreters;
  private List<String> labels;

  // Parameters created during initialization.
  private TfodFrameManager frameManager;
  private Thread frameManagerThread;
  private final Rate rate;

  // Store all of the data relevant to a set of recognitions together in an AnnotatedYuvRgbFrame,
  // guarded by annotatedFrameLock for when multiple callbacks attempt to update the
  // annotatedFrame, or multiple clients attempt to access it.
  private final Object annotatedFrameLock = new Object();
  private AnnotatedYuvRgbFrame annotatedFrame;
  private long lastReturnedFrameTime = 0;

  private static String getFilenameWithoutExtension(String filename) {
    int lastIndex = filename.lastIndexOf('.');
    return (lastIndex == -1) ? filename : filename.substring(0, lastIndex);
  }

  /** Return whether this device (android version, specs, etc.) are compatible with the library. */
  public static boolean isDeviceCompatible() {
    // To comply with FTC min version, throw an exception if the android version is less than 23.
    // This allows the library to be available on all devices, but only actually usable in 6.0+.
    return Build.VERSION.SDK_INT >= Build.VERSION_CODES.M;
  }

  public TFObjectDetector(TfodParameters params, FrameGenerator frameGenerator) {
    this(params, frameGenerator, (annotatedFrame) -> {});
  }

  public TFObjectDetector(
      TfodParameters params,
      FrameGenerator frameGenerator,
      AnnotatedFrameCallback clientCallback) {

    if (!isDeviceCompatible()) {
      throw new RuntimeException("TFOD library requires Android 6.0+. Please upgrade.");
    } else {
      Log.d(TAG, "Android version is supported! Continuing.");
    }

    this.params = params;
    this.frameGenerator = frameGenerator;
    this.clientCallback = clientCallback;
    this.rate = new Rate(params.maxFrameRate);

    try {
      // Try to initialize the stored frame to something non-null. This also ensures that any
      // asynchronous setup being done in the frameGenerator gets done before the frame manager
      // starts, so there's no unexpected delays there.
      YuvRgbFrame frame = frameGenerator.getFrame();
      annotatedFrame = new AnnotatedYuvRgbFrame(frame, new ArrayList<>(), System.nanoTime());
    } catch (InterruptedException e) {
      // TODO(vasuagrawal): Figure out if this is the right exception / behavior.
      throw new RuntimeException("TFObjectDetector constructor interrupted while getting frame!");
    }
  }

  /** Memory-map the model file in Resources as read only */
  private static MappedByteBuffer loadModel(Activity activity, String filename) throws IOException {

    final Resources res = activity.getResources();

    int modelNameId =
        res.getIdentifier(getFilenameWithoutExtension(filename), "raw", activity.getPackageName());

    AssetFileDescriptor fileDescriptor = res.openRawResourceFd(modelNameId);

    FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
    FileChannel fileChannel = inputStream.getChannel();

    long startOffset = fileDescriptor.getStartOffset();
    long declaredLength = fileDescriptor.getDeclaredLength();

    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
  }

  /**
   * Load each of the labels from a file into an ArrayList.
   *
   * This function expects the file to have a single label on each line, with the number of
   * labels (lines) corresponding to the number of classes the model can identify, ignoring the
   * background class. That is, if the model is trained on 2 classes, the file should look
   * something like this:
   *
   * <pre>
   *   class A
   *   class B
   * </pre>
   *
   * Instead of like this:
   *
   * <pre>
   *   ??? (background class)
   *   class A
   *   class B
   * </pre>
   * */
  private static ArrayList<String> loadLabels(Activity activity, String filename)
      throws IOException {

    final Resources res = activity.getResources();
    final ArrayList<String> labels = new ArrayList<>();

    int labelNameId =
        res.getIdentifier(getFilenameWithoutExtension(filename), "raw", activity.getPackageName());

    try (BufferedReader br =
        new BufferedReader(new InputStreamReader(res.openRawResource(labelNameId),
            StandardCharsets.UTF_8))) {
      String line;
      while ((line = br.readLine()) != null) {
        labels.add(line);
        Log.d(TAG, "Added label: " + line);
      }
    }

    return labels;
  }

  /** Load the models and labels, and create as many interpreters as necessary. */
  private void loadResources(Activity activity) throws IOException {

    Log.i(TAG, "Loading the labels.");
    labels = loadLabels(activity, params.labelName);

    Log.i(TAG, "Loading the interpreters.");

    final MappedByteBuffer modelData = loadModel(activity, params.modelName);
    interpreters = new ArrayList<>();

    for (int i = 0; i < params.numExecutorThreads; i++) {
      interpreters.add(new Interpreter(modelData, params.numInterpreterThreads));
    }
  }

  // TODO(vasuagrawal): Figure out if this should be a Context instead of an Activity
  public void initialize(Activity activity) throws IOException {
    Log.i(TAG, "TFOD thread name: " + Thread.currentThread().getName());
    loadResources(activity);

    // Create a TfodFrameManager, which handles feeding tasks to the executor. Each task consists
    // of processing a single camera frame, passing it through the model (via the interpreter),
    // and returning a list of recognitions.
    frameManager =
        new TfodFrameManager(
            frameGenerator,
            interpreters,
            labels,
            params,
            (receivedAnnotatedFrame) -> {

              // Run this first so that we don't have any chance of data corruption, and we're not
              // holding the lock forever.
              clientCallback.onResult(receivedAnnotatedFrame);

              synchronized (annotatedFrameLock) {
                Log.v(TAG, "Frame change: setting a new annotatedFrame");
                annotatedFrame = receivedAnnotatedFrame;
              }
            });
    Log.i(TAG, "Starting frame manager thread");
    frameManagerThread = new Thread(frameManager, "FrameManager");
    frameManagerThread.start();
  }

  /** Perform whatever cleanup is necessary to release all acquired resources.
   *
   * <p> Note: TFObjectDetector does not claim ownership of the FrameGenerator. As such,
   * any responsibility for its cleanup will be on the caller, not the TFObjectDetector.
   */
  public void shutdown() {
    frameManagerThread.interrupt();
  }

  /**
   * Take the most up to date annotatedYuvRgbFrame available, blocking if there isn't one.
   *
   * <p>If tracking is enabled, the recognitions will always be from the most recent frame
   * processed by the tracker. If tracking is disabled, the recognitions will correspond to the
   * most recent frame processed by the model. There is currently no way to receive the raw
   * recognitions from the model if tracking is enabled.
   *
   * <p>If this method is called multiple times without the TFObjectDetector receiving a new
   * detection, the object returned will be the same. In this case, the client is free to modify the
   * returned object, as the TFObjectDetector promises to not modify the annotatedYuvRgbFrame.
   *
   * This method may internally sleep if it is being called too quickly. The maximum frame rate
   * at which this function will return is determined by the maxFrameRate parameter. That is, if
   * returning immediately would violate the maxFrameRate (last return was too recent), this
   * method will sleep to correct the time difference. The sleep is checked internally, so this
   * method does not throw. Rather, the interrupted flag is set upon interruption.
   *
   * @return A list of the most recent recognitions available.
   */
  public @NonNull AnnotatedYuvRgbFrame takeAnnotatedFrame() {
    rate.sleep();

    synchronized (annotatedFrameLock) {
      return annotatedFrame;
    }
  }

  /**
   * Poll for a new AnnotatedYuvRgbFrame, returning null if a new one isn't available.
   *
   * If a new frame has arrived since the last time this method was called, it will be returned.
   * Otherwise, null will be returned.
   *
   * Note that this method still takes a lock internally, and thus calling this method too
   * frequently may degrade performance of the detector.
   *
   * @return A new frame if one is available, null otherwise.
   */
  public AnnotatedYuvRgbFrame pollAnnotatedFrame() {
    synchronized (annotatedFrameLock) {
      // Can only do this safely because we know the annotatedFrame can never be null after the
      // constructor has happened.
      if (annotatedFrame.getFrameTimeNanos() > lastReturnedFrameTime) {
        lastReturnedFrameTime = annotatedFrame.getFrameTimeNanos();
        return annotatedFrame;
      }
    }

    return null;
  }

  /**
   * Draw both normal and debug information onto the provided canvas.
   *
   * This method is a superset of draw(), so the user will only need to call one of the two methods.
   * @param canvas Canvas to draw on.
   */
  public void drawDebug(Canvas canvas) {
    frameManager.drawDebug(canvas);
  }

  public void draw(Canvas canvas) {
    frameManager.draw(canvas);
  }
}
