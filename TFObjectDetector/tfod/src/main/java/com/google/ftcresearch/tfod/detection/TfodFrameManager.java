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

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.Log;

import com.google.ftcresearch.tfod.generators.FrameGenerator;
import com.google.ftcresearch.tfod.util.AnnotatedYuvRgbFrame;
import com.google.ftcresearch.tfod.util.ImageUtils;
import com.google.ftcresearch.tfod.util.Recognition;
import com.google.ftcresearch.tfod.util.RollingAverage;
import com.google.ftcresearch.tfod.util.Timer;
import com.google.ftcresearch.tfod.util.YuvRgbFrame;
import com.google.ftcresearch.tfod.tracking.MultiBoxTracker;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Queue;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.tensorflow.lite.Interpreter;

/**
 * Class to indefinitely read frames and return the most recent recognitions via a callback.
 *
 * <p>The TfodFrameManager constantly reads frames from the supplied {@link FrameGenerator}. These
 * frames are then periodically passed through a {@link RecognizeImageRunnable} to get the
 * recognitions for that frame asynchronously. The TfodFrameManager also passes all frames through a
 * {@link MultiBoxTracker}, if specified in the parameters. The resultant recognitions (from the
 * tracker or the runnable) are then passed into the {@link AnnotatedFrameCallback}.
 */
class TfodFrameManager implements Runnable {

  private static final String TAG = "TfodFrameManager";
  private static final Paint paint = new Paint(); // Used to draw recognitions without tracker

  static {
    paint.setColor(Color.RED);
    paint.setStyle(Paint.Style.STROKE);
    paint.setStrokeWidth(10);
  }

  // Parameters passed in to the constructor
  private final TfodParameters params;
  private final ExecutorService executor;
  private final FrameGenerator frameGenerator;
  private final List<Interpreter> interpreters;
  private final List<String> labels;
  private final AnnotatedFrameCallback tfodCallback;

  // Output locations for the interpreters, so we're not constantly reallocating
  private final List<float[][][]> outputLocations = new ArrayList<>();
  private final List<float[][]> outputClasses = new ArrayList<>();
  private final List<float[][]> outputScores = new ArrayList<>();
  private final List<float[]> numDetections = new ArrayList<>();

  private final Queue<Integer> availableIds = new ConcurrentLinkedQueue<>();
  private final RollingAverage averageInferenceTime;
  private long lastSubmittedFrameTimeNanos;

  private final Object lastRecognizedFrameLock = new Object();
  private volatile AnnotatedYuvRgbFrame lastRecognizedFrame; // The most recent returned frame

  private final MultiBoxTracker tracker;

  TfodFrameManager(
      FrameGenerator frameGenerator,
      List<Interpreter> interpreters,
      List<String> labels,
      TfodParameters tfodParameters,
      AnnotatedFrameCallback tfodCallback) {
    this.frameGenerator = frameGenerator;
    this.interpreters = interpreters;
    this.labels = labels;
    this.params = tfodParameters;
    this.tfodCallback = tfodCallback;
    this.executor = Executors.newFixedThreadPool(params.numExecutorThreads);
    this.averageInferenceTime = new RollingAverage(params.timingBufferSize);
    this.tracker = params.trackerDisable ? null : new MultiBoxTracker(params);

    // Create the output arrays for the different interpreters
    for (int i = 0; i < params.numExecutorThreads; i++) {
      outputLocations.add(new float[1][params.maxNumDetections][4]);
      outputClasses.add(new float[1][params.maxNumDetections]);
      outputScores.add(new float[1][params.maxNumDetections]);
      numDetections.add(new float[1]);
    }

    // Mark all of the interpreters as available.
    for (int i = 0; i < params.numExecutorThreads; i++) {
      Log.d(TAG, "Adding interpreter: " + i);
      availableIds.add(i);
    }

    // TODO(vasuagrawal): Do one inference task and get an inference time to use as a seed.
    // Make sure one inference task is done here before doing in the executor, so that we can
    // have a somewhat accurate estimate for the rollingAverage seed (which keeps all the
    // executors spaced evenly). The alternative is to just pick some reasonable value based on
    // experimental data (e.g. 300 ms), or just let it be 0 and let the system adjust automatically.
  }

  /** Transform all locations in the source list of recognitions by m, returning a new list */
  private List<Recognition> transformRecognitionLocations(List<Recognition> source, Matrix m) {
    List<Recognition> output = new ArrayList<>();
    for (Recognition recognition : source) {

      // It's fine to modify the location in place since getLocation() returns a copy.
      RectF location = recognition.getLocation();
      m.mapRect(location);

      output.add(new Recognition(recognition.getLabel(), recognition.getConfidence(), location));
    }

    return output;
  }

  private void receiveNewRecognitions(AnnotatedYuvRgbFrame annotatedFrame) {
    for (Recognition recognition : annotatedFrame.getRecognitions()) {
      Log.d(TAG, "Received: " + recognition);
    }

    synchronized (lastRecognizedFrameLock) {
      if (lastRecognizedFrame == null ||
          annotatedFrame.getFrameTimeNanos() > lastRecognizedFrame.getFrameTimeNanos()) {
        Log.v(TAG, "Setting a new annotated frame.");
        lastRecognizedFrame = annotatedFrame;
      } else {
        Log.w(TAG, "Received an out of order recognition. Something is likely wrong!");
        return; // We don't want to process / send this frame anywhere.
      }
    }

    if (!params.trackerDisable) {
      Log.d(TAG, "Sending received recognitions to the tracker");
      final List<Recognition> recognitions;
      final byte[] yuvFrame;

      if (params.trackerFrameResizeEnable) {
        Timer timer = new Timer(TAG);
        timer.start("Preprocessing for tracker update in receive");
        // To support tracker resizing, we need to get the resized frame and transform the
        // locations on the recognitions.
        yuvFrame = annotatedFrame.getFrame().resize(params.trackerFrameSize).getLuminosity();

        // Convert the recognitions to the tracker frame coordinates
        Matrix originalToTrackerTransform =
            ImageUtils.transformBetweenImageSizes(annotatedFrame.getFrame().getSize(),
                params.trackerFrameSize);
        recognitions = transformRecognitionLocations(annotatedFrame.getRecognitions(),
            originalToTrackerTransform);
        timer.end();
      } else {
        // Not performing any tracker resizing, just use the original stuff.
        recognitions = annotatedFrame.getRecognitions();
        yuvFrame = annotatedFrame.getFrame().getLuminosity();
      }

      tracker.trackResults(recognitions, yuvFrame, annotatedFrame.getFrameTimeNanos());
    } else {
      Log.d(TAG, "Directly calling tfod callback, skipping tracker.");
      tfodCallback.onResult(annotatedFrame);
    }
  }

  private void submitRecognitionTask(final AnnotatedYuvRgbFrame annotatedFrame) {

    // See if there's an interpreter available to handle this frame.
    final Integer interpreterId = availableIds.poll();

    if (interpreterId != null) { // There's actually an available interpreter, we will use it
      RecognizeImageRunnable task =
          new RecognizeImageRunnable(
              annotatedFrame,
              interpreters.get(interpreterId),
              params,
              labels,
              outputLocations.get(interpreterId),
              outputClasses.get(interpreterId),
              outputScores.get(interpreterId),
              numDetections.get(interpreterId),
              (recognitions) -> {
                long endTimeNanos = System.nanoTime();
                long elapsedNanos = endTimeNanos - annotatedFrame.getFrameTimeNanos();
                long elapsedMs = TimeUnit.MILLISECONDS.convert(elapsedNanos, TimeUnit.NANOSECONDS);
                Log.i(TAG, "[" + interpreterId + "] Ran for " + elapsedMs + " ms");

                averageInferenceTime.add(elapsedNanos);
                Log.i(
                    TAG,
                    "Average inference time: "
                        + TimeUnit.MILLISECONDS.convert(
                            (long) averageInferenceTime.get(), TimeUnit.NANOSECONDS)
                        + "ms");

                receiveNewRecognitions(annotatedFrame);

                // Finally, mark the interpreter as available.
                availableIds.add(interpreterId);
              });

      Log.i(TAG, "Submitting recognition task with interpreter " + interpreterId);
      lastSubmittedFrameTimeNanos = annotatedFrame.getFrameTimeNanos();
      executor.submit(task);
    } else {
      // TODO(vasuagrawal): Add this to the statistics made available in the future (dropped frames)
      Log.d(TAG, "No available interpreters");
    }
  }

  /**
   * Determine whether enough time has elapsed since we last submitted a frame.
   *
   * <p>The specified parameters for object detection may potentially allow more than a single
   * executor thread. In that event, it becomes possible to pipeline model evaluation (through
   * {@link RecognizeImageRunnable}) to minimize the latency between receiving recognitions. This
   * object maintains a RollingAverage of the compute time spent per frame. Dividing this average
   * latency by the number of executor threads yields a minimum time that should elapse between
   * frames submitted to the executor service to ensure that recognitions are being returned as
   * evenly spaced in time as possible, rather than in burst loads.
   *
   * @param frameTimeNanos Time (in nanoseconds) to determine if enough time has elapsed from.
   */
  private boolean enoughInterFrameTimeElapsed(final long frameTimeNanos) {

    final long elapsedNanos = frameTimeNanos - lastSubmittedFrameTimeNanos;
    final long minimumTimeNanos = (long) averageInferenceTime.get() / params.numExecutorThreads;

    return elapsedNanos >= minimumTimeNanos;
  }

  /**
   * Convert a frame to an appropriate format for the tracker (luminance) and submit it.
   *
   * @param annotatedFrame Annotated (with timestamp) input frame to send to tracker.
   */
  private void submitFrameToTracker(AnnotatedYuvRgbFrame annotatedFrame) {

    // Potentially give the tracker a resized version of the current frame.
    final YuvRgbFrame frame;
    if (params.trackerFrameResizeEnable) {
      frame = annotatedFrame.getFrame().resize(params.trackerFrameSize);
    } else {
      frame = annotatedFrame.getFrame();
    }

    final long frameTimeNanos = annotatedFrame.getFrameTimeNanos();

    // The UV flip doesn't matter here since the Y channel is always first.
    byte[] yuvFrame = frame.getLuminosity();
    tracker.onFrame(frame.getWidth(), frame.getHeight(), frame.getWidth(), 0, yuvFrame, frameTimeNanos);
    Log.d(TAG, "Submitted frame to tracker at time " + frameTimeNanos + " ns!");
  }

  /**
   * Constantly get frames, (maybe) pass them to an interpreter and through a tracker.
   *
   * <p>First, a frame is pulled from the frameGenerator. If enough time has elapsed since the last
   * time a frame was submitted {@see TfodFrameManager::enoughInterFrameTimeElapsed}, the current
   * frame gets submitted to an available interpreter (if any). Furthermore, every frame is passed
   * through the tracker, which is used to help interpolate recognitions between outputs from
   * interpreters, as well as to compensate for the latency of running the network. Finally, after
   * passing the frame through the tracker, the most recent list of recognitions (what the tracker
   * currently believes) is returned through the callback.
   */
  @Override
  public void run() {
    Log.i(TAG, "Frame manager thread name: " + Thread.currentThread().getName());
    Timer timer = new Timer(TAG);

    while (!Thread.currentThread().isInterrupted()) {
      // First, grab the frame.
      timer.start("Waiting for frame");
      final YuvRgbFrame frame;
      try {
        frame = frameGenerator.getFrame();
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        return;
      }

      final long frameTimeNanos = System.nanoTime();
      timer.end();

      Log.d(TAG, "Got an image from frame generator");
      AnnotatedYuvRgbFrame annotatedFrame = new AnnotatedYuvRgbFrame(frame, new ArrayList<>(),
          frameTimeNanos);

      // Then determine if we've waited long enough to submit the frame
      if (enoughInterFrameTimeElapsed(frameTimeNanos)) {
        Log.i(TAG, "Trying to submit recognition task (pending interpreter)");
        timer.start("Submitting recognition task");
        submitRecognitionTask(annotatedFrame);
        timer.end();
      } else {
        Log.d(TAG, "Not enough time has elapsed");
      }

      // If the tracker isn't disabled, feed it the newest frame, and then pass the results back up.
      if (!params.trackerDisable) {
        submitFrameToTracker(annotatedFrame);
        tracker.printResults();

        timer.start("Preprocessing for tracker in main loop");
        final List<Recognition> recognitions;
        if (params.trackerFrameResizeEnable) {
          // Map the recognitions back to original coordinates.
          final Matrix trackerToOriginalTransform =
              ImageUtils.transformBetweenImageSizes(params.trackerFrameSize, frame.getSize());
          recognitions = transformRecognitionLocations(tracker.getRecognitions(),
              trackerToOriginalTransform);
        } else {
          recognitions = tracker.getRecognitions();
        }
        timer.end();

        tfodCallback.onResult(new AnnotatedYuvRgbFrame(frame, recognitions, frameTimeNanos));
      }
    }

    // Make sure we clean up executor before returning from this thread.
    if (!executor.isShutdown()) {
      executor.shutdown();
      try {
        executor.awaitTermination(100, TimeUnit.MILLISECONDS);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
      }
    }
  }

  /** Draw both normal and debug information to the canvas. */
  void drawDebug(Canvas canvas) {

    // Draw first so debug information gets put on top.
    draw(canvas);

    if (!params.trackerDisable) {
      tracker.drawDebug(canvas);
    } // There's no debug information without the tracker
  }

  /**
   * Only draw recognitions to the canvas.
   *
   * If the tracker is enabled, this will pass through to the tracker's draw implementation,
   * which uses different colors for each tracked object (changing colors when a new association
   * is made). If the tracker is not enabled, each object will simply be boxed in red.
   * */
  void draw(Canvas canvas) {
    if (!params.trackerDisable) {
      tracker.draw(canvas);
    } else {
      final AnnotatedYuvRgbFrame annotatedFrame = lastRecognizedFrame;

      if (annotatedFrame != null) {
        for (Recognition recognition : lastRecognizedFrame.getRecognitions()) {
          canvas.drawRect(recognition.getLocation(), paint);
        }
      }
    }
  }
}
