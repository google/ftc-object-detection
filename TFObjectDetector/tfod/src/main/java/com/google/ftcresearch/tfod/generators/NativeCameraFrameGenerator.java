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

package com.google.ftcresearch.tfod.generators;

import android.content.Context;
import android.hardware.Camera;
import android.support.annotation.NonNull;
import android.util.Log;
import android.util.Pair;
import android.widget.FrameLayout;


import com.google.ftcresearch.tfod.util.YuvRgbFrame;
import com.google.ftcresearch.tfod.util.Size;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;

public class NativeCameraFrameGenerator implements FrameGenerator {

  private static final String TAG = "NativeCameraGen";

  private final Camera camera;
  private final CameraPreview cameraPreview;

  private final Size cameraSize;

  private final BlockingQueue<YuvRgbFrame> frameQueue = new ArrayBlockingQueue<>(1);

  private static Size getBestSize(List<Size> sizes, int minSize, float aspectRatio) {
    // Return the "best" size, which is the smallest one that meets minSize in both dimensions

    List<Pair<Integer, Size>> aspectRatioPairs = new ArrayList<>();
    for (Size size : sizes) {
      if (size.width >= minSize && size.height >= minSize) {
        float aspectRatioDiff = Math.abs(aspectRatio - ((float) size.width / size.height)) * 100;
        aspectRatioPairs.add(new Pair<>((int) aspectRatioDiff, size));
        Log.v(TAG, String.format("Size (%d x %d) has diff %f", size.width, size.height,
            aspectRatioDiff));
      }
    }

    if (aspectRatioPairs.isEmpty()) {
      throw new IllegalArgumentException("No sizes given which meet criteria!");
    }

    Collections.sort(aspectRatioPairs, (a, b) -> {
      if (a.first < b.first) {
        return -1;
      } else if (a.first > b.first) {
        return 1;
      } else { // Equal. Can just compare widths.
        return Integer.compare(a.second.width, b.second.width);
      }
    });

    for (Pair pair : aspectRatioPairs) {
      Log.v(TAG, "Sorted: " + pair.first + ", " + pair.second);
    }

    return aspectRatioPairs.get(0).second;
  }

  public NativeCameraFrameGenerator(Context context, FrameLayout layout, int minSize, float aspectRatio) {

    try {
      camera = Camera.open();
    } catch (Exception e) {
      throw new RuntimeException("Unable to open camera", e);
    }

    // Note that this only changes the preview, and does not affect the byte order passed into
    // the callback.
    camera.setDisplayOrientation(90);

    List<Camera.Size> cameraSizes = camera.getParameters().getSupportedPreviewSizes();
    List<Size> sizes = new ArrayList<>();
    for (Camera.Size size : cameraSizes) {
      Log.d(TAG, String.format("Found camera size: (%d x %d)", size.width, size.height));
      sizes.add(new Size(size.width, size.height));
    }

    cameraSize = getBestSize(sizes, minSize, aspectRatio);
    Log.i(TAG, "Using camera size " + cameraSize);

    Camera.Parameters parameters = camera.getParameters();
    parameters.setPreviewSize(cameraSize.width, cameraSize.height);
    camera.setParameters(parameters);

    // Create our Preview view and set it as the content of our activity.
    cameraPreview = new CameraPreview(context, camera);
    layout.addView(cameraPreview);
  }

  @Override
  @NonNull public YuvRgbFrame getFrame() throws InterruptedException {
    Log.v(TAG, "Getframe called");

    // Submit a callback, wait for it. If we don't get the callback soon enough, try again.
    // Eventually, we'll get a frame, or die trying.
    while (true) {
      camera.setOneShotPreviewCallback((yuvBytes, cam) -> {
        ByteBuffer yuvData = ByteBuffer.wrap(yuvBytes);
        frameQueue.add(new YuvRgbFrame(yuvData, cameraSize, true));
      });

      YuvRgbFrame frame = frameQueue.poll(100, TimeUnit.MILLISECONDS);
      if (frame != null) { // null frame indicates timeout was reached.
        return frame;
      }
    }
  }

  @Override
  public void onDestroy() {

  }
}