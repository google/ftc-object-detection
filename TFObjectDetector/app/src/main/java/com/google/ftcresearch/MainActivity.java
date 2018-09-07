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

package com.google.ftcresearch;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.media.Image;
import android.os.Bundle;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.RelativeLayout;

import com.google.ftcresearch.tfod.util.Recognition;
import com.google.ftcresearch.tfod.detection.TFObjectDetector;
import com.google.ftcresearch.tfod.detection.TfodParameters;
import com.google.ftcresearch.tfod.util.Timer;
import com.google.ftcresearch.tfod.generators.FrameGenerator;
import com.google.ftcresearch.tfod.generators.ImageFrameGenerator;
import com.google.ftcresearch.tfod.generators.MovingImageFrameGenerator;
import com.google.ftcresearch.tfod.generators.NativeCameraFrameGenerator;
import com.google.ftcresearch.tfod.util.YuvRgbFrame;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {

  private static final String TAG = "MainActivity";
  private static final String FRAME_GENERATOR_TYPE = "camera";

  private FrameGenerator frameGenerator;
  private TFObjectDetector tfod;

  private final Timer timer = new Timer(TAG);

  /** Handle the user giving us permission to use the camera. */
  @Override
  public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                         @NonNull int[] grantResults) {
    Log.i(TAG, "Request permission result function callback called");
    for (int i = 0; i < permissions.length; i++) {
      Log.i(TAG, String.format("Request %s was granted: %d", permissions[i], grantResults[i]));
      if (permissions[i].equals(Manifest.permission.CAMERA)) {
        if (grantResults[i] == PackageManager.PERMISSION_GRANTED) {

          // This is the same initialization as below, but must be repeated here.
          startFrameGenerator();
          initializeTfod();
        } else {
          Log.w(TAG, "Quitting because camera permission not granted.");
          finishAndRemoveTask();
        }
      }
    }
  }

  /** Perform necessary initialization to start whichever frame generator is specified. */
  private void startFrameGenerator() {
    switch (FRAME_GENERATOR_TYPE) {
      case "static": // Static image
        {
          frameGenerator = ImageFrameGenerator.makeFromResourceId(this, R.raw.img_01290);
          break;
        }
      case "moving": // Move an image around
        {
          frameGenerator = MovingImageFrameGenerator.makeFromResourceId(this, R.raw.img_01290);
          break;
        }
      case "camera": // Try to use camera 1 api (via NativeCameraFrameGenerator)
        {
          frameGenerator = new NativeCameraFrameGenerator(this, R.id.bottom_frame, 300,
              1920.0f / 1080.0f);
          break;
        }
      default:
        throw new IllegalArgumentException("Need to choose a different frameGeneratorType");
    }
  }

  /**
   * Ask the user for permission to use the camera.
   * @return true if permission is already granted, false if a request was made to use it.
   */
  private boolean requestCameraPermission() {
    // Make sure camera permission is granted.
    if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) !=
        PackageManager.PERMISSION_GRANTED) {
      ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, 0);
      return false;
    } else {
      Log.i(TAG, "Camera permission already granted!");
      return true;
    }
  }

  @Override
  protected void onCreate(Bundle savedInstanceState) {
    super.onCreate(savedInstanceState);

    setContentView(R.layout.activity_linear);

    final boolean permissionGranted;
    if (FRAME_GENERATOR_TYPE.equals("camera")) {
      permissionGranted = requestCameraPermission();
    } else {
      permissionGranted = true;
    }

    if (permissionGranted) {
      startFrameGenerator();
      initializeTfod();
    }
  }

  private void initializeTfod() {
    // Create a new TFObjectDetector, and try to initialize it. This should be akin to what happens
    // in the "init" stage of the FTC competition.
    tfod =
        new TFObjectDetector(
            new TfodParameters.Builder()
                .numExecutorThreads(4)
                .numInterpreterThreads(1)
//                .trackerDisable(true)
                .drawRecognitionsEnable(R.id.top_frame)
                .build(),
            frameGenerator);

    tfod.initialize(this);
  }

  @Override
  protected void onDestroy() {

    super.onDestroy();

    if (tfod != null) {
      Log.i(TAG, "Shutting down tfod");
      tfod.shutdown(this);
    }

    // tfod doesn't shut down the frame generator, so we do that ourselves.
    if (frameGenerator != null) {
      frameGenerator.shutdown(this);
    }
  }
}
