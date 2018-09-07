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

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.media.Image;
import android.support.annotation.NonNull;
import android.util.Log;


import com.google.ftcresearch.tfod.R;
import com.google.ftcresearch.tfod.util.Rate;
import com.google.ftcresearch.tfod.util.YuvRgbFrame;
import com.google.ftcresearch.tfod.util.Size;

import java.nio.IntBuffer;
import java.util.concurrent.TimeUnit;

/**
 * Implements {@link FrameGenerator} by returning the input image at specified frame rate.
 *
 * <p>The delay is simulated by sleeping for an appropriate amount inside the call to getFrame().
 */
public class ImageFrameGenerator implements FrameGenerator {

  private static final String TAG = "ImageFrameGenerator";

  private static final double FRAME_RATE = 0.1; // Hz
  private final Rate rate = new Rate(FRAME_RATE);

  private final YuvRgbFrame frame;

  /** @param bm Bitmap to return periodically. */
  public ImageFrameGenerator(Bitmap bm) {

    final IntBuffer rgbFrame = IntBuffer.allocate(bm.getWidth() * bm.getHeight());
    bm.copyPixelsToBuffer(rgbFrame);
    frame = new YuvRgbFrame(rgbFrame, new Size(bm.getWidth(), bm.getHeight()));

    Log.d(TAG, String.format("Created new ImageFrameGenerator with a (%d x %d) frame",
        frame.getWidth(), frame.getHeight()));
  }

  /**
   * Get the stored frame, after a delay to mimic the desired frame rate.
   *
   * <p>Note that, for efficiency, this function returns the same bitmap as originally given in the
   * constructor. Any changes to the returned bitmap will thus persist in all future frames returned
   * by this function. The user is expected to not modify the resultant frame.
   *
   * @return {@inheritDoc}
   */
  @Override
  @NonNull public YuvRgbFrame getFrame() throws InterruptedException {

    rate.checkedSleep();
    return frame;
  }

  @Override
  public void shutdown(Activity activity) {
  }

  /** Convenience method to handle loading bitmap from a resource id. */
  public static ImageFrameGenerator makeFromResourceId(Context context, int resourceId) {
    final Bitmap bm = BitmapFactory.decodeResource(context.getResources(), resourceId);
    final Bitmap bmScaled = Bitmap.createScaledBitmap(bm, 1920, 1080, true);
    return new ImageFrameGenerator(bmScaled);
  }
}