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

package com.google.ftcresearch.tfod.util;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.support.annotation.NonNull;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/** Threadsafe class to handle a frame in either ARGB_8888 or YUV420SP, and convert between the two.
 *
 * All data here is intended to be immutable by the user. However, for efficiency, the user is
 * given back the internal buffer, and is expected to not modify it. Furthermore, conversion
 * between the two data formats is done lazily, but is cached. That is, if a YuvRgbFrame is
 * created with ARGB_8888 data, and then YUV420SP data is requested, the first call will perform
 * the conversion, while subsequent calls will return near-instantly as they will return a
 * ByteBuffer which points to the same data.
 */
public class YuvRgbFrame {

  private static final String TAG = "YuvRgbFrame";

  private final boolean uvFlipped; // Whether the U and V channels are flipped (NV21 vs NV12)
  private ByteBuffer yuvFrame; // YUV420SP format, aka Y, U, V appended in a single array
  private IntBuffer rgbFrame; // ARGB_8888 format
  private final Size size;
  private final Map<Size, YuvRgbFrame> resizeCache = new ConcurrentHashMap<>();
  private final Object resizeBmLock = new Object();
  private Bitmap resizeBm;

  /** Construct a YuvRgbFrame from RGB data.
   *
   * @param rgbFrame IntBuffer, backed by array, containing row-major ARGB_8888 formatted data.
   * @param size Size of the actual image corresponding to the pixelwise data in rgbFrame.
   */
  public YuvRgbFrame(@NonNull IntBuffer rgbFrame, Size size) {
    this.uvFlipped = false; // Will never be flipped if starting as an RGB frame
    this.rgbFrame = rgbFrame;
    this.size = size;
    this.yuvFrame = null;

    // Save the current frame in the cache, in case a trivial resize is requested.
    resizeCache.put(size, this);
  }

  /** Construct a YuvRgbFrame from YUV data.
   *
   * @param yuvFrame ByteBuffer, backed by array, containing row-major YUV420SP formatted data.
   * @param size Size of the actual image corresponding to the pixelwise data in yuvFrame.
   * @param uvFlipped Whether the u and v channels are flipped in the YUV420SP format. If using
   *                  one option gives you an image whose colors seem inverted, use the other
   *                  option.
   */
  public YuvRgbFrame(@NonNull ByteBuffer yuvFrame, Size size, boolean uvFlipped) {
    this.uvFlipped = uvFlipped; // May be flipped.
    this.yuvFrame = yuvFrame;
    this.size = size;
    this.rgbFrame = null;

    // Save the current frame in the cache, in case a trivial resize is requested.
    resizeCache.put(size, this);
  }

  /** Convert RGB image data to YUV420SP image data */
  private static ByteBuffer convertRgbToYuv(IntBuffer rgbFrame, Size size) {

    ByteBuffer yuvFrame = ByteBuffer.allocate(3 * size.width * size.height);
    ImageUtils.convertBuffersARGB8888ToYuv420SP(rgbFrame, yuvFrame, size.width, size.height);
    return yuvFrame;
  }

  /** Lazily get YUV420SP formatted image data.
   *
   * If the image was originally given in YUV420SP format, this function will simply return a new
   * ByteBuffer wrapper around the original data. Note that this will preserve a u-v flip in the
   * original data, if any.
   *
   * If the image was not originally given in the YUV420SP format, this function will first
   * convert the supplied ARGB_8888 data to YUV420SP using native conversion, and then return a
   * wrapper around that. The data will also be cached, so that future lookups return nearly
   * instantly. The result of this conversion will not have a u-v flip.
   *
   * The dimension of the original input data is preserved, and the original data is unmodified.
   *
   * @return ByteBuffer containing YUV420SP formatted data without a u-v flip.
   */
  public synchronized ByteBuffer getYuvData() {
    if (yuvFrame == null) {
      Timer timer = new Timer(TAG);
      timer.start("Converting RGB to YUV");
      // Need to convert it from RGB frame first.
      yuvFrame = convertRgbToYuv(rgbFrame, size);
      timer.end();
    } else {
      Log.v(TAG, "Able to skip RGB to YUV conversion!");
    }

    // Return a duplicated ByteBuffer pointing to the same thing, but with a position at 0.
    ByteBuffer newYuvFrame = yuvFrame.duplicate();
    newYuvFrame.position(0);
    return newYuvFrame;
  }

  /** Convert YUV420SP image data to RGB. */
  private static IntBuffer convertYuvToRgb(ByteBuffer yuvFrame, Size size, boolean uvFlipped) {

    IntBuffer rgbFrame = IntBuffer.allocate(size.width * size.height);
    ImageUtils.convertBuffersYUV420SPToARGB8888(yuvFrame, rgbFrame, size.width, size.height, uvFlipped);
    return rgbFrame;
  }

  /** Lazily get ARGB_8888 formatted image data.
   *
   * IF the image was originally given in ARGB_8888 format, this function will simply return a
   * new IntBuffer wrapper around the original data.
   *
   * If the image was not originally given in the ARGB_8888 format, this function will first
   * convert the supplied YUV420SP data to ARGB_8888 using native conversion, and then return a
   * wrapper around that. The converted data will also be cached, so that future lookups return
   * nearly instantly.
   *
   * The dimension of the original input data is preserved, and the original data is unmodified.
   *
   * @return IntBuffer containing ARGB_8888 formatted image data.
   */
  public synchronized IntBuffer getRgbData() {
    if (rgbFrame == null) {
      Timer timer = new Timer(TAG);
      timer.start("Converting YUV to RGB");
      // Need to convert it from YUV first.
      rgbFrame = convertYuvToRgb(yuvFrame, size, uvFlipped);
      timer.end();
    } else {
      Log.v(TAG, "Able to skip YUV to RGB conversion!");
    }

    // NOTE: Can't return .asReadOnlyBuffer() because that prevents use of .array().
    // Return a duplicated IntBuffer pointing to the same thing, but with a position of 0.
    IntBuffer newRgbFrame = rgbFrame.duplicate();
    newRgbFrame.position(0);
    return newRgbFrame;
  }

  /** Get a bitmap with the image data.
   *
   * This function calls getRgbData internally to get the data, and thus is subject to the same
   * lazy data loading policies. The returned Bitmap will have a copy of the RGB data, and thus
   * is safe to modify (e.g. draw on) as desired.
   *
   * @return A bitmap containing the image data for this YuvRgbFrame.
   */
  public synchronized Bitmap getCopiedBitmap() {

    // The RGB data is copied into the new Bitmap, so it can be drawn on.
    Bitmap bm = Bitmap.createBitmap(getWidth(), getHeight(), Bitmap.Config.ARGB_8888);
    bm.copyPixelsFromBuffer(getRgbData());
    return bm;
  }

  public byte[] getLuminosity() {

    ByteBuffer yuvData = getYuvData();
    if (yuvData.hasArray()) {
      return yuvData.array();
    } else {
      byte[] luminosity = new byte[getWidth() * getHeight()];
      getYuvData().get(luminosity, 0, getWidth() * getHeight());
      return luminosity;
    }
  }

  public Size getSize() {
    return size;
  }

  public int getWidth() {
    return size.width;
  }

  public int getHeight() {
    return size.height;
  }

  /**
   * Return a YuvRgbFrame which has been resized.
   *
   * The results of this operation are cached. If the resize method is called twice on the same
   * YuvRgbFrame, the second call will return a reference to the same frame as the first call.
   * That is, any modifications to the YuvRgbFrames returned from this call will persist in
   * future frames.
   * */
  public YuvRgbFrame resize(Size newSize) {
    YuvRgbFrame cached = resizeCache.get(newSize);
    if (cached != null) {
      Log.d(TAG, "Returning cached frame of size: " + newSize);
      return cached;
    }

    Bitmap scaledBm;

    // It seems like Bitmap isn't completely thread-safe, so guard the access with a lock
    synchronized (resizeBmLock) {
      if (resizeBm == null) {
        IntBuffer rgbData = getRgbData();
        if (rgbData.hasArray()) {
          resizeBm = Bitmap.createBitmap(rgbData.array(), getWidth(), getHeight(),
              Bitmap.Config.ARGB_8888);
        } else {
          int[] rgbArray = new int[getWidth() * getHeight()];
          rgbData.get(rgbArray);

          resizeBm = Bitmap.createBitmap(rgbArray, getWidth(), getHeight(),
              Bitmap.Config.ARGB_8888);
        }
      }

      scaledBm = Bitmap.createScaledBitmap(resizeBm, newSize.width, newSize.height, false);
    }

    // Get the pixels from the new Bitmap
    IntBuffer newRgbFrame = IntBuffer.allocate(newSize.width * newSize.height);
    scaledBm.copyPixelsToBuffer(newRgbFrame);

    // Cache the new image in case we need it again
    YuvRgbFrame newYuvRgbFrame = new YuvRgbFrame(newRgbFrame, newSize);
    resizeCache.put(newSize, newYuvRgbFrame);

    return newYuvRgbFrame;
  }

  /**
   * Return a new YuvRgbFrame which has been rotated clockwise.
   *
   * Note that any sort of rotation will take a non trivial amount of time to perform (10ms),
   * especially on larger images, and thus it is encouraged that you simply orient your camera in
   * the proper orientation so as to not require any rotations to have the image data in the
   * right layout.
   *
   * @param degrees Amount to rotate frame clockwise, in degrees.
   * @return New YuvRgbFrame with rotated image data.
   */
  public YuvRgbFrame rotate(int degrees /* clockwise */) {

    final Timer timer = new Timer(TAG);
    timer.start("Rotation by " + degrees + " degrees");

    // TODO(vasugrawal): Switch between this and the bitmap implementation, if this is faster.
    // Implementation to do the rotation as a raw image operation. Currently this relies on a
    // very naive implementation of transpose and flips (all in java), and may become
    // significantly faster if written in C++ or with a matrix library.
//    int[] rgbArray = getRgbData().array().clone(); // Need a safe copy to be able to modify
//    Size newSize = ImageUtils.rotateMatrix(rgbArray, getWidth(), getHeight(), degrees);
//    IntBuffer rgbData = IntBuffer.wrap(rgbArray);
//    YuvRgbFrame yuvRgbFrame = new YuvRgbFrame(rgbData, newSize);

    Bitmap bm = Bitmap.createBitmap(getWidth(), getHeight(), Bitmap.Config.ARGB_8888);
    bm.copyPixelsFromBuffer(getRgbData());
    Matrix matrix = new Matrix();
    matrix.setRotate(degrees);
    Bitmap rotatedBm = Bitmap.createBitmap(bm, 0, 0, getWidth(), getHeight(), matrix, true);
    IntBuffer rgbData = IntBuffer.allocate(rotatedBm.getWidth() * rotatedBm.getHeight());
    rotatedBm.copyPixelsToBuffer(rgbData);
    YuvRgbFrame yuvRgbFrame = new YuvRgbFrame(rgbData, new Size(rotatedBm.getWidth(), rotatedBm
        .getHeight()));

    timer.end();
    return yuvRgbFrame;
  }

  public static YuvRgbFrame makeEmptyFrame(Size size) {
    IntBuffer rgbFrame = IntBuffer.allocate(size.width * size.height);
    return new YuvRgbFrame(rgbFrame, size);
  }
}