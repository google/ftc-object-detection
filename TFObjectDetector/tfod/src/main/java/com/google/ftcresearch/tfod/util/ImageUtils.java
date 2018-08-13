/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.google.ftcresearch.tfod.util;

import android.graphics.Matrix;
import android.util.Log;

import java.nio.ByteBuffer;
import java.nio.IntBuffer;

/** Utility class for manipulating images. */
public class ImageUtils {
  @SuppressWarnings("unused")
  private static final String TAG = "ImageUtils";

  static {
    // We want this to throw an exception if conversion isn't happening natively.
    System.loadLibrary("image_utils");
  }

  /**
   * Return a matrix which transforms coordinates from the source image to the destination image
   * @param source Size of the original image.
   * @param destination Size of the new image.
   * @return The transformation between the two frames.
   */
  public static Matrix transformBetweenImageSizes(Size source, Size destination) {
    final float scaleFactorX = (float) destination.width / source.width;
    final float scaleFactorY = (float) destination.height / source.height;

    Matrix m = new Matrix();
    m.setScale(scaleFactorX, scaleFactorY);
    return m;
  }

  /**
   * Returns a transformation matrix from one reference frame into another. Handles cropping (if
   * maintaining aspect ratio is desired) and rotation.
   *
   * @param srcWidth Width of source frame.
   * @param srcHeight Height of source frame.
   * @param dstWidth Width of destination frame.
   * @param dstHeight Height of destination frame.
   * @param applyRotation Amount of rotation to apply from one frame to another. Must be a multiple
   *     of 90.
   * @param maintainAspectRatio If true, will ensure that scaling in x and y remains constant,
   *     cropping the image if necessary.
   * @return The transformation fulfilling the desired requirements.
   */
  public static Matrix getTransformationMatrix(
      final int srcWidth,
      final int srcHeight,
      final int dstWidth,
      final int dstHeight,
      final int applyRotation,
      final boolean maintainAspectRatio) {
    final Matrix matrix = new Matrix();

    if (applyRotation != 0) {
      if (applyRotation % 90 != 0) {
        Log.w(TAG, String.format("Rotation of %d %% 90 != 0", applyRotation));
      }

      // Translate so center of image is at origin.
      matrix.postTranslate(-srcWidth / 2.0f, -srcHeight / 2.0f);

      // Rotate around origin.
      matrix.postRotate(applyRotation);
    }

    // Account for the already applied rotation, if any, and then determine how
    // much scaling is needed for each axis.
    final boolean transpose = (Math.abs(applyRotation) + 90) % 180 == 0;

    final int inWidth = transpose ? srcHeight : srcWidth;
    final int inHeight = transpose ? srcWidth : srcHeight;

    // Apply scaling if necessary.
    if (inWidth != dstWidth || inHeight != dstHeight) {
      final float scaleFactorX = dstWidth / (float) inWidth;
      final float scaleFactorY = dstHeight / (float) inHeight;

      if (maintainAspectRatio) {
        // Scale by minimum factor so that dst is filled completely while
        // maintaining the aspect ratio. Some image may fall off the edge.
        final float scaleFactor = Math.max(scaleFactorX, scaleFactorY);
        matrix.postScale(scaleFactor, scaleFactor);
      } else {
        // Scale exactly to fill dst from src.
        matrix.postScale(scaleFactorX, scaleFactorY);
      }
    }

    if (applyRotation != 0) {
      // Translate back from origin centered reference to destination frame.
      matrix.postTranslate(dstWidth / 2.0f, dstHeight / 2.0f);
    }

    return matrix;
  }

  /**
   * Transpose a matrix A into B.
   *
   * Assumes row major storage for A and B. A and B must be different arrays.
   *
   * @param a Source matrix.
   * @param b Destination matrix.
   * @param width Width of matrix A.
   * @param height Width of matrix B.
   */
  public static void transposeMatrix(int[] a, int[] b, int width, int height) {
    if (a == b) {
      throw new IllegalArgumentException("Array A cannot be the same as Array B!");
    }

    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        b[j * height + i] = a[i * width + j];
      }
    }
  }

  /**
   * Flip matrix A horizontally into matrix B.
   *
   * Assumes row major storage for A and B. A and B must be different arrays.
   *
   * @param a Source matrix.
   * @param b Destination matrix.
   * @param width Width of matrix A.
   * @param height Width of matrix B.
   */
  public static void flipMatrixLeftRight(int[] a, int[] b, int width, int height) {
    if (a == b) {
      throw new IllegalArgumentException("Array A cannot be the same as Array B!");
    }

    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        b[i * width + (width - j - 1)] = a[i * width + j];
      }
    }
  }

  /**
   * Flip matrix A vertically into matrix B.
   *
   * Assumes row major storage for A and B. A and B must be different arrays.
   *
   * @param a Source matrix.
   * @param b Destination matrix.
   * @param width Width of matrix A.
   * @param height Width of matrix B.
   */
  public static void flipMatrixUpDown(int[] a, int[] b, int width, int height) {
    if (a == b) {
      throw new IllegalArgumentException("Array A cannot be the same as Array B!");
    }

    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        b[(height - i - 1) * width + j] = a[i * width + j];
      }
    }
  }

  //TODO(vasuagrawal): See how long we're spending in these methods and optimize if necessary
  public static Size rotateMatrix(int[] a, int width, int height, int rotation) {
    if (rotation != 90 && rotation != 180 && rotation != 270 && rotation != 0) {
      throw new IllegalArgumentException("Rotation is not a simple rotation (0, 90, 180, 2700! " +
          "Won't rotate!");
    }

    int[] b = new int[width * height];
    switch (rotation) {
      case 90: {
        ImageUtils.transposeMatrix(a, b, width, height);
        ImageUtils.flipMatrixLeftRight(b, a, height, width);
        break;
      }

      case 180: {
        ImageUtils.flipMatrixLeftRight(a, b, width, height);
        ImageUtils.flipMatrixUpDown(b, a, width, height);
        break;
      }

      case 270: {
        ImageUtils.transposeMatrix(a, b, width, height);
        ImageUtils.flipMatrixUpDown(b, a, height, width);
      }
    }

    return Size.getRotatedSize(new Size(width, height), rotation);
  }

  /**
   * Converts YUV420 semi-planar data to ARGB 8888 data using the supplied width and height. The
   * input and output must already be allocated and non-null. For efficiency, no error checking is
   * performed.
   *
   * @param input The array of YUV 4:2:0 input data.
   * @param output A pre-allocated array for the ARGB 8:8:8:8 output data.
   * @param width The width of the input image.
   * @param height The height of the input image.
   * @param uvFlipped Whether the U and V channels are flipped.
   */
  public static native void convertYUV420SPToARGB8888(
      byte[] input, int[] output, int width, int height, boolean uvFlipped);

  /**
   * Converts 32-bit ARGB8888 image data to YUV420SP data. This is useful, for instance, in creating
   * data to feed the classes that rely on raw camera preview frames.
   *
   * @param input An array of input pixels in ARGB8888 format.
   * @param output A pre-allocated array for the YUV420SP output data.
   * @param width The width of the input image.
   * @param height The height of the input image.
   */
  public static native void convertARGB8888ToYUV420SP(
      int[] input, byte[] output, int width, int height);


  private static native void yuv420spToArgb8888(
      ByteBuffer inputBuffer, byte[] inputArray, boolean isInputDirect,
      IntBuffer outputBuffer, int[] outputArray, boolean isOutputDirect,
      int width, int height, boolean uvFlipped);

  public static void convertBuffersYUV420SPToARGB8888(
      ByteBuffer input, IntBuffer output, int width, int height, boolean uvFlipped) {

    boolean isInputDirect = input.isDirect();
    byte[] inputArray = input.hasArray() ? input.array() : null;

    if (!isInputDirect && inputArray == null) {
      throw new RuntimeException("Input buffer is not direct and doesn't have array!");
    }

    boolean isOutputDirect = output.isDirect();
    int[] outputArray = output.hasArray() ? output.array() : null;

    if (!isOutputDirect && outputArray == null) {
      throw new RuntimeException("Output buffer is not direct and doesn't have array!");
    }

    yuv420spToArgb8888(input, inputArray, isInputDirect, output, outputArray, isOutputDirect,
        width, height, uvFlipped);
  }


  private static native void argb8888ToYuv420sp(
      IntBuffer inputBuffer, int[] inputArray, boolean isInputDirect,
      ByteBuffer outputBuffer, byte[] outputArray, boolean isOutputDirect,
      int width, int height);

  public static void convertBuffersARGB8888ToYuv420SP(
      IntBuffer input, ByteBuffer output, int width, int height) {

    boolean isInputDirect = input.isDirect();
    int[] inputArray = input.hasArray() ? input.array() : null;

    if (!isInputDirect && inputArray == null) {
      throw new RuntimeException("Input buffer is not direct and doesn't have array!");
    }

    boolean isOutputDirect = output.isDirect();
    byte[] outputArray = output.hasArray() ? output.array() : null;

    if (!isOutputDirect && outputArray == null) {
      throw new RuntimeException("Output buffer is not direct and doesn't have array!");
    }

    argb8888ToYuv420sp(input, inputArray, isInputDirect, output, outputArray, isOutputDirect,
        width, height);
  }
}