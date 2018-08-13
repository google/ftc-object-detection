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

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;

import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

@RunWith(Parameterized.class)
public class ImageUtilsTest {

  @Parameterized.Parameters
  public static Collection<Object []> data() {
    return Arrays.asList(new Object[][] {
        {1, 1}, {1, 3}, {3, 1}, {10, 10}, {10, 11}, {11, 10}, {11, 11}
    });
  }

  @Parameterized.Parameter(0)
  public int sourceWidth;

  @Parameterized.Parameter(1)
  public int sourceHeight;

  private static void printMatrix(int[] A, int width, int height) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        System.out.print("" + A[i * width + j] + " ");
      }
      System.out.println();
    }
  }

  private static int getIndex(int row, int col, int width, int height) {
    return row * width + col;
  }

  private static void fillMatrix(int[] A, int width, int height) {
    // Fill A with something useful
    int counter = 0;
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        A[i * width + j] = counter++;
      }
    }
  }

  @Test
  public void transposeMatrix() {
    int[] A = new int[sourceWidth * sourceHeight];
    int[] B = new int[sourceWidth * sourceHeight];

    fillMatrix(A, sourceWidth, sourceHeight);

    ImageUtils.transposeMatrix(A, B, sourceWidth, sourceHeight);

    printMatrix(A, sourceWidth, sourceHeight);
    printMatrix(B, sourceHeight, sourceWidth);

    for (int row = 0; row < sourceHeight; row++) {
      for (int col = 0; col < sourceWidth; col++) {
        assertEquals(A[getIndex(row, col, sourceWidth, sourceHeight)],
                     B[getIndex(col, row, sourceHeight, sourceWidth)]);
      }
    }
  }

  @Test
  public void flipMatrixLeftRight() {
    int[] A = new int[sourceWidth * sourceHeight];
    int[] B = new int[sourceWidth * sourceHeight];

    fillMatrix(A, sourceWidth, sourceHeight);

    ImageUtils.flipMatrixLeftRight(A, B, sourceWidth, sourceHeight);

    printMatrix(A, sourceWidth, sourceHeight);
    printMatrix(B, sourceWidth, sourceHeight);

    for (int row = 0; row < sourceHeight; row++) {
      for (int col = 0; col < sourceWidth; col++) {
        assertEquals(A[getIndex(row, col, sourceWidth, sourceHeight)],
                     B[getIndex(row, sourceWidth - col - 1, sourceWidth, sourceHeight)]);
      }
    }
  }

  @Test
  public void flipMatrixUpDown() {
    int[] A = new int[sourceWidth * sourceHeight];
    int[] B = new int[sourceWidth * sourceHeight];

    fillMatrix(A, sourceWidth, sourceHeight);

    ImageUtils.flipMatrixUpDown(A, B, sourceWidth, sourceHeight);

    printMatrix(A, sourceWidth, sourceHeight);
    printMatrix(B, sourceWidth, sourceHeight);

    for (int row = 0; row < sourceHeight; row++) {
      for (int col = 0; col < sourceWidth; col++) {
        assertEquals(A[getIndex(row, col, sourceWidth, sourceHeight)],
            B[getIndex(sourceHeight - row - 1, col, sourceWidth, sourceHeight)]);
      }
    }
  }

  @Test
  public void rotateMatrix() {

    int[] A = new int[sourceWidth * sourceHeight];
    fillMatrix(A, sourceWidth, sourceHeight);
    int[] reference = A.clone();

    Size currentSize = new Size(sourceWidth, sourceHeight);

    for (int rotation : new int[]{90, 180, 270}) {
      for (int i = 0; i < 4; i++) {
        currentSize = ImageUtils.rotateMatrix(A, currentSize.width, currentSize.height, rotation);
      }

      // After 4 rotations, the array should be back to how it started.
      assertEquals(sourceWidth, currentSize.width);
      assertEquals(sourceHeight, currentSize.height);
      assertArrayEquals(reference, A);
    }
  }
}
