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

package com.google.ftcresearch.tfod;

import com.google.ftcresearch.tfod.util.RollingAverage;

import org.junit.Test;

import java.util.LinkedList;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class RollingAverageTest {

  private static final double ALLOWABLE_DELTA = 1e-5;

  @Test
  public void returnsDefault() {
    RollingAverage average;

    double[] testValues = {-42, -10.5, -1, -0.1, 0, 0.1, 1, 10.5, 42};

    for (double value : testValues) {
      average = new RollingAverage(10, value);
      assertEquals(value, average.get(), 0); // No delta allowed, value should be identical.
    }
  }

  @Test
  public void defaultConstructorReturnsZero() {
    RollingAverage average = new RollingAverage(10);
    assertEquals(0, average.get(), 0);
  }

  @Test
  public void bufferSizeOneKeepsNewestValue() {
    RollingAverage average = new RollingAverage(1);

    for (double x = -Math.PI; x <= Math.PI; x += Math.PI / 32) {
      double value = Math.sin(x);
      average.add(value);
      assertEquals(value,average.get(),  ALLOWABLE_DELTA);
    }
  }

  @Test
  public void referenceImplementationComparison() {
    int BUFFER_SIZE = 20;
    List<Double> values = new LinkedList<>();
    RollingAverage average = new RollingAverage(BUFFER_SIZE);

    for (double x = -Math.PI; x <= Math.PI; x += Math.PI / 32) {
      double value = Math.sin(x);
      values.add(value);
      average.add(value);

      if (values.size() > BUFFER_SIZE) {
        values.remove(0);
      }

      double rollingAverage = values.stream().mapToDouble(d -> d).sum() / values.size();
      assertEquals(rollingAverage, average.get(), ALLOWABLE_DELTA);
    }
  }

  @Test
  public void addTwice() {
    RollingAverage average = new RollingAverage(2);

    average.add(1);
    assertEquals(1, average.get(), 0);

    average.add(2);
    assertEquals(1.5, average.get(), ALLOWABLE_DELTA);

    average.add(3);
    assertEquals(2.5,average.get(),  ALLOWABLE_DELTA);
  }
}
