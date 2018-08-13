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

import android.util.Log;

import java.util.ArrayDeque;
import java.util.concurrent.TimeUnit;

/**
 * Class to support nested timing of blocks of code.
 *
 * This object measures the difference in system time (using System.nanoTime()) between the start
 * and end times. Multiple calls to start() and end() are supported, enabling nested timing. The
 * timing information is printed using verbose logging (Log.v).
 *
 * Usage example:
 *
 * <pre>
 *   Timer timer = new Timer(TAG);
 *
 *   timer.start("section name");
 *   [... your block of code ...]
 *
 *   timer.start("new section name");
 *   [... another block of code ...]
 *
 *   timer.end();
 *   timer.end();
 * </pre>
 *
 * Which will have the following sample output:
 *
 * <pre>
 *   Timer: section "new section name" took X1 ms (Y1 ns)
 *   Timer: section "section name" took X2 ms (Y2 ns)
 * </pre>
 */
public class Timer {

  private final ArrayDeque<NamedTimestampNanos> times = new ArrayDeque<>();
  private final String timerName;

  public Timer(String timerName) {
    this.timerName = timerName;
  }

  /** Start timing a new section. */
  public void start(String sectionName) {
    times.addLast(new NamedTimestampNanos(sectionName, System.nanoTime()));
  }

  /** Stop timing the most recent section, and print verbose log the execution time. */
  public void end() {
    final NamedTimestampNanos nts = times.removeLast();
    final long endTime = System.nanoTime();
    final long diffTime = endTime - nts.timestampNanos;

    Log.v(timerName, String.format("Timer: section \"%s\" took %d ms (%d ns)", nts.name,
        TimeUnit.MILLISECONDS.convert(diffTime, TimeUnit.NANOSECONDS), diffTime));
  }

  /** Class to store a timestamp along with a section name. */
  private class NamedTimestampNanos {
    final String name;
    final long timestampNanos;

    NamedTimestampNanos(String name, long timestampNanos) {
      this.name = name;
      this.timestampNanos = timestampNanos;
    }
  }
}
