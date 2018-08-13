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

import android.graphics.RectF;
import android.support.annotation.NonNull;
import java.util.Locale;
import java.util.Objects;

/**
 * An immutable result describing a single instance of an object recognized by the model.
 *
 * <p>A given frame may have multiple recognitions, with each recognition corresponding to a
 * different instance of an object.
 */
public class Recognition {

  private final String label;
  private final float confidence;
  private final RectF location;

  public Recognition(
      @NonNull final String label, final float confidence, @NonNull final RectF location) {
    this.label = label;
    this.confidence = confidence;
    this.location = location;
  }

  @Override
  public boolean equals(Object o) {
    if (!(o instanceof Recognition)) {
      return false;
    }

    Recognition other = (Recognition) o;
    return confidence == other.confidence
        && label.equals(other.label)
        && location.equals(other.location);
  }

  @Override
  public int hashCode() {
    return Objects.hash(confidence, label, location);
  }

  @Override
  public String toString() {
    return String.format(
        Locale.getDefault(),
        "Recognition(label=%s, confidence=%.3f, location=%s",
        label,
        confidence,
        location);
  }

  public String getLabel() {
    return label;
  }

  public float getConfidence() {
    return confidence;
  }

  /**
   * Get a copy of the location (in image coordinates) corresponding to this recognition.
   *
   * <p>A copy is necessary to ensure that the Recognition is immutable, as this is the only
   * object which could be modified.
   */
  public RectF getLocation() {
    return new RectF(location);
  }
}
