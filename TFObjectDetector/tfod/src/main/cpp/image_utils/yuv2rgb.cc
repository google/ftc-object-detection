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

// This is a collection of routines which converts various YUV image formats
// to ARGB.

#include "yuv2rgb.h"

#ifndef MAX
#define MAX(a, b) ({__typeof__(a) _a = (a); __typeof__(b) _b = (b); _a > _b ? _a : _b; })
#define MIN(a, b) ({__typeof__(a) _a = (a); __typeof__(b) _b = (b); _a < _b ? _a : _b; })
#endif

// This value is 2 ^ 18 - 1, and is used to clamp the RGB values before their ranges
// are normalized to eight bits.
static const int kMaxChannelValue = 262143;

static inline uint32_t YUV2RGB(int nY, int nU, int nV) {
  nY -= 16;
  nU -= 128;
  nV -= 128;
  if (nY < 0) nY = 0;

  // This is the floating point equivalent. We do the conversion in integer
  // because some Android devices do not have floating point in hardware.
  // nR = (int)(1.164 * nY + 2.018 * nU);
  // nG = (int)(1.164 * nY - 0.813 * nV - 0.391 * nU);
  // nB = (int)(1.164 * nY + 1.596 * nV);

  int nR = 1192 * nY + 1634 * nV;
  int nG = 1192 * nY - 833 * nV - 400 * nU;
  int nB = 1192 * nY + 2066 * nU;

  nR = MIN(kMaxChannelValue, MAX(0, nR));
  nG = MIN(kMaxChannelValue, MAX(0, nG));
  nB = MIN(kMaxChannelValue, MAX(0, nB));

  nR = (nR >> 10) & 0xff;
  nG = (nG >> 10) & 0xff;
  nB = (nB >> 10) & 0xff;

  return 0xff000000 | (nR << 16) | (nG << 8) | nB;
}

//  Accepts a YUV 4:2:0 image with a plane of 8 bit Y samples followed by an
//  interleaved U/V plane containing 8 bit 2x2 subsampled chroma samples,
//  except the interleave order of U and V is reversed. Converts to a packed
//  ARGB 32 bit output of the same pixel dimensions.
void ConvertYUV420SPToARGB8888(const uint8_t* const yData,
                               const uint8_t* const uvData,
                               uint32_t* const output, const int width,
                               const int height, const bool uv_flipped) {
  const uint8_t* pY = yData;
  const uint8_t* pUV = uvData;
  uint32_t* out = output;

  for (int y = 0; y < height; y++) {
    for (int x = 0; x < width; x++) {
      int nY = *pY++;
      int offset = (y >> 1) * width + 2 * (x >> 1);
#ifdef __APPLE__
      int nU = pUV[offset];
      int nV = pUV[offset + 1];
#else
      int nV = pUV[offset];
      int nU = pUV[offset + 1];
#endif

      if (uv_flipped) { // U and V channels are flipped from where they're supposed to be.
        // Flip arguments rather than explicitly swapping.
        *out++ = YUV2RGB(nY, nV, nU);
      } else {
        *out++ = YUV2RGB(nY, nU, nV); // The standardized way.
      }
    }
  }
}
