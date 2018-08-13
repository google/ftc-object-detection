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

// This file binds the native image utility code to the Java class
// which exposes them.

#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <android/log.h>
#include <sstream>

#include "rgb2yuv.h"
#include "yuv2rgb.h"

#define IMAGEUTILS_METHOD(METHOD_NAME) \
    Java_com_google_ftcresearch_tfod_util_ImageUtils_##METHOD_NAME // NOLINT

#ifdef __cplusplus
extern "C" {
#endif

// Converting YUV to ARGB with Arrays
JNIEXPORT void JNICALL
IMAGEUTILS_METHOD(convertYUV420SPToARGB8888)(
    JNIEnv* env, jclass clazz, jbyteArray input, jintArray output,
    jint width, jint height, jboolean uvFlipped);

// Converting ARGB to YUV with Arrays
JNIEXPORT void JNICALL
IMAGEUTILS_METHOD(convertARGB8888ToYUV420SP)(
    JNIEnv* env, jclass clazz, jintArray input, jbyteArray output,
    jint width, jint height);

// Conversion functions for ByteBuffers
JNIEXPORT void JNICALL
IMAGEUTILS_METHOD(yuv420spToArgb8888)(
    JNIEnv* env, jclass clazz, jobject inputBuffer, jbyteArray inputArray, jboolean isInputDirect,
    jobject outputBuffer, jintArray outputArray, jboolean isOutputDirect,
    jint width, jint height, jboolean uvFlipped);

JNIEXPORT void JNICALL
IMAGEUTILS_METHOD(argb8888ToYuv420sp)(
    JNIEnv* env, jclass clazz, jobject inputBuffer, jintArray inputArray, jboolean isInputDirect,
    jobject outputBuffer, jbyteArray outputArray, jboolean isOutputDirect,
    jint width, jint height);

#ifdef __cplusplus
}
#endif

JNIEXPORT void JNICALL
IMAGEUTILS_METHOD(convertYUV420SPToARGB8888)(
    JNIEnv* env, jclass clazz, jbyteArray input, jintArray output,
    jint width, jint height, jboolean uvFlipped) {
  jboolean inputCopy = JNI_FALSE;
  jbyte* const i = env->GetByteArrayElements(input, &inputCopy);

  jboolean outputCopy = JNI_FALSE;
  jint* const o = env->GetIntArrayElements(output, &outputCopy);

  ConvertYUV420SPToARGB8888(reinterpret_cast<uint8_t*>(i),
                            reinterpret_cast<uint8_t*>(i) + width * height,
                            reinterpret_cast<uint32_t*>(o), width, height, uvFlipped);

  env->ReleaseByteArrayElements(input, i, JNI_ABORT);
  env->ReleaseIntArrayElements(output, o, 0);
}

JNIEXPORT void JNICALL
IMAGEUTILS_METHOD(convertARGB8888ToYUV420SP)(
    JNIEnv* env, jclass clazz, jintArray input, jbyteArray output,
    jint width, jint height) {
  jboolean inputCopy = JNI_FALSE;
  jint* const i = env->GetIntArrayElements(input, &inputCopy);

  jboolean outputCopy = JNI_FALSE;
  jbyte* const o = env->GetByteArrayElements(output, &outputCopy);

  ConvertARGB8888ToYUV420SP(reinterpret_cast<uint32_t*>(i),
                            reinterpret_cast<uint8_t*>(o), width, height);

  env->ReleaseIntArrayElements(input, i, JNI_ABORT);
  env->ReleaseByteArrayElements(output, o, 0);
}

JNIEXPORT void JNICALL
IMAGEUTILS_METHOD(yuv420spToArgb8888)(
    JNIEnv* env, jclass clazz, jobject inputBuffer, jbyteArray inputArray, jboolean isInputDirect,
    jobject outputBuffer, jintArray outputArray, jboolean isOutputDirect,
    jint width, jint height, jboolean uvFlipped) {

  // Assign the input pointer depending on whether the ByteBuffer is direct or not.
  uint8_t* input;
  if ((bool) isInputDirect) {
    void* i = env->GetDirectBufferAddress(inputBuffer); // Trusting that this won't be null.
    input = reinterpret_cast<uint8_t*>(i);
  } else {
    jboolean inputCopy = JNI_FALSE;
    jbyte* i = env->GetByteArrayElements(inputArray, &inputCopy);
    input = reinterpret_cast<uint8_t*>(i);
  }

  // Assign the output pointer depending on whether the IntBuffer is direct or not.
  uint32_t* output;
  if ((bool) isOutputDirect) {
    void* o = env->GetDirectBufferAddress(outputBuffer); // Trusting that this won't be null.
    output = reinterpret_cast<uint32_t*>(o);
  } else {
    jboolean outputCopy = JNI_FALSE;
    jint* o = env->GetIntArrayElements(outputArray, &outputCopy);
    output = reinterpret_cast<uint32_t*>(o);
  }

  // Actually perform the conversion with one line ...
  ConvertYUV420SPToARGB8888(input, input + width * height, output, width, height, uvFlipped);

  // Clean up after ourselves, releasing input and output arrays if we actually got them.
  if (!(bool) isInputDirect) {
    env->ReleaseByteArrayElements(inputArray, reinterpret_cast<jbyte*>(input), JNI_ABORT);
  }

  if (!(bool) isOutputDirect) {
    env->ReleaseIntArrayElements(outputArray, reinterpret_cast<jint*>(output), 0);
  }
}

JNIEXPORT void JNICALL
IMAGEUTILS_METHOD(argb8888ToYuv420sp)(
    JNIEnv* env, jclass clazz, jobject inputBuffer, jintArray inputArray, jboolean isInputDirect,
    jobject outputBuffer, jbyteArray outputArray, jboolean isOutputDirect,
    jint width, jint height) {

  // Assign the input pointer depending on whether the IntBuffer is direct or not.
  uint32_t* input;
  if ((bool) isInputDirect) {
    void* i = env->GetDirectBufferAddress(inputBuffer); // Trusting that this won't be null.
    input = reinterpret_cast<uint32_t*>(i);
  } else {
    jboolean inputCopy = JNI_FALSE;
    jint* i = env->GetIntArrayElements(inputArray, &inputCopy);
    input = reinterpret_cast<uint32_t*>(i);
  }

  // Assign the output pointer depending on whether the ByteBuffer is direct or not.
  uint8_t* output;
  if ((bool) isOutputDirect) {
    void* o = env->GetDirectBufferAddress(outputBuffer); // Trusting that this won't be null.
    output = reinterpret_cast<uint8_t*>(o);
  } else {
    jboolean outputCopy = JNI_FALSE;
    jbyte* o = env->GetByteArrayElements(outputArray, &outputCopy);
    output = reinterpret_cast<uint8_t*>(o);
  }

  // Actually perform the conversion with one line ...
  ConvertARGB8888ToYUV420SP(input, output, width, height);

  // Clean up after ourselves, releasing input and output arrays if we actually got them.
  if (!(bool) isInputDirect) {
    env->ReleaseIntArrayElements(inputArray, reinterpret_cast<jint*>(input), JNI_ABORT);
  }

  if (!(bool) isOutputDirect) {
    env->ReleaseByteArrayElements(outputArray, reinterpret_cast<jbyte*>(output), 0);
  }
}

