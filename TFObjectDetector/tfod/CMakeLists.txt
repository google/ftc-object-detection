# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.4.1)

# Presumably, this makes CMAKE more verbose during compilation
set(CMAKE_VERBOSE_MAKEFILE on)

## Creates and names a library, sets it as either STATIC
## or SHARED, and provides the relative paths to its source code.
## You can define multiple libraries, and CMake builds them for you.
## Gradle automatically packages shared libraries with your APK.
#
#add_library( # Sets the name of the library.
#             native-lib
#
#             # Sets the library as a shared library.
#             SHARED
#
#             # Provides a relative path to your source file(s).
#             src/main/cpp/native-lib.cpp )
#
## Searches for a specified prebuilt library and stores the path as a
## variable. Because CMake includes system libraries in the search path by
## default, you only need to specify the name of the public NDK library
## you want to add. CMake verifies that the library exists before
## completing its build.
#
#find_library( # Sets the name of the path variable.
#              log-lib
#
#              # Specifies the name of the NDK library that
#              # you want CMake to locate.
#              log )
#
## Specifies libraries CMake should link to your target library. You
## can link multiple libraries, such as libraries you define in this
## build script, prebuilt third-party libraries, or system libraries.
#
#target_link_libraries( # Specifies the target library.
#                       native-lib
#
#                       # Links the target library to the log library
#                       # included in the NDK.
#                       ${log-lib} )


# TODO: See if we need to include the linker script?
# The linker script might be necessary to control the variables that are exported from the C++
# library. I think only the JNI variables need to be exported from the library?

# These flags are used for both of the libraries below
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSTANDALONE_DEMO_LIB \
                    -std=c++11 -fno-exceptions -fno-rtti -O2 -Wno-narrowing \
                    -fPIE")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} \
                              -Wl,--allow-multiple-definition \
                              -Wl,--whole-archive -fPIE -v")

# Create the image utils library
file(GLOB_RECURSE image_utils_sources ../tfod/src/main/cpp/image_utils/*.*)
add_library(image_utils
            SHARED
            ${image_utils_sources})
target_link_libraries(image_utils
        android
        log
        jnigraphics
        m
        atomic
        z
        )

# Create the object tracking library
file(GLOB_RECURSE object_tracking_sources ../tfod/src/main/cpp/object_tracking/*.*)
add_library(object_tracking
            SHARED
            ${object_tracking_sources})
target_link_libraries(object_tracking
        android
        log
        jnigraphics
        m
        atomic
        z
)