#!/usr/bin/env python3

# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse
import json
import subprocess

description_text = """\
Use this script to correctly rotate portrait videos for processing.

Oftentimes when recording portrait video, rather than setting the aspect ratio
to a portrait aspect ratio, the video is simply given a metadata tag which
indicates that the video should be interpreted as portrait. Unfortunately, this
tag is not interpreted by OpenCV when it opens videos, so the portrait videos
must actually be rotated. This script accomplishes that.

The specified folder is crawled to find all .mp4 files. If the file has a
metadata tag which indicates it is rotated, the video is re-encoded with ffmpeg
to correct the rotation. Checking the metadata requires ffprobe to be
installed, while re-encoding the video requires ffmpeg. Installing ffmpeg
usually installs ffprobe.

In short, run this script whenever importing new video files to label to make
sure they are processed correctly.
"""

epilog_text = """\
example:
    ./rotate_videos.py --dry [folder]          sanity check with a dry run
    ./rotate_videos.py -d [folder]             rotate videos, deleting originals
"""

parser = argparse.ArgumentParser(
        description=description_text,
        epilog=epilog_text,
        formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("folder", type=str, help="Folder containing videos")
parser.add_argument("-d", "--delete", action="store_true",
        default=False, help="Delete any files which get rotated")
parser.add_argument("--dry", action="store_true",
        default=False, help="Don't actually perform the conversion")
args = parser.parse_args()

for root, dirs, files in os.walk(args.folder):
    for name in files:
        if not name.endswith(".mp4"): continue

        file_path = os.path.join(root, name)
        #  print(file_path)

        # Determine if the file is rotated, and correct it.
        process = subprocess.run([
                "ffprobe",
                "-show_entries", "stream_tags=rotate",
                "-select_streams", "v:0",
                "-of", "json",
                file_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        output_json = process.stdout.decode('utf-8')
        output = json.loads(output_json)

        try:
            # Just need to access this.
            rotation = output['streams'][0]['tags']['rotate']
            print("Video rotated %d degrees" % int(rotation))

            # Attempt to rotate the video by the amount determined. Turns out
            # that just by encoding the video through ffmpeg with 0 filters
            # attached, we can get the rotation handled properly.
            new_name = os.path.splitext(name)[0] + "_rotated.mp4"
            new_path = os.path.join(root, new_name)

            print("Converting %s to %s" % (file_path, new_path))

            if not args.dry:
                subprocess.run([
                        "ffmpeg",
                        "-i", file_path,
                        "-an", # Get rid of audio
                        "-preset", "ultrafast", # try to speed ot up
                        new_path,
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )

                if args.delete:
                    os.remove(file_path)
                    print("Removed %s" % file_path)

        except KeyError as e:
            pass # This is fine, probably no rotation.
