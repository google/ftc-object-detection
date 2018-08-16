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

from PIL import Image, ImageDraw, ImageFont
import io
import tensorflow as tf
fnt = ImageFont.truetype('LucidaSansRegular.ttf', 12)

flags = tf.app.flags
flags.DEFINE_string('records', 'records/train.records', 'Path to records to decode')
flags.DEFINE_string('decoded_dir', 'decoded', 'Path to write decoded records')
FLAGS = flags.FLAGS


example = tf.train.Example()
counter = 0
for record in tf.python_io.tf_record_iterator(FLAGS.records):
    example.ParseFromString(record)
    f = example.features.feature
    height = f['image/height'].int64_list.value[0]
    width = f['image/width'].int64_list.value[0]
    e = f['image/encoded'].bytes_list.value[0]

    im = Image.open(io.BytesIO(e))
    draw = ImageDraw.Draw(im)

    for i in range(len(f['image/object/class/text'].bytes_list.value)):
        class_text = f['image/object/class/text'].bytes_list.value[i]
        xmin = f['image/object/bbox/xmin'].float_list.value[i]
        ymin = f['image/object/bbox/ymin'].float_list.value[i]
        xmax = f['image/object/bbox/xmax'].float_list.value[i]
        ymax = f['image/object/bbox/ymax'].float_list.value[i]
        draw.rectangle([xmin*width, ymin*height, xmax*width, ymax*height], outline="rgb(255,0,0)")
        draw.text((xmin*width, ymin*height), class_text.decode('utf-8'), font=fnt, fill=(255,0,0,255))

    im.save(os.path.join(FLAGS.decoded_dir, "%05d.png" % counter)
    counter += 1
