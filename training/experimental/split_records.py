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

import tensorflow as tf
import random
flags = tf.app.flags
flags.DEFINE_string('train_output_path', 'records/train.records', 'Path to output training TFRecord')
flags.DEFINE_string('eval_output_path', 'records/eval.records', 'Path to output eval TFRecord')
flags.DEFINE_string('input_path', 'records/label.records', 'Path to input TFRecord')
flags.DEFINE_float('train_fraction', 0.7, 'Fraction of records to include in training set')
FLAGS = flags.FLAGS

def main(_):
  train_writer = tf.python_io.TFRecordWriter(FLAGS.train_output_path)
  eval_writer = tf.python_io.TFRecordWriter(FLAGS.eval_output_path)
  reader = tf.python_io.tf_record_iterator(FLAGS.input_path)
  for example in reader:
      writer = train_writer if random.random() < FLAGS.train_fraction else eval_writer
      writer.write(example)
  train_writer.close()
  eval_writer.close()

if __name__ == '__main__':
  tf.app.run()
