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

from google.protobuf import text_format
from object_detection.protos import string_int_label_map_pb2

LABEL_MAP="train_data/label.pbtxt"
f = open(LABEL_MAP).read()
m = text_format.Parse(f, string_int_label_map_pb2.StringIntLabelMap())
layers = [item.name for item in m.item]
layer_dict = dict([(item.id, item.name) for item in m.item])
