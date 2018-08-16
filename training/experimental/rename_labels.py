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
import sys

root = sys.argv[1]
files = list(os.listdir(root))
for name in files:

    new_name = name.split(".")[0] + ".txt"

    with open(os.path.join(root, name)) as f:
        lines = f.readlines()
        
        new_lines = []
        for line in lines[1:]:
            if "white_whiffle" in line:
                new_lines.append(line.replace("white_whiffle", "w"))
            elif "yellow_cube" in line:
                new_lines.append(line.replace("yellow_cube", "c"))

        if len(new_lines) > 0:
            with open(os.path.join(root, new_name), "w") as f2:
                f2.write("".join(new_lines))

    os.remove(os.path.join(root, name))
    print(name)
