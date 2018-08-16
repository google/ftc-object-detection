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

import numpy as np
from matplotlib import pyplot as plt

data = np.load("blur_experiment.npz")

# Make each column a set of data
blurs = data['blurs']
detections = data['detections'].T
weighted = data['weighted'].T

# Which blur configuration had the best results
best_detections = np.argmax(detections, axis=1)
best_weighted = np.argmax(weighted, axis=1)
best_weighted_blurs = blurs[best_weighted]


cumsum = np.cumsum(best_weighted_blurs)
average = cumsum / np.arange(1, len(cumsum) + 1)

print(best_weighted_blurs)
print(len(best_weighted_blurs))

max_counts = np.array([np.argmax(np.bincount(best_weighted_blurs[:i+1]))
                         for i in range(len(best_weighted_blurs))])

#  plt.plot(best_detections.T)
plt.plot(best_weighted_blurs)
plt.plot(average)
plt.plot(max_counts)

plt.ylabel("Blur Amount")
plt.xlabel("Frame Number")
plt.title("Weighted detection best performance")
plt.legend(["Highest Weighted Confidence Blur", "Best Average Blur",
            "Best Overall Blur"])


#  plt.plot(data['weighted'].T)
plt.show()
