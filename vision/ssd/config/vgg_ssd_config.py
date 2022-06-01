import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors


image_size = 512
image_mean = np.array([123, 117, 104])  # RGB layout
image_std = 1.0

iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

# ssd-300
# specs = [
#     SSDSpec(38, 8, SSDBoxSizes(30, 60), [2]),
#     SSDSpec(19, 16, SSDBoxSizes(60, 111), [2, 3]),
#     SSDSpec(10, 32, SSDBoxSizes(111, 162), [2, 3]),
#     SSDSpec(5, 64, SSDBoxSizes(162, 213), [2, 3]),
#     SSDSpec(3, 100, SSDBoxSizes(213, 264), [2]),
#     SSDSpec(1, 300, SSDBoxSizes(264, 315), [2])
# ]
# [4,6,6,6,4,4]
#ssd-512
specs = [
    SSDSpec(64, 8, SSDBoxSizes(30, 60), [2]),
    SSDSpec(32, 16, SSDBoxSizes(60, 111), [2, 3]),
    SSDSpec(16, 32, SSDBoxSizes(111, 162), [2, 3]),
    SSDSpec(8, 64, SSDBoxSizes(162, 213), [2, 3]),
    SSDSpec(4, 100, SSDBoxSizes(213, 264), [2]),
    SSDSpec(2, 300, SSDBoxSizes(264, 315), [2]),
    SSDSpec(1, 512, SSDBoxSizes(315, 366), [2])
]
# [4,6,6,6,4,4,4]

priors = generate_ssd_priors(specs, image_size)