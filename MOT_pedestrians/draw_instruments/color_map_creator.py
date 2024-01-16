import os

import numpy as np

from definitions import ROOT_DIR
from MOT_pedestrians.draw_instruments.mostly_different_colors import get_k_colors
import cv2

square_size = 40
k = 6
size = 800
for k in range(2,9,2):
    colors = get_k_colors(k*k).__iter__()
    img = np.zeros((size+1,size+1,3), np.uint8)
    step = size/k
    for left_bound in range(k):
        for up_bound in range(k):
            img[round(left_bound * step):round((left_bound+1) * step),round(up_bound * step):round((up_bound+1) * step)] = \
                list(reversed(colors.__next__()))
    cv2.imwrite(os.path.join(ROOT_DIR,f"examples/color_illustrations/{k*k}_colors.jpg"),img)
