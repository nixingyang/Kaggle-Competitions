"""
    https://github.com/akuxcw/GridMask/blob/master/imagenet_grid/utils/grid.py
    https://www.kaggle.com/haqishen/gridmask
"""

import numpy as np
from albumentations import ImageOnlyTransform
from PIL import Image


class Grid(object):

    def __init__(self, d_lower_bound, d_upper_bound, rotate, ratio, mode):
        self.d_lower_bound = d_lower_bound
        self.d_upper_bound = d_upper_bound
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode

    def __call__(self, img):
        h, w = img.shape[:2]
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = int(
            min(h, w) *
            np.random.uniform(low=self.d_lower_bound, high=self.d_upper_bound))
        l = int(d * self.ratio + 0.5)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh // d + 1):
            s = d * i + st_h
            t = s + l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t, :] *= 0
        for i in range(-1, ww // d + 1):
            s = d * i + st_w
            t = s + l
            s = max(min(s, ww), 0)
            t = max(min(t, ww), 0)
            mask[:, s:t] *= 0
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) //
                    2:(ww - w) // 2 + w]
        if self.mode == 1:
            mask = 1 - mask
        mask = np.expand_dims(mask, -1)
        img = img * mask
        return img


class GridMask(ImageOnlyTransform):  # pylint: disable=abstract-method

    def __init__(self,
                 d_lower_bound=96 / 224,
                 d_upper_bound=120 / 224,
                 rotate=360,
                 ratio=0.6,
                 mode=1,
                 always_apply=False,
                 p=0.8):
        super(GridMask, self).__init__(always_apply, p)
        self.gridmask_instance = Grid(d_lower_bound, d_upper_bound, rotate,
                                      ratio, mode)

    def apply(self, img, **params):
        return self.gridmask_instance(img)
