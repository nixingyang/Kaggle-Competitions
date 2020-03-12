from urllib.request import urlopen

import cv2
import numpy as np
from albumentations import Compose

if __name__ == "__main__":
    import augment_and_mix
    from autoaugment import AutoAugment
    from grid_mask import GridMask
    from HengCherKeng import HengCherKeng
    from random_erasing import RandomErasing
else:
    from . import augment_and_mix  # pylint: disable=relative-beyond-top-level
    from .autoaugment import AutoAugment  # pylint: disable=relative-beyond-top-level
    from .grid_mask import GridMask  # pylint: disable=relative-beyond-top-level
    from .HengCherKeng import HengCherKeng  # pylint: disable=relative-beyond-top-level
    from .random_erasing import RandomErasing  # pylint: disable=relative-beyond-top-level


class BaseImageAugmentor(object):

    def __init__(self):
        # Initiation
        self.transforms = [HengCherKeng(always_apply=True, p=1.0)]
        self.transformer = None

    def add_transforms(self, additional_transforms):
        self.transforms += additional_transforms

    def compose_transforms(self):
        self.transformer = Compose(transforms=self.transforms)

    def apply_augmentation(self, image_content_array):
        transformed_image_content_list = []
        for image_content in image_content_array:
            transformed_image_content = self.transformer(
                image=image_content)["image"]
            transformed_image_content_list.append(transformed_image_content)
        return np.array(transformed_image_content_list)


class AugMixImageAugmentor(BaseImageAugmentor):

    def __init__(self, **kwargs):
        super(AugMixImageAugmentor, self).__init__(**kwargs)
        augment_and_mix.IMAGE_SIZE = kwargs["image_height"]
        additional_transforms = [augment_and_mix.AugMix()]
        self.add_transforms(additional_transforms)


class AutoAugmentImageAugmentor(BaseImageAugmentor):

    def __init__(self, **kwargs):
        super(AutoAugmentImageAugmentor, self).__init__(**kwargs)
        additional_transforms = [AutoAugment()]
        self.add_transforms(additional_transforms)


class GridMaskImageAugmentor(BaseImageAugmentor):

    def __init__(self, **kwargs):
        super(GridMaskImageAugmentor, self).__init__(**kwargs)
        additional_transforms = [GridMask()]
        self.add_transforms(additional_transforms)


class RandomErasingImageAugmentor(BaseImageAugmentor):

    def __init__(self, **kwargs):
        super(RandomErasingImageAugmentor, self).__init__(**kwargs)
        additional_transforms = [RandomErasing()]
        self.add_transforms(additional_transforms)


def example():
    print("Loading the image content ...")
    raw_data = urlopen(
        url="https://avatars3.githubusercontent.com/u/15064790").read()
    raw_data = np.frombuffer(raw_data, np.uint8)
    image_content = cv2.imdecode(raw_data, cv2.IMREAD_COLOR)
    image_content = cv2.cvtColor(image_content, cv2.COLOR_BGR2RGB)

    print("Initiating the image augmentor ...")
    image_augmentor = BaseImageAugmentor()
    image_augmentor.compose_transforms()

    print("Generating the batch ...")
    image_content_list = [image_content] * 8
    image_content_array = np.array(image_content_list, dtype=np.float32)

    print("Applying data augmentation ...")
    image_content_array = image_augmentor.apply_augmentation(
        image_content_array)

    print("Visualization ...")
    for image_index, image_content in enumerate(image_content_array.astype(
            np.uint8),
                                                start=1):
        image_content = cv2.cvtColor(image_content, cv2.COLOR_RGB2BGR)
        cv2.imshow("image {}".format(image_index), image_content)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("All done!")


if __name__ == "__main__":
    example()
