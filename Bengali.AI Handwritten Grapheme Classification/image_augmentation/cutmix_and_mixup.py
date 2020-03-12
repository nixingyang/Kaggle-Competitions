import numpy as np


def rand_bbox(height, width, lamda):
    # Size of the cropping region
    cut_ratio = np.sqrt(1 - lamda)
    cut_height = np.int(height * cut_ratio)
    cut_width = np.int(width * cut_ratio)

    # Coordinates of the center
    center_width = np.random.randint(width)
    center_height = np.random.randint(height)

    # Coordinates of the bounding box
    width_start = np.clip(center_width - cut_width // 2, 0, width)
    width_end = np.clip(center_width + cut_width // 2, 0, width)
    height_start = np.clip(center_height - cut_height // 2, 0, height)
    height_end = np.clip(center_height + cut_height // 2, 0, height)

    return width_start, width_end, height_start, height_end


def perform_cutmix(image_content_alpha,
                   one_hot_encoding_tuple_alpha,
                   image_content_beta,
                   one_hot_encoding_tuple_beta,
                   alpha=0.4):
    """
        https://github.com/clovaai/CutMix-PyTorch
        https://www.kaggle.com/c/bengaliai-cv19/discussion/126504
    """
    # Get lamda from a Beta distribution
    lamda = np.random.beta(alpha, alpha)

    # Get coordinates of the bounding box
    height, width = image_content_alpha.shape[:2]
    width_start, width_end, height_start, height_end = rand_bbox(
        height, width, lamda)
    lamda = 1 - (height_end - height_start) * (width_end -
                                               width_start) / (height * width)

    # Copy the region from the second image
    image_content = image_content_alpha.copy()
    image_content[height_start:height_end, width_start:
                  width_end] = image_content_beta[height_start:height_end,
                                                  width_start:width_end]

    # Modify the one hot encoding vector
    one_hot_encoding_tuple = []
    for one_hot_encoding_alpha, one_hot_encoding_beta in zip(
            one_hot_encoding_tuple_alpha, one_hot_encoding_tuple_beta):
        one_hot_encoding = one_hot_encoding_alpha * lamda + one_hot_encoding_beta * (
            1 - lamda)
        one_hot_encoding_tuple.append(one_hot_encoding)
    one_hot_encoding_tuple = tuple(one_hot_encoding_tuple)

    return image_content, one_hot_encoding_tuple


def perform_mixup(image_content_alpha,
                  one_hot_encoding_tuple_alpha,
                  image_content_beta,
                  one_hot_encoding_tuple_beta,
                  alpha=0.4):
    """
        https://github.com/facebookresearch/mixup-cifar10
        https://www.kaggle.com/c/bengaliai-cv19/discussion/126504
    """
    # Get lamda from a Beta distribution
    lamda = np.random.beta(alpha, alpha)

    # MixUp
    image_content = lamda * image_content_alpha + (1 -
                                                   lamda) * image_content_beta

    # Modify the one hot encoding vector
    one_hot_encoding_tuple = []
    for one_hot_encoding_alpha, one_hot_encoding_beta in zip(
            one_hot_encoding_tuple_alpha, one_hot_encoding_tuple_beta):
        one_hot_encoding = one_hot_encoding_alpha * lamda + one_hot_encoding_beta * (
            1 - lamda)
        one_hot_encoding_tuple.append(one_hot_encoding)
    one_hot_encoding_tuple = tuple(one_hot_encoding_tuple)

    return image_content, one_hot_encoding_tuple
