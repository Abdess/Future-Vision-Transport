import cv2
import numpy as np
import random
import six

class_colors = [(random.randint(0, 255), random.randint(0, 255),
                 random.randint(0, 255)) for _ in range(5000)]


def pred_seg(model=None,
             inp=None,
             out_fname=None,
             n_classes=None,
             colors=class_colors,
             prediction_width=None,
             prediction_height=None,
             read_image_type=1):
    assert (inp is not None)
    assert ((type(inp) is np.ndarray) or isinstance(inp, six.string_types)), \
        "Les données d'entrée doivent être une image CV ou le nom du fichier d'entrée."

    if isinstance(inp, six.string_types):
        inp = cv2.imread(inp, read_image_type)

    assert (len(inp.shape) == 3 or len(inp.shape) == 1
            or len(inp.shape) == 4), "L'image devrait être h,w,3"

    output_width = 256
    output_height = 128
    input_width = prediction_width
    input_height = prediction_height
    n_classes = n_classes

    x = get_img_arr(inp, input_width, input_height, ordering='channels_last')
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)

    seg_img = show_seg_img(pr,
                           inp,
                           n_classes=n_classes,
                           colors=colors,
                           prediction_width=prediction_width,
                           prediction_height=prediction_height)

    if out_fname is not None:
        cv2.imwrite(out_fname, seg_img)

    return pr


def get_img_arr(image_input,
                width,
                height,
                imgNorm="sub_mean",
                ordering='channels_last',
                read_image_type=1):
    """ Chargement du tableau d'images à partir de l'entrée """

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        img = cv2.imread(image_input, read_image_type)

    if imgNorm == "sub_and_divide":
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif imgNorm == "sub_mean":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = np.atleast_3d(img)

        means = [103.939, 116.779, 123.68]

        for i in range(min(img.shape[2], len(means))):
            img[:, :, i] -= means[i]

        img = img[:, :, ::-1]
    elif imgNorm == "divide":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img / 255.0

    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img


def show_seg_img(seg_arr,
                 inp_img=None,
                 n_classes=None,
                 colors=class_colors,
                 prediction_width=None,
                 prediction_height=None):
    if n_classes is None:
        n_classes = np.max(seg_arr)

    seg_img = rgb_seg_img(seg_arr, n_classes, colors=colors)

    if inp_img is not None:
        original_h = inp_img.shape[0]
        original_w = inp_img.shape[1]
        seg_img = cv2.resize(seg_img, (original_w, original_h),
                             interpolation=cv2.INTER_NEAREST)

    if (prediction_height is not None) and (prediction_width is not None):
        seg_img = cv2.resize(seg_img, (prediction_width, prediction_height),
                             interpolation=cv2.INTER_NEAREST)
        if inp_img is not None:
            inp_img = cv2.resize(inp_img,
                                 (prediction_width, prediction_height))

    return seg_img


def rgb_seg_img(seg_arr, n_classes, colors=class_colors):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]

    seg_img = np.zeros((output_height, output_width, 3))

    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c) * (colors[c][2])).astype('uint8')

    return seg_img
