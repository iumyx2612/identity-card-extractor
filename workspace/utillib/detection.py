import numpy as np
import sys, os
import tensorflow as tf
import time
import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


def load_model(path):
    """
    function to load the model with time logger
    :param path: exact path from current working directory to model
    :return: loaded model
    """
    print("Loading model...")
    start_time = time.time()

    # Load model and build the detection function
    detect_func = tf.saved_model.load(path)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("Done! Took {} seconds".format(elapsed_time))

    return detect_func


def load_label(path):
    """
    reads a label map and return a category index
    :param path: exact path from current working directory to label file
    :return: A category index, which is a dictionary that maps integer ids to dicts
    containing categories, e.g.
    {1: {'id': 1, 'name': 'dog'}, 2: {'id': 2, 'name': 'cat'}, ...}
    """
    category_index = label_map_util.create_category_index_from_labelmap(path, use_display_name=True)
    return category_index


def get_center_point(coordinate_dict):
    pass


def predict(model_path, label_path, image_path):
    """
    Function to display bbox and class prediction on selected image
    :param model_path: path to model folder, should contain "assets", "variables" and saved_model.pb
    :param label_path: path to label_map.pbtxt
    :param image_path: path to selected image
    :return:
    """
    print('Running inference for {}... '.format(image_path), end='')
    # load the model
    model = load_model(model_path)
    # load the label
    category_index = load_label(label_path)
    # load image into numpy array
    image_np = np.array(Image.open(image_path))
    # input neeeds to be a tensor
    input_tensor = tf.convert_to_tensor(image_np)
    # input expected to be in batch -> add new dim to input
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections
    num_detections = int(detections.pop('num_detections'))

    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    # apply non max suppression
    boxes = detections['detection_boxes']
    boxes_score = detections['detection_scores']
    selected_indices = tf.image.non_max_suppression(boxes=boxes,
                                                    scores=boxes_score,
                                                    max_output_size=4,
                                                    iou_threshold=0.3)
    selected_boxes = tf.gather(boxes, selected_indices).numpy()
    selected_scores = tf.gather(boxes_score, selected_indices).numpy()
    detections['detection_boxes'] = selected_boxes
    detections['detection_scores'] = selected_scores

    image_np_for_detections = image_np.copy()


    # visualize prediction
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image=image_np_for_detections,
        boxes=detections['detection_boxes'],
        classes=detections['detection_classes'],
        scores=detections['detection_scores'],
        category_index=category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=20,
        min_score_thresh=0.7
    )
    img = Image.fromarray(image_np_for_detections, 'RGB')
    img.show()