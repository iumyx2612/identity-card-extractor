import numpy as np
import sys, os
import tensorflow as tf
import time
import cv2

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util, config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


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


def predict_from_ckpt(path_to_config,
                      path_to_ckpt,
                      label_path,
                      image_path,
                      min_score=0.6,
                      iou_thresh=0.6,
                      box_to_visualize=20,
                      agnostic_mode=False,
                      skip_score=False,
                      apply_NMS=False):
    """
    Function to build the detection model from a checkpoint file
    :param path_to_config: path to config file
    :param path_to_ckpt: path to checkpoint file
    :param label_path: path to label file
    :param image_path: path to image file
    :param min_score: score threshold to display box
    :param iou_thresh: IoU threshold
    :param box_to_visualize: maximum output box after non max suppression
                            for corner detection this should be set to 4
                            for text detection this should be high enough to display all boxes that cover enough detections
    :param agnostic_mode: whether to display the label
    :param skip_score: whether to display the confidence score
    :param apply_NMS: True to use non-max suppression
    :return:
    """
    print("Building model from checkpoint...")
    start_load_model_time = time.time()

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(path_to_config)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(path_to_ckpt, 'ckpt-0')).expect_partial()

    end_load_model_time = time.time()
    elapsed_load_model_time = end_load_model_time-start_load_model_time
    print("Took %s to load model" %elapsed_load_model_time)

    print("Running detection for %s" %image_path)
    start_interference_time = time.time()
    # load the label
    category_index = load_label(label_path)
    # load image into numpy array
    image_np = np.array(Image.open(image_path))
    # input neeeds to be a tensor
    input_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)
    # input expected to be in batch -> add new dim to input
    input_tensor = input_tensor[tf.newaxis, ...]

    @tf.function
    def detect_fn(image, detection_model):
        """Detect objects in image."""

        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)

        return detections

    detections = detect_fn(input_tensor, detection_model)
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
    if apply_NMS:
        boxes = detections['detection_boxes']
        boxes_score = detections['detection_scores']
        selected_indices = tf.image.non_max_suppression(boxes=boxes,
                                                        scores=boxes_score,
                                                        max_output_size=box_to_visualize,
                                                        iou_threshold=iou_thresh)
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
        min_score_thresh=min_score,
        line_thickness=2,
        agnostic_mode=agnostic_mode,
        skip_scores=skip_score
    )
    img = Image.fromarray(image_np_for_detections, 'RGB')
    end_interference_time = time.time()
    elapsed_time = end_interference_time - start_interference_time
    print("Took %s to run detection" %elapsed_time)
    print("Total time %s"  %(elapsed_time + elapsed_load_model_time))
    img.show()


def predict_from_saved(model_path,
                       label_path,
                       image_path,
                       iou_thresh=0.3,
                       min_score=0.7,
                       box_to_visualize=20,
                       agnostic_mode=False,
                       skip_score=False,
                       apply_NMS=True):
    """
    Function to display bbox and class prediction on selected image
    This function only use with Faster R-CNN type model
    :param model_path: path to model folder, should contain "assets", "variables" and saved_model.pb
    :param label_path: path to label_map.pbtxt
    :param image_path: path to selected image
    :param iou_thresh: IoU threshold
    :param min_score: score threshold to display box
    :param box_to_visualize: maximum output box after non max suppression
                            for corner detection this should be set to 4
                            for text detection this should be high enough to display all boxes that cover enough detections
    :param agnostic_mode: whether to display the label
    :param skip_score: whether to display the confidence score
    :param apply_NMS: True to use non max suppression
    :return:
    """
    # load the model
    print("Loading model...")
    start_time = time.time()

    # Load model and build the detection function
    model = tf.saved_model.load(model_path)

    end_time = time.time()
    elapsed_load_model_time = end_time - start_time

    print("Done! Took {} seconds to load model".format(elapsed_load_model_time))

    print('Running inference for {}... '.format(image_path))
    start_time = time.time()
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
    if apply_NMS:
        boxes = detections['detection_boxes']
        boxes_score = detections['detection_scores']
        selected_indices = tf.image.non_max_suppression(boxes=boxes,
                                                        scores=boxes_score,
                                                        max_output_size=box_to_visualize,
                                                        iou_threshold=iou_thresh)
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
        min_score_thresh=min_score,
        line_thickness=2,
        agnostic_mode=agnostic_mode,
        skip_scores=skip_score
    )
    img = Image.fromarray(image_np_for_detections, 'RGB')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Took %s to run detection" %elapsed_time)
    print("Total time %s" %(elapsed_time + elapsed_load_model_time))
    img.show()


def visualize_proposal_regions(path_to_config,
                               path_to_ckpt,
                               label_path,
                               image_path,
                               min_score=0.6,
                               iou_thresh=0.6,
                               box_to_visualize=20,
                               apply_NMS=False):
    """
    Function to visualize proposal regions of RPN
    :param path_to_config: path to config file
    :param path_to_ckpt: path to checkpoint file
    :param label_path: path to label file
    :param image_path: path to image file
    :param min_score: score threshold to display box
    :param iou_thresh: IoU threshold
    :param box_to_visualize: maximum output box after non max suppression
                            for corner detection this should be set to 4
                            for text detection this should be high enough to display all boxes that cover enough detections
    :param apply_NMS: True to use non-max suppression
    :return:
    """
    print("Building model from checkpoint...")
    start_load_model_time = time.time()

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(path_to_config)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.join(path_to_ckpt, 'ckpt-0')).expect_partial()

    end_load_model_time = time.time()
    elapsed_time = end_load_model_time-start_load_model_time
    print("Took %s to load model" %elapsed_time)

    print("Running visualization of proposal regions for %s" %image_path)
    start_interference_time = time.time()
    # load the label
    category_index = load_label(label_path)
    # load image into numpy array
    image_np = np.array(Image.open(image_path))
    # input neeeds to be a tensor
    input_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)
    # input expected to be in batch -> add new dim to input
    input_tensor = input_tensor[tf.newaxis, ...]

    @tf.function
    def detect_fn(image, detection_model):
        """Detect objects in image."""

        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)

        return detections

    detections = detect_fn(input_tensor, detection_model)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections
    num_detections = int(detections.pop('num_detections'))

    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # apply non max suppression
    if apply_NMS:
        boxes = detections['detection_boxes']
        boxes_score = detections['detection_scores']
        selected_indices = tf.image.non_max_suppression(boxes=boxes,
                                                        scores=boxes_score,
                                                        max_output_size=box_to_visualize,
                                                        iou_threshold=iou_thresh)
        selected_boxes = tf.gather(boxes, selected_indices).numpy()
        selected_scores = tf.gather(boxes_score, selected_indices).numpy()
        detections['detection_boxes'] = selected_boxes
        detections['detection_scores'] = selected_scores

    image_np_for_detections = image_np.copy()

    # visualize prediction
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image=image_np_for_detections,
        boxes=detections['detection_boxes'],
        classes=None,
        scores=detections['detection_scores'],
        category_index=category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=500,
        min_score_thresh=min_score,
        skip_labels=True,
        agnostic_mode=True
    )
    img = Image.fromarray(image_np_for_detections, 'RGB')
    end_interference_time = time.time()
    elapsed_interference_time = end_interference_time - start_interference_time
    print("Took %s to run interference" %elapsed_interference_time)
    img.show()