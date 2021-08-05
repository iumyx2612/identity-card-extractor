"""Miscellaneous utility functions."""

import os
import sys
import shutil
import re
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tqdm import tqdm
import time


def check_num_image_fit_with_annotation(src):
    """
    Check if number of images is same as number of annotations
    :param src: Folder to check, should be like this
    ┣ 📂1
    ┃ ┣ 📜image1.jpg
    ┃ ┣ 📜annotation1.xml
    ┃ ┣ 📜image2.jpg
    ┃ ┣ 📜annotation2.xml
    """
    num_annotations = num_images = 0
    image_format = [".jpeg", ".JPEG", ".Jpeg", ".jpg", ".JPG", ".png", ".PNG"]
    for file in os.listdir(src):
        if ".xml" in file:
            num_annotations += 1
        elif any(format in file for format in image_format):
            num_images += 1
    print("Num images: ", num_images)
    print("Num annotations: ", num_annotations)


def copy_images_and_annotations(src, dst):
    """
    Function to copy image with annotation next to it to separate image and annotation folder with order
    :param src: the source folder to copy from
    it should be like this
    📦A
    ┣ 📂1
    ┃ ┣ 📜image1.jpg
    ┃ ┣ 📜annotation1.xml
    ┃ ┣ 📜image2.jpg
    ┃ ┣ 📜annotation2.xml
    ┣ 📂2
    ┃ ┣ 📜image1.jpeg
    ┃ ┣ 📜annotation1.xml
    ┃ ┣ 📜image2.jpeg
    ┃ ┣ 📜annotation2.xml
    or like this
    ┣ 📂1
    ┃ ┣ 📜image1.jpg
    ┃ ┣ 📜annotation1.xml
    ┃ ┣ 📜image2.jpg
    ┃ ┣ 📜annotation2.xml
    :param dst: Destination to copy files to
    It should be like this
    📦A
    ┣ 📂Annotations
    ┗ 📂Images
    """
    image_format = [".jpeg", ".JPEG", ".Jpeg", ".jpg", ".JPG", ".png", ".PNG"]
    start_time = time.time()
    for folder in os.listdir(src):
        path = os.path.join(src, folder)
        if os.path.isdir(path):
            print("Copying at folder %s" % folder)
            for file in tqdm(os.listdir(path)):
                file_path = os.path.join(path, file)
                if ".xml" in file:
                    shutil.copy(file_path, os.path.join(dst, "Annotations"))
                elif any(format in file for format in image_format):
                    shutil.copy(file_path, os.path.join(dst, "Images"))
        else:
            if ".xml" in folder:
                shutil.copy(path, os.path.join(dst, "Annotations"))
            elif any(format in folder for format in image_format):
                shutil.copy(path, os.path.join(dst, "Images"))
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Took %s to copy all images" %elapsed_time)


def inspect_annotation_with_image(image_path, annotation_path):
    """
    Function to inspect a single image with its annotation
    :param image_path: image file path
    :param annotation_path: annotation file path
    :return: show the image with annotation
    """
    with Image.open(image_path) as image:
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        for member in root.findall('object'):
            # bbox contains 4 coordinate of format [xmin, ymin, xmax, ymax]
            bbox = member.find("bndbox")

            # if object is None, ignore
            if member.find("name") is None:
                continue

            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)

            draw = ImageDraw.Draw(image)
            draw.rectangle([xmin, ymin, xmax, ymax], width=2)
            draw.text([xmin, ymin], "%s"%member.find("name").text)
        image.show()


def inspect_every_images_with_annotations_of_folder(image_path, annotation_path):
    """
    Inspect every image along with its annotation from a folder
    Remember to close the image window before it shows a new one
    :param image_path: path to directory containing images
    :param annotation_path: path to directory containing annotation
    :return:
    """
    assert os.path.isdir(image_path)
    assert os.path.isdir(annotation_path)
    list_images = os.listdir(image_path)
    list_annotations = os.listdir(annotation_path)
    for i in range(len(list_images)):
        image_file, annotation_file = list_images[i], list_annotations[i]
        image = os.path.join(image_path, image_file)
        annotation = os.path.join(annotation_path, annotation_file)
        with Image.open(image) as img:
            tree = ET.parse(annotation)
            root = tree.getroot()
            for member in root.findall('object'):
                # bbox contains 4 coordinate of format [xmin, ymin, xmax, ymax]
                bbox = member.find("bndbox")

                # if object is None, ignore
                if member.find("name") is None:
                    continue

                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)

                draw = ImageDraw.Draw(img)
                draw.rectangle([xmin, ymin, xmax, ymax], width=2)
            img.show()





