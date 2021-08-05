import os, glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse

import tensorflow
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple

tf = tensorflow.compat.v1
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)


def xml_to_csv(path):
    xml_list = []
    for xml_file in os.listdir(path):
        try:
            assert ".xml" in xml_file
            xml_path = os.path.join(path, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for member in root.findall('object'):
                # bbox contains 4 coordinate of format [xmin, ymin, xmax, ymax]
                bbox = member.find("bndbox")

                # if object is None, ignore
                if member.find("name") is None:
                    continue

                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member.find('name').text,
                         int(bbox.find('xmin').text),
                         int(bbox.find('ymin').text),
                         int(bbox.find('xmax').text),
                         int(bbox.find('ymax').text)
                         )
                xml_list.append(value)
            column_name = ['filename', 'width', 'height',
                           'class', 'xmin', 'ymin', 'xmax', 'ymax']
        except:
            pass
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def class_text_to_int(label_map_dict, row_label):
    return label_map_dict[row_label]


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path, label_map_dict):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        if row['class'] == 'ư':
            print(filename)
        classes.append(class_text_to_int(label_map_dict, row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def create_tf_record(labels_path, out_path, image_dir, xml_dir, csv_path, debug=None):
    label_map = label_map_util.load_labelmap(labels_path)
    label_map_dict = label_map_util.get_label_map_dict(label_map)
    writer = tf.python_io.TFRecordWriter(out_path)
    path = os.path.join(image_dir)
    examples = xml_to_csv(xml_dir)
    grouped = split(examples, 'filename')
    for idx, group in enumerate(grouped):
        if debug:
            if idx % 1 == 0:
                print('On image %d of %d, file name: %s' %(idx + 1, len(grouped), group[0]))
        else:
            if idx % 100 == 0:
                print('On image %d of %d' %(idx + 1, len(grouped)))
        tf_example = create_tf_example(group, path, label_map_dict)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created the TFRecord file: {}'.format(out_path))
    if csv_path is not None:
        examples.to_csv(csv_path, index=None)
        print('Successfully created the CSV file: {}'.format(csv_path))