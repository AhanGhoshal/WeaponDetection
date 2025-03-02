import numpy as np
from sklearn.model_selection import train_test_split
from xml.dom import minidom

import matplotlib.pyplot as plt
import cv2
import random
import os
from PIL import Image

import pandas as pd
from xml.dom import minidom
import csv

image_dir = r'F:\Weapon Detection System\ANPR\WeaponDataset\images'
annot_dir = r'F:\Weapon Detection System\ANPR\WeaponDataset\annotations\xmls'


def rescaling(path_image, targetSize, xmin, ymin, xmax, ymax):
    imageToPredict = cv2.imread(path_image, 3)

    y_ = imageToPredict.shape[0]
    x_ = imageToPredict.shape[1]

    x_scale = targetSize / x_
    y_scale = targetSize / y_
    img = cv2.resize(imageToPredict, (targetSize, targetSize));
    img = np.array(img);

    (origLeft, origTop, origRight, origBottom) = (xmin, ymin, xmax, ymax)

    xmin = int(np.round(origLeft * x_scale))
    ymin = int(np.round(origTop * y_scale))
    xmax = int(np.round(origRight * x_scale))
    ymax = int(np.round(origBottom * y_scale))

    return img, xmin, ymin, xmax, ymax


def extract_xml_contents(annot_directory, image_dir, target_size=300):
    file = minidom.parse(annot_directory)

    height, width = cv2.imread(image_dir).shape[:2]

    xmin = file.getElementsByTagName('xmin')
    x1 = float(xmin[0].firstChild.data)

    ymin = file.getElementsByTagName('ymin')
    y1 = float(ymin[0].firstChild.data)

    xmax = file.getElementsByTagName('xmax')
    x2 = float(xmax[0].firstChild.data)

    ymax = file.getElementsByTagName('ymax')
    y2 = float(ymax[0].firstChild.data)

    class_name = file.getElementsByTagName('name')
    if class_name[0].firstChild.data == 'knife':
        class_num = 1
    else:
        class_num = 0

    files = file.getElementsByTagName('filename')
    file_name = files[0].firstChild.data

    img, xmin, ymin, xmax, ymax = rescaling(image_dir, target_size, x1, y1, x2, y2)
    width = img.shape[0]
    height = img.shape[1]
    return file_name, width, height, class_num, xmin, ymin, xmax, ymax


def xml_to_csv(image_dir, annot_dir):
    xml_list = []

    mat_files = os.listdir(annot_dir)
    img_files = os.listdir(image_dir)

    for i, image in enumerate(img_files):
        xp = image.split('.')
        mat_path = os.path.join(annot_dir, (str(xp[0]) + '.xml'))
        img_path = os.path.join(image_dir, image)
        value = extract_xml_contents(mat_path, img_path)

        xml_list.append(value)

    columns_name = ['file_name', 'width', 'height', 'class_num',
                    'xmin', 'ymin', 'xmax', 'ymax']

    xml_df = pd.DataFrame(xml_list, columns=columns_name)

    return xml_df


train_labels_df = xml_to_csv(image_dir, annot_dir)
train_labels_df.to_csv(('dataset.csv'), index=None)

class_list = sorted(['knife', 'no weapon'])


def preprocess_dataset(image_dir, csv_file):
    labels = []
    boxes = []
    img_list = []

    with open(csv_file) as csvfile:

        rows = csv.reader(csvfile)
        columns = next(iter(rows))
        none = {}
        for i, row in enumerate(rows):

            img_path = row[0]
            full_path = os.path.join(image_dir, img_path)
            img = cv2.imread(full_path)
            if img is None:
                none[i] = str(full_path)
            else:
                img = cv2.imread(full_path)

                image = cv2.resize(img, (224, 224))

                image = image.astype("float") / 255.0

                img_list.append(image)

                labels.append(int(row[3]))

                arr = [float(row[4]) / 224,
                       float(row[5]) / 224,
                       float(row[6]) / 224,
                       float(row[7]) / 224]
                boxes.append(arr)

        return labels, boxes, img_list, none


train_labels, train_boxes, train_img, train_none = preprocess_dataset(image_dir, 'dataset.csv')

combined_list = list(zip(train_img, train_boxes, train_labels))
random.shuffle(combined_list)
train_img, train_boxes, train_labels = zip(*combined_list)

plt.figure(figsize=(15, 20));

random_range = random.sample(range(1, len(train_img)), 20)

for itr, i in enumerate(random_range, 1):
    a1, b1, a2, b2 = train_boxes[i]
    img_size = 300

    x1 = a1 * img_size
    x2 = a2 * img_size
    y1 = b1 * img_size
    y2 = b2 * img_size

    image = train_img[i]

    cv2.rectangle(image, (int(x1), int(y1)),
                  (int(x2), int(y2)),
                  (0, 255, 0),
                  3);

    img = np.clip(train_img[i], 0, 1)
    plt.subplot(4, 5, itr)
    plt.imshow(img)
    plt.axis('off')

train_images, val_images, train_labels, val_labels, train_boxes, val_boxes = train_test_split(np.array(train_img),
                                                                                              np.array(train_labels),
                                                                                              np.array(train_boxes),
                                                                                              test_size=0.1,
                                                                                              random_state=43)

print('Total Training Images: {}, Total Test Images: {}'.format(
    len(train_images),
    len(val_images)))

import tensorflow as tf
from keras.api.layers import Dense, Input
from keras.api.models import Model

from keras.api.layers import GlobalAveragePooling2D, Dropout
from keras.api.optimizers import SGD

image_size = 224

N_mobile = tf.keras.applications.NASNetMobile(input_tensor=Input(
    shape=(image_size, image_size, 3)),
    include_top=False,
    weights='imagenet')


def create_model(no_of_classes):
    N_mobile.trainable = False
    base_model_output = N_mobile.output
    flattened_output = GlobalAveragePooling2D()(base_model_output)
    print(flattened_output.shape)
    class_prediction = Dense(256, activation="relu")(flattened_output)
    class_prediction = Dense(128, activation="relu")(class_prediction)
    class_prediction = Dropout(0.2)(class_prediction)
    class_prediction = Dense(64, activation="relu")(class_prediction)
    class_prediction = Dropout(0.2)(class_prediction)
    class_prediction = Dense(32, activation="relu")(class_prediction)
    class_prediction = Dense(no_of_classes, activation='softmax', name="class_output")(class_prediction)
    box_output = Dense(256, activation="relu")(flattened_output)
    box_output = Dense(128, activation="relu")(box_output)
    box_output = Dropout(0.2)(box_output)

    box_output = Dense(64, activation="relu")(box_output)
    box_output = Dropout(0.2)(box_output)

    box_output = Dense(32, activation="relu")(box_output)
    box_predictions = Dense(4, activation='sigmoid',
                            name="box_output")(box_output)

    model = Model(inputs=N_mobile.input, outputs=[box_predictions, class_prediction])

    return model


model = create_model(2)
model.summary()

losses = {
    "box_output": "mean_squared_error",
    "class_output": "sparse_categorical_crossentropy"
}
loss_weights = {
    "box_output": 1.0,
    "class_output": 1.0
}

metrics = {
    'class_output': 'accuracy',
    'box_output': 'mse'
}
stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0.0001, patience=40,
                                        restore_best_weights=True
                                        )

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.0002,
                                                 patience=30, min_lr=1e-7, verbose=0)

opt = SGD(learning_rate=1e-3, momentum=0.9)

model.compile(optimizer=opt, loss=losses, loss_weights=loss_weights,
              metrics=metrics)

with tf.device('/device:GPU:0'):
    history = model.fit(x=train_images,
                        y={
                            "box_output": train_boxes,
                            "class_output": train_labels
                        },
                        validation_data=(
                            val_images,
                            {
                                "box_output": val_boxes,
                                "class_output": val_labels
                            }), batch_size=32, epochs=25,
                        callbacks=[reduce_lr, stop])





model.save("WeaponPrediction.h5")
