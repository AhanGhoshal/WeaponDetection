import os
from keras.api.models import load_model
import numpy as np
import cv2
predicted = []
class_list = sorted(['knife', 'no weapon'])
label_names = sorted(class_list)


def preprocess(img, image_size=224):
    image = cv2.resize(img, (image_size, image_size))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float") / 255.0
    # Expand dimensions as predict expect image in batches
    image = np.expand_dims(image, axis=0)
    return image


def postprocess(image, results):
    # Split the results into class probabilities and box coordinates
    bounding_box, class_probs = results
    class_index = np.argmax(class_probs)

    # Use this index to get the class name.
    class_label = label_names[class_index]

    h, w = image.shape[:2]

    # Extract the Coordinates
    x1, y1, x2, y2 = bounding_box[0]

    # Convert the coordinates from relative (i.e. 0-1) to actual values
    x1 = int(w * x1)
    x2 = int(w * x2)
    y1 = int(h * y1)
    y2 = int(h * y2)

    # return the lable and coordinates
    return class_label, (x1, y1, x2, y2), class_probs


model = load_model('F:\Weapon Detection System\ANPR\WeaponPrediction.h5')
img_dir = 'F:\Weapon Detection System\ANPR\WeaponDataset\images'
test_image = os.listdir(img_dir)

for i, img in enumerate(test_image):
    if i<250:
        file_dir = os.path.join(img_dir, img)
        image = cv2.imread(file_dir)

        image = preprocess(image)

        #image = cv2.resize(image, (300, 300, 3))

        predict = model.predict(image)
        predicted_classes = np.argmax(predict[1], axis=1)
        predicted.append(predicted_classes)
    else:
        break

#predictions = model.predict(test_image, batch_size=32)
#predicted_classes = np.argmax(predictions['class_output'], axis=1)







import pandas as pd

df = pd.read_csv("annotations.csv")

df['is_knife'] = (df['class_name'] == 'knife').astype(int)


is_knife_list = df['is_knife'].tolist()

print(is_knife_list)
predicted_classes = [arr[0] for arr in predicted]
test_labels = is_knife_list
test_labels1 = is_knife_list[0:250]
print(predicted_classes)


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(test_labels1, predicted_classes)
print(f"Accuracy Score: {accuracy * 100:.2f}%")

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cf = confusion_matrix(test_labels1, predicted_classes)
cm_disp = ConfusionMatrixDisplay(confusion_matrix = cf, display_labels = [0, 1])


cm_disp.plot()
plt.show()