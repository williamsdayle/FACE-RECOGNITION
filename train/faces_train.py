import os
from PIL import Image
import numpy as np
import cv2
import pickle

def train(img_dir, flag):

    cascade_face = cv2.CascadeClassifier('../cascades/data/haarcascade_frontalface_alt2.xml')

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    if flag == 1:

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        image_dir = os.path.join(BASE_DIR, 'images')

    if flag == 2:

        image_dir = img_dir

    y_labels = []

    x_train = []

    current_id = 0

    labels_ids = {}

    for root, dirs, files in os.walk(image_dir):

        for file in files:

            if file.endswith('jpg'):

                path = os.path.join(root, file)

                label = os.path.basename(root).replace(' ', '-').lower()

                if label not in labels_ids:
                    labels_ids[label] = current_id
                    current_id += 1
                id_ = int(labels_ids[label])

                #y_labels.append(label)

                #x_train.append(path)

                pil_image = Image.open(path).convert('L')  # Conversão em escala de cinza

                size = (600, 600)

                final_image = pil_image.resize(size, Image.ANTIALIAS)

                image_array = np.array(final_image, 'uint8')

                faces = cascade_face.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)

                for (x, y, w, h) in faces:

                    segmentation = image_array[y:y + h, x:x + w]

                    x_train.append(segmentation)

                    y_labels.append(id_)

    with open('labels.pickle', 'wb') as f:
        pickle.dump(labels_ids, f)

    recognizer.train(x_train, np.array(y_labels))
    recognizer.save('trainner.yml')