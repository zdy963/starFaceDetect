import os
import random
import json
import pickle
import time
import cv2
import dlib
import numpy as np
from flask import Flask, render_template, request, g
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import img_to_array
import tensorflow as tf


app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
modelpath = 'transfull.model'


with open('config.json') as config:
    app.secret_key = json.load(config)['app_secret']


def loadModel():
    tf.reset_default_graph()

    global lb
    labelpath = 'transfull.lb'
    lb = pickle.loads(open(labelpath, "rb").read())

    global graph
    graph = tf.get_default_graph()

    # initialize cnn based face detector with the weights
    global cnn_face_detector
    cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')


def extractFaceCor(image, reduce=None):
    """
    Extract face coordinates, using CNN
    :param image:
    :param reduce:
    :return:
    """
    print('[INFO] Extracting face from image')
    origin_image = image
    dim = 0
    if not reduce:
        height, width, depth = image.shape
        image = cv2.pyrDown(image)
        dim +=1
        if height * width > 8e6:
            image = cv2.pyrDown(image)
            dim += 1
    else:
        for i in range(int(reduce)):
            image = cv2.pyrDown(image)
            dim += 1

    faces = cnn_face_detector(image, 1)
    if len(faces) == 0:
        cor = extractFaceCor(origin_image, dim-1)
    else:
        face_dict = {'lst': []}
        expand_perc = dim * 2
        for face in faces:
            x = face.rect.left()
            y = face.rect.top()
            w = face.rect.right() - x
            h = face.rect.bottom() - y
            face_dict['lst'].append(w * h)
            face_dict[w * h] = {'cor': [x * expand_perc, y * expand_perc, w * expand_perc, h * expand_perc]}

        cor = face_dict[max(face_dict['lst'])]['cor']

    return cor


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/upload', methods=['POST'])
def upload():
    """
    Receive image from front-end
    Then, predict
    :return: predict result
    """
    # Get image
    print('[INFO] Receiving image from front-end')
    img = request.files.get('image').read()

    # Change it into opencv form
    image = cv2.imdecode(np.fromstring(img, np.uint8), 1)

    # Copy an output
    output = image.copy()

    # Extract face from image
    cor = extractFaceCor(image)
    face = image[cor[1]:cor[1] + cor[3], cor[0]:cor[0] + cor[2]]
    cv2.imwrite('face.jpg', face)
    print('[INFO] Face extracted success!', cor)

    # Change the attrs for image to fit the model
    print('[INFO] Resizing face image')
    face = cv2.resize(face, (224, 224))
    # face = cv2.resize(face, (128, 128))
    face = face.astype("float") / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)

    # Predict similar stars
    print('[INFO] Predicting similar stars')
    model = load_model(modelpath)
    # with graph.as_default():
    proba = model.predict(face, verbose=1)[0]
    print(proba)

    # Get similar stars name and percentage
    labels = []
    for i in range(3):
        # print(i)
        idx = np.argmax(proba)
        print(idx)
        curr_prob = proba[idx]
        proba[idx] = 0
        label = lb.classes_[idx]

        # build the label and draw the label on the image
        result = {'label': label, 'prob': '%.2f' % (curr_prob*100) + '%'}
        labels.append(result)

    # Decode image
    curr_time = str(time.time())
    randomNum = random.randint(0, 100)  # 生成的随机整数n，其中0<=n<=100
    uniqueName = 'img/' + curr_time + str(randomNum) + '.jpg'
    cv2.imwrite('static/'+uniqueName, output)

    # Return data to front-end
    return render_template('show.html', img_path=uniqueName, pred=labels)


if __name__ == '__main__':
    print(('* Loading Keras model and Flask starting server...'
           'please wait until server has fully started'))
    loadModel()
    app.run(debug=True, port=5050)