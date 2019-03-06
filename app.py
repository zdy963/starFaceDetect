import os
import random
import json
import pickle
import time
import cv2
import numpy as np
from flask import Flask, render_template, request, g
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import img_to_array
import tensorflow as tf


app = Flask(__name__)
basedir = os.path.abspath(os.path.dirname(__file__))
modelpath = 'transfull_06.model'


with open('config.json') as config:
    app.secret_key = json.load(config)['app_secret']


def loadModel():
    tf.reset_default_graph()

    global lb
    labelpath = 'transfull.lb'
    lb = pickle.loads(open(labelpath, "rb").read())

    global graph
    graph = tf.get_default_graph()

    global net
    prototxt = 'deploy.prototxt.txt'
    model = 'res10_300x300_ssd_iter_140000.caffemodel'
    # load our serialized model from disk
    net = cv2.dnn.readNetFromCaffe(prototxt, model)


def extractFaceCor2(image):
    """
    Extract Face Coordinates using opencv and dnn
    :param image:
    :return:
    """
    print('[INFO] Using Opencv with DNN to Extract face')
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
    (startX, startY, endX, endY) = box.astype("int")
    cor = [startX, startY, endX-startX, endY-startY]
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
    image = cv2.pyrDown(image)

    # Copy an output
    output = image.copy()

    # Extract face from image
    cor = extractFaceCor2(image)
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
    app.run(host='0.0.0.0', debug=True, port=5050)