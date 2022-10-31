import os
import cv2
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from skimage.feature import canny
from skimage.transform import hough_line, hough_line_peaks
from skimage.transform import rotate
from skimage.color import rgb2gray
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
import itertools
import glob
import tensorflow as tf
import requests,io
print('TensorFlow version:', tf.__version__)

if not os.path.isfile('./model_resnet.tflite'):
    API_ENDPOINT = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={}'
    link_r = requests.get(API_ENDPOINT.format('https://disk.yandex.ru/d/QavLH1pvpRhLOA'))
    response= requests.get(link_r.json()['href'])
    if response.status_code == 200:
        with open("./model_resnet.tflite", 'wb') as f:
            f.write(response.content)

tich_file=glob.glob('./Object_Detection/test/*')
print(len(tich_file),tich_file)


letters=['0' ,'1' ,'2','3' ,'4' ,'5', '6', '7', '8','9', 'A', 'B', 'C', 'E' ,'H','K', 'M','O', 'P', 'T', 'X', 'Y']

def decode_batch(out):
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = ''
        for c in out_best:
            if c < len(letters):
                outstr += letters[c]
        ret.append(outstr)
    return ret



for i in range(len(tich_file)):
    img_name1 = tich_file[i]
    path = img_name1
    image0 = cv2.imread(img_name1, 1)
    image_height, image_width, _ = image0.shape
    image = cv2.resize(image0, (1024, 1024))
    image = image.astype(np.float32)
    paths = './model_resnet.tflite'
    interpreter = tf.lite.Interpreter(model_path=paths)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    X_data1 = np.float32(image.reshape(1, 1024, 1024, 3))
    input_index = (interpreter.get_input_details()[0]['index'])
    interpreter.set_tensor(input_details[0]['index'], X_data1)
    interpreter.invoke()
    detection = interpreter.get_tensor(output_details[0]['index'])
    net_out_value2 = interpreter.get_tensor(output_details[1]['index'])
    net_out_value3 = interpreter.get_tensor(output_details[2]['index'])
    net_out_value4 = interpreter.get_tensor(output_details[3]['index'])
    img = image0
    razmer = img.shape

    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Converts from one colour space to the other
    img3 = img[:, :, :]

    box_x = int(detection[0, 0, 0] * image_height)
    box_y = int(detection[0, 0, 1] * image_width)
    box_width = int(detection[0, 0, 2] * image_height)
    box_height = int(detection[0, 0, 3] * image_width)
    if np.min(detection[0, 0, :]) >= 0:
        cv2.rectangle(img2, (box_y, box_x), (box_height, box_width), (230, 230, 21), thickness=5)

        #plt.imshow(img2)
        plt.xticks([]), plt.yticks([])  # Hides the graph ticks and x / y axis
        #plt.show()
        image = img3[box_x:box_width, box_y:box_height, :]
        grayscale = rgb2gray(image)
        edges = canny(grayscale, sigma=3.0)
        out, angles, distances = hough_line(edges)
        _, angles_peaks, _ = hough_line_peaks(out, angles, distances, num_peaks=20)
        angle = np.mean(np.rad2deg(angles_peaks))
        if 0 <= angle <= 90:
            rot_angle = angle - 90
        elif -45 <= angle < 0:
            rot_angle = angle - 90
        elif -90 <= angle < -45:
            rot_angle = 90 + angle
        if abs(rot_angle) > 20:
            rot_angle = 0
        rotated = rotate(image, rot_angle, resize=True) * 255
        rotated = rotated.astype(np.uint8)
        rotated1 = rotated[:, :, :]
        minus = np.abs(int(np.sin(np.radians(rot_angle)) * rotated.shape[0]))
        if rotated.shape[1] / rotated.shape[0] < 2 and minus > 10:
            rotated1 = rotated[minus:-minus, :, :]
        lab = cv2.cvtColor(rotated1, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        paths = './model1_nomer.tflite'
        interpreter = tf.lite.Interpreter(model_path=paths)
        interpreter.allocate_tensors()
        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        #         img =rotated1
        img = final  # лучше работает при плохом освещении

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (128, 64))
        img = img.astype(np.float32)
        img /= 255

        img1 = img.T
        img1.shape
        X_data1 = np.float32(img1.reshape(1, 128, 64, 1))
        input_index = (interpreter.get_input_details()[0]['index'])
        interpreter.set_tensor(input_details[0]['index'], X_data1)

        interpreter.invoke()

        net_out_value = interpreter.get_tensor(output_details[0]['index'])
        pred_texts = decode_batch(net_out_value)
        print(pred_texts)
    else:
        #plt.imshow(image0)
        plt.xticks([]), plt.yticks([])  # Hides the graph ticks and x / y axis
        #plt.show()
        print('нет')