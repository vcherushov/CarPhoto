from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import sqlite3
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



class MyHandler(FileSystemEventHandler):


    def on_any_event(self, event):
        pass

    def on_created(self, event):
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

        import psycopg2

        def update_table(car_id, number):
            try:
                # Подключиться к существующей базе данных
                connection = psycopg2.connect(user="postgres",
                                              # пароль, который указали при установке PostgreSQL
                                              password="Naruto2010",
                                              host="127.0.0.1",
                                              port="5432",
                                              database="car")

                cursor = connection.cursor()
                print("Таблица до обновления записи")
                sql_select_query = """select * from truck_detect where id = %s"""
                cursor.execute(sql_select_query, (car_id,))
                record = cursor.fetchone()
                print(record)

                # Обновление отдельной записи
                sql_update_query = """Update truck_detect set number = %s where id = %s"""
                cursor.execute(sql_update_query, (number, car_id))
                connection.commit()
                count = cursor.rowcount
                print(count, "Запись успешно обновлена")

                print("Таблица после обновления записи")
                sql_select_query = """select * from truck_detect where id = %s"""
                cursor.execute(sql_select_query, (car_id,))
                record = cursor.fetchone()
                print(record)

            except Exception as error:
                print("Ошибка при работе с PostgreSQL", error)
            finally:
                if connection:
                    cursor.close()
                    connection.close()
                    print("Соединение с PostgreSQL закрыто")


        tich_file = event.src_path
        print(tich_file)

        letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T',
                   'X', 'Y']


        img_name1 = tich_file
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

            # plt.imshow(img2)
            plt.xticks([]), plt.yticks([])  # Hides the graph ticks and x / y axis
            # plt.show()
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

            name = event.src_path[-6:-5]
            update_table(name, pred_texts)
        else:
            # plt.imshow(image0)
            plt.xticks([]), plt.yticks([])  # Hides the graph ticks and x / y axis
            # plt.show()
            print('нет')



    def on_deleted(self, event):
        pass

    def on_modified(self, event):
        pass

    def on_moved(self, event):
        pass


event_handler = MyHandler()
observer = Observer()
observer.schedule(event_handler, path='Object_Detection/test', recursive=False)
observer.start()





while True:
    try:
        pass
    except KeyboardInterrupt:
        observer.stop()