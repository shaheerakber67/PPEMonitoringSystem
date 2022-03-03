import os
# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from absl import app, flags, logging
from absl.flags import FLAGS
import detect_video
from new_function import *
from flask import make_response
from flask import Flask, Response, render_template, request, redirect, url_for, send_file
import urllib.request
import gmail

upload_dir = 'temp'

data = [['ID','Class Name'],
        [0,'with_mask'],
        [1,'without_mask'],
        [2,'with_gloves'],
        [3,'without_gloves'],
        [4,'with_labcoat'],
        [5,'without_labcoat'],
]
path = "H:/NEW_FYP_project/yolov4-custom-functions-master/violations/" 
if not os.path.exists(upload_dir):
    os.makedirs(upload_dir)

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])



def main():
    
    if request.method == 'POST':
        video = request.files['video']
        url = request.form.get('url')

        if url != '':
            urllib.request.urlretrieve(url, os.path.join(upload_dir, 'video.mp4'))
        else:
            video.save(os.path.join(upload_dir, 'video.mp4'))

        return redirect(url_for('detection'))

    return render_template('index.html')

@app.route('/detection')
def detection():
    return render_template('detection.html')



@app.route('/report')
def report():
    return Response(PDF.print_page(path,data))

@app.route('/email')
def email():
    return Response(gmail.main())

@app.route('/video_feed')
def video_feed():
    return app.response_class(detect_video.main())

if __name__ == '__main__':
    app.run()
