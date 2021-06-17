# run by typing python3 main.py in a terminal 
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, render_template
from flask import send_from_directory
from ai import get_yolo_net, yolo_forward, yolo_save_img
from utils import allowed_file, and_syntax

# setup the webserver
app = Flask(__name__)
UPLOAD_FOLDER = 'images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024

# set up yolo net for prediction
# you will need to change names, weights, and configuration files.
names_path = os.path.join('yolo', 'yolo.names')
LABELS = open(names_path).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weights_path = os.path.join('yolo', 'yolo.weights')
cfg_path = os.path.join('yolo', 'yolo.cfg')
net = get_yolo_net(cfg_path, weights_path)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/', methods=['POST'])
def home_post():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for('results', filename=filename))

@app.route('/uploads/<filename>')
def results(filename): 
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image = cv2.imread(image_path)
    (class_ids, labels, boxes, confidences) = yolo_forward(net, LABELS, image, confidence_level=0.3)

    # if we detected something
    if len(class_ids) > 0:
        found = True
        new_filename = 'yolo_' + filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
        yolo_save_img(image, class_ids, boxes, labels, confidences, COLORS, file_path=file_path)

        # confidences: rounding and changing to percent, putting in function
        format_confidences = []
        for percent in confidences:
            format_confidences.append(str(round(percent*100)) + '%')
        format_confidences = and_syntax(format_confidences)
        # labels: sorting and capitalizing, putting into function
        labels = set(labels)
        labels = [plant.capitalize() for plant in labels]
        labels = and_syntax(labels)
        return render_template('results.html', confidences=format_confidences, labels=labels, 
            old_filename = filename, 
            filename=new_filename) 
    else:
        found = False
        # replace 'Objects' with whatever you are trying to detect
        return render_template('results.html', labels='No Objects', old_filename=filename, filename=filename) 

@app.route('/files/<path:filename>')
def files(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

if __name__ == "__main__":
    # Only for debugging while developing
    app.run(port=80, debug=True)
