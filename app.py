import os
from flask import Flask, request, app, jsonify, url_for, render_template, flash, redirect
import tensorflow as tf
import numpy as np
import pandas as pd
from werkzeug.utils import secure_filename
from preprocessing import input_audio_preprocessing
from preprocessing import predict_from_input
import genres
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import InputRequired

app = Flask(__name__)




UPLOAD_FOLDER = 'upload_folder/'
ALLOWED_EXTENSIONS = {'wav', 'mp3'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = "secret"

class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload File")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

model = tf.keras.models.load_model('colab_model_96-96_84-08_83-78.keras')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/classify_api',methods=['GET','POST'])
def classify_api():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        file.save(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
        if file and allowed_file(file.filename):
            #filename = secure_filename(file.filename)
            #filename = file.filename
            #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            input_array = input_audio_preprocessing(os.path.join(os.path.abspath(os.path.dirname(__file__)),app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
            print(input_array.shape)
            prediction = predict_from_input(model, input_array)
            return jsonify(genres.Genre(prediction).name)
        return "An error occurred"
    return render_template('classify_page.html', form=form)    

if __name__ == "__main__":
    app.run(debug=True)
