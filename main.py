from flask import Flask , render_template, jsonify, request, make_response

import tensorflow as tf
import numpy as np
import cv2
import os
import time

app = Flask(__name__)

model = tf.keras.models.load_model(os.path.join(os.getcwd(), 'DenseNet121.h5')) # load .h5 Model

@app.route("/") 
def home_view(): 
	params = {	'covid' : 0}
	return render_template('index.html', params = params)



@app.route('/upload', methods=['POST'])
def upload():
	params = { 'covid' : 20}

	if (request.method == 'POST'):
		try:
			img = request.files['imageIN'] # Get Images from user
			# img.save(secure_filename(img.filename)) # for save image on server

			img = cv2.imdecode(np.fromstring(img.read(), np.uint8), cv2.IMREAD_COLOR) # preprocessing on image
			img = cv2.resize(img ,(224,224))
			img = np.array(img) / 255.0
			img = img.reshape(-1, 224, 224 ,3)
			
			prediction = model.predict(img) # get predictions on image
			print(prediction)


			params['covid'] = prediction[0,0] * 100


			params['status'] = True
		except:
			params['status'] = False
	return jsonify(params)

