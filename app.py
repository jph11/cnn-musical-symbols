from flask import Flask, render_template, request

from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import re
import sys
import os
import base64
import codecs
sys.path.append(os.path.abspath('./'))

from load import *

# init flask app
app = Flask(__name__)

global model, graph
model, graph = init()

def convertImage(imgData1):
	imgstr = re.search(b'base64,(.*)',imgData1).group(1)
	#imgstr = imgstr.decode('utf-8')
	#imgstr += 1
	#imgstr = imgData1[imgstr:]
	#imgstr = str(imgstr, 'utf-8')
	print(imgstr)
	with open('output.png', 'wb') as output:
		#output.write(imgstr.decode('base64'))
		output.write(codecs.decode(imgstr, 'base64'))

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict/', methods=['GET', 'POST'])
def predict():
	imgData = request.get_data()
	convertImage(imgData)

	#THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
	#my_out = os.path.join(THIS_FOLDER, 'out.png')
	im = load_img('output.png', grayscale=True, target_size=[40, 40])
	x = np.asarray(im).astype('float32') / 255
	x = x.reshape(1, 40, 40, 1)
	with graph.as_default():
		out = model.predict(x)
		response = np.array_str(np.argmax(out,axis=1))
		return response

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port) 
