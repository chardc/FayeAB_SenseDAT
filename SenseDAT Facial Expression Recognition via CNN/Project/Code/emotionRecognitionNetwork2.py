"""
Created on Sat Apr 27 11:44:13 2019

@author: Aswin Matthews Ashok

"""

import numpy as np
import keras as k
from keras.callbacks import ReduceLROnPlateau
from keras import optimizers

def trainCnnNetwork(trainX, trainY, valX, valY, storagePath):
	model = k.Sequential([
		k.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(48, 48, 1)),
		k.layers.experimental.preprocessing.RandomFlip("horizontal",
														   input_shape=(48,
																		48,
																		3)),
		k.layers.experimental.preprocessing.RandomRotation(0.1),
		k.layers.experimental.preprocessing.RandomZoom(0.1),

		k.layers.Conv2D(16, 3, padding='same', activation='relu'),
		k.layers.BatchNormalization(),
		k.layers.Conv2D(16, 3, padding='same', activation='relu'),
		k.layers.BatchNormalization(),
		k.layers.MaxPool2D(2, 2),

		k.layers.Conv2D(32, 3, padding='same', activation='relu'),
		k.layers.BatchNormalization(),
		k.layers.Conv2D(32, 3, padding='same', activation='relu'),
		k.layers.BatchNormalization(),
		k.layers.MaxPool2D(2, 2),

		k.layers.Conv2D(64, 3, padding='same', activation='relu'),
		k.layers.BatchNormalization(),
		k.layers.Conv2D(64, 3, padding='same', activation='relu'),
		k.layers.BatchNormalization(),
		k.layers.MaxPool2D(2, 2),

		k.layers.Conv2D(128, 3, padding='same', activation='relu'),
		k.layers.BatchNormalization(),
		k.layers.Conv2D(128, 3, padding='same', activation='relu'),
		k.layers.BatchNormalization(),
		k.layers.MaxPool2D(2, 2),

		k.layers.Flatten(),

		k.layers.Dense(256, activation='relu'),
		k.layers.Dropout(0.2),
		k.layers.Dense(128, activation='relu'),
		k.layers.Dropout(0.2),
		k.layers.Dense(64, activation='relu'),
		k.layers.Dropout(0.2),
		k.layers.Dense(7, activation='softmax')
	])

	model.compile(optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99),
				  loss='categorical_crossentropy',
				  metrics='accuracy')

	lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
	epochs = 200
	history = model.fit(trainX, trainY, validation_data=(valX, valY), epochs=epochs, batch_size=500,
						callbacks=[ReduceLROnPlateau()])
	# Print model summary
	model.summary()
	print(history.history.keys())
	# Storing Model
	model.save(storagePath)

def predictArrayLabels(data,model):
	categorical = model.predict(data)
	predictions = np.argmax(categorical,axis = 1)
	return predictions

def predictImageLabel(image,model):
	image = np.expand_dims(np.array([image]),axis = 3)
	data = image.astype(float)
	categorical = model.predict(data)
	return categorical