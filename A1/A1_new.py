# Importing all necessary libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import pandas as pd

def A1_training():
	#The original img size
	img_width, img_height = 178, 218

	train_data_dir = './Datasets/celeba/img'
	train_data_csv = './Datasets/celeba/labels.csv'
	validation_data_dir = './Datasets/celeba_test/img'
	validation_data_csv = './Datasets/celeba_test/labels.csv'
	nb_train_samples =5000
	nb_validation_samples = 1000
	epochs = 10
	batch_size = 16

	#read csv file with separator symbol '\t'
	df_train = pd.read_csv(train_data_csv,sep='\t')
	df_train.gender = df_train.gender.apply(str)
	df_validation = pd.read_csv(validation_data_csv,sep='\t')
	df_validation.gender = df_validation.gender.apply(str)

	#make sure the channel is right
	if K.image_data_format() == 'channels_first':
		input_shape = (3, img_width, img_height)
	else:
		input_shape = (img_width, img_height, 3)

	#Three layers
	model = Sequential()

	#first layer with 16 filter , orignal size is 178*218, use large kernel at first
	model.add(Conv2D(16, (5, 5), input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	#second layer with 32 filter
	model.add(Conv2D(32, (2, 2)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	#third layer with 64 filter
	model.add(Conv2D(64, (2, 2)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	#Final Hidden Layer
	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
				optimizer='rmsprop',
				metrics=['accuracy'])

	train_datagen = ImageDataGenerator(rescale=1. / 255)

	test_datagen = ImageDataGenerator(rescale=1. / 255)

	train_generator = train_datagen.flow_from_dataframe(
		dataframe = df_train,
		directory = train_data_dir,
		x_col = 'img_name',
		y_col = 'gender',
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode='binary')

	validation_generator = test_datagen.flow_from_dataframe(
		dataframe = df_validation,
		directory = validation_data_dir,
		x_col = 'img_name',
		y_col = 'gender',
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode='binary')

	training_history = model.fit(
		train_generator,
		steps_per_epoch=nb_train_samples // batch_size,
		epochs=epochs,
		validation_data=validation_generator,
		validation_steps=nb_validation_samples // batch_size)


	model.save('model_A1.keras')

if __name__ == "__main__":
    A1_training()