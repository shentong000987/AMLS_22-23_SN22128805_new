# Importing all necessary libraries
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import pandas as pd

def B2_training():
	#The original img size
	img_width, img_height = 500, 500

	train_data_dir = './Datasets/cartoon_set/img'
	train_data_csv = './Datasets/cartoon_set/labels.csv'
	validation_data_dir = './Datasets/cartoon_set_test/img'
	validation_data_csv = './Datasets/cartoon_set_test/labels.csv'
	nb_train_samples =10000
	nb_validation_samples = 2500
	epochs = 5
	batch_size = 16

	#read csv file with separator symbol '\t'
	df_train = pd.read_csv(train_data_csv,sep='\t')
	#change the column value from int to str for later use
	df_train.eye_color = df_train.eye_color.apply(str)
	df_validation = pd.read_csv(validation_data_csv,sep='\t')
	#change the column value from int to str for later use
	df_validation.eye_color = df_validation.eye_color.apply(str)

	#make sure the channel is right
	if K.image_data_format() == 'channels_first':
		input_shape = (3, img_width, img_height)
	else:
		input_shape = (img_width, img_height, 3)

	#Three layers
	model = Sequential()

	#first layer with 32 filter , orignal size is 500*500, use large kernel at first
	model.add(Conv2D(32, (7, 7), input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	#second layer with 32 filter
	model.add(Conv2D(32, (5, 5)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	#third layer with 64 filter
	model.add(Conv2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	#Final Hidden Layer
	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))

	#multiclassification with 5 classes
	model.add(Dense(5))
	model.add(Activation('softmax'))


	model.compile(loss="categorical_crossentropy",
				optimizer= "adam",
				metrics=['accuracy'])


	train_datagen = ImageDataGenerator(rescale=1. / 255)

	test_datagen = ImageDataGenerator(rescale=1. / 255)

	train_generator = train_datagen.flow_from_dataframe(
		dataframe = df_train,
		directory = train_data_dir,
		x_col = 'file_name',
		y_col = 'eye_color',
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode='categorical')

	validation_generator = test_datagen.flow_from_dataframe(
		dataframe = df_validation,
		directory = validation_data_dir,
		x_col = 'file_name',
		y_col = 'eye_color',
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode='categorical')

	model.fit(
		train_generator,
		steps_per_epoch=nb_train_samples // batch_size,
		epochs=epochs,
		validation_data=validation_generator,
		validation_steps=nb_validation_samples // batch_size)


	model.save('model_B2.keras')

if __name__ == "__main__":
    B2_training()