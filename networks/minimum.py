from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import Conv2D


def build_model(input_size):
	input_shape = (input_size[0], input_size[1], 3)
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
					input_shape=input_shape, padding='same'))
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu',
					padding='same'))
    model.add(Conv2D(filters=3, kernel_size=(3, 3), activation='relu',
					padding='sigmoid'))
	optimizer = Adam(lr=0.001)
	# optimizer = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=optimizer,loss='mse')
    return model