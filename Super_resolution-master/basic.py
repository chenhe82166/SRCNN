from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam,SGD


def build_model(input_size):
    input_shape = (input_size[0], input_size[1], 3)
    model = Sequential()
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu',init='glorot_uniform',padding='same',use_bias=True,input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu',init='glorot_uniform',padding='same',use_bias=True))
    model.add(Conv2D(filters=3, kernel_size=(3, 3), activation='sigmoid',init='glorot_uniform',padding='same',use_bias=True))
    optimizer = Adam(lr=0.001)
    # optimizer = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=optimizer,loss='mse')
    return model
