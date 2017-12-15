from argparse import ArgumentParser
import os
import numpy as np
from PIL import Image

from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam,SGD
from keras import initializers

#python srcnn.py --size 64 64 --batch 2 --epoch 100 --input images/  --test tests/


def build_model(input_size):
    input_shape = (input_size[0], input_size[1], 3)
    model = Sequential()
    model.add(Conv2D(filters=64, kernel_size=(9, 9), activation='relu',kernel_initializer=initializers.random_normal(mean=0,stddev=0.001),bias_initializer='zeros',padding='same',use_bias=True,input_shape=input_shape))
    model.add(Conv2D(filters=32, kernel_size=(1, 1), activation='relu',kernel_initializer=initializers.random_normal(mean=0,stddev=0.001),bias_initializer='zeros',padding='same',use_bias=True))
    model.add(Conv2D(filters=3, kernel_size=(5, 5), activation='linear',kernel_initializer=initializers.random_normal(mean=0,stddev=0.001),bias_initializer='zeros',padding='same',use_bias=True))
    optimizer = Adam(lr=0.001)
    # optimizer = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=optimizer,loss='mse')
    return model
	
def save_model(model, name):
    json = model.to_json()#保存网络结构但不包含权值
    with open(name, 'w') as f:
        f.write(json)
		
def to_dirname(name):
    if name[-1:] == '/':
        return name
    else:
        return name + '/'
		
def load_images(name, size, ext='.jpg'):
    x_images = [] #x_images是从训练图像中随机裁剪32*32大小并降低图像质量再放大到64*64的lr图，y_images是64*64大小的hr图
    y_images = []
    for file in os.listdir(name):
        if os.path.splitext(file)[1] != ext:
            continue
        image = Image.open(name+file)
        if image.mode != "RGB":
            image.convert("RGB")
        x_image = image.resize((size[0]//2, size[1]//2))
        x_image = x_image.resize(size, Image.BICUBIC)#resize参数指定大小与质量 Image.NEAREST：最低质量 Image.BICUBIC：三次样条插值
        x_image = np.array(x_image)
        y_image = image.resize(size)
        y_image = np.array(y_image)
        x_images.append(x_image)
        y_images.append(y_image)
    x_images = np.array(x_images)
    y_images = np.array(y_images)

    x_images = x_images / 255
    y_images = y_images / 255
    return x_images, y_images
	
def get_args():
    description = 'Build SRCNN models and train'
    parser = ArgumentParser(description=description)
    parser.add_argument('-z', '--size', type=int, nargs=2, default=[128, 128], help='image size after expansion')
    parser.add_argument('-b', '--batch', type=int, default=64, help='batch size')
    parser.add_argument('-e', '--epoch', type=int, default=500, help='number of epochs')
    parser.add_argument('-i', '--input', type=str, default='images', help='data sets path')
    parser.add_argument('-t', '--test', type=str, default=None, help='test data path')
    return parser.parse_args()


def main():
    args = get_args()

    image_size = args.size 
    batch = args.batch 
    epochs = args.epoch 
    input_dirname = to_dirname(args.input) 
    if args.test:
        test_dirname = to_dirname(args.test) 
    else:
        test_dirname = False

    model = build_model(image_size)
    save_model(model, 'model.json')
    x_images, y_images = load_images(input_dirname, image_size)
    model.fit(x_images, y_images, batch_size=batch,epochs=epochs,verbose=1,validation_split=0.01)
    model.save_weights('weights.hdf5')

    if test_dirname:
        x_test, y_test = load_images(test_dirname, image_size)
        eva = model.evaluate(x_test, y_test, batch_size=batch, verbose=1)
        print('Evaluate: ' + str(eva))


if __name__ == '__main__':
    main()
