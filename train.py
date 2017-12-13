from argparse import ArgumentParser

from basic import build_model
from modules.file import save_model
from modules.image import load_images, to_dirname
#from modules.interface import show


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
    model.fit(x_images, y_images, batch_size=batch, epochs=epochs)
    model.save_weights('weights.hdf5')

    if test_dirname:
        x_test, y_test = load_images(test_dirname, image_size)
        eva = model.evaluate(x_test, y_test, batch_size=batch, verbose=1)
        print('Evaluate: ' + str(eva))


if __name__ == '__main__':
    main()
