import codecs
import numpy
import os
import urllib.request
import gzip
import shutil
from skimage.io import imsave


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


class Mnist:
    def __init__(self, data_path):
        self.data_path = data_path

    def download(self):
        mnist_url = 'http://yann.lecun.com/exdb/mnist/'

        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)

        files_to_download = [
            'train-images-idx3-ubyte.gz',
            'train-labels-idx1-ubyte.gz',
            't10k-images-idx3-ubyte.gz',
            't10k-labels-idx1-ubyte.gz'
        ]

        for file in files_to_download:
            if not os.path.exists(self.data_path + file):
                print('Downloading ', file)
                urllib.request.urlretrieve(mnist_url + file, self.data_path + file)

        print('Download finished')

    def extract(self):
        files = os.listdir(self.data_path)
        for file in files:
            if file.endswith('gz'):
                print('Extracting ', file)
                with gzip.open(self.data_path + file, 'rb') as f_in:
                    with open(self.data_path + file.split('.')[0], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

        for file in files:
            print('Removing ', file)
            os.remove(self.data_path + file)

        print('Files extracted')

    def get_mnist_data(self):
        files = os.listdir(self.data_path)
        data_dict = {}
        for file in files:
            if file.endswith('ubyte'):
                print('Reading ', file)
                with open(self.data_path + file, 'rb') as f:
                    data = f.read()
                    magic_number = get_int(data[:4])
                    length = get_int(data[4:8])

                    if magic_number == 2051:
                        category = 'images'
                        num_rows = get_int(data[8:12])
                        num_cols = get_int(data[12:16])
                        parsed = numpy.frombuffer(data, dtype=numpy.uint8, offset=16)
                        parsed = parsed.reshape(length, num_rows, num_cols)
                    elif magic_number == 2049:
                        category = 'labels'
                        parsed = numpy.frombuffer(data, dtype=numpy.uint8, offset=8)
                        parsed = parsed.reshape(length)

                    if length == 10000:
                        set_name = 'test'
                    elif length == 60000:
                        set_name = 'train'

                    data_dict[set_name + '_' + category] = parsed

        return data_dict

    def convert_data_to_images(self):
        sets = ['train', 'test']
        data_dict = self.get_mnist_data()

        for set_name in sets:
            images = data_dict[set_name + '_images']
            labels = data_dict[set_name + '_labels']
            number_of_samples = images.shape[0]
            for i in range(number_of_samples):
                image = images[i]
                label = labels[i]

                path = self.data_path + 'images/' + set_name + '/' + str(label) + '/'

                if not os.path.exists(path):
                    os.makedirs(path)

                file_number = len(os.listdir(path))
                imsave(path + '%05d.png' % file_number, image)


mnist = Mnist('data/mnist/')
mnist.download()
mnist.extract()
mnist.convert_data_to_images()

print('MNIST data set ready')
