import codecs
import numpy
import os

data_path = 'data/mnist/'

files = os.listdir(data_path)


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def get_mnist_data():
    data_dict = {}
    for file in files:
        if file.endswith('ubyte'):
            print('Reading ', file)
            with open(data_path + file, 'rb') as f:
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
