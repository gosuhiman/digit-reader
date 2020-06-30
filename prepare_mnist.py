import os
import urllib.request
import gzip
import shutil

data_path = 'data/mnist/'


def download():
    mnist_url = 'http://yann.lecun.com/exdb/mnist/'

    if not os.path.exists(data_path):
        os.makedirs(data_path)

    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]

    for file in files:
        if not os.path.exists(data_path + file):
            print('Downloading ', file)
            urllib.request.urlretrieve(mnist_url + file, data_path + file)

    print('Download finished')


def extract():
    files = os.listdir(data_path)
    for file in files:
        if file.endswith('gz'):
            print('Extracting ', file)
            with gzip.open(data_path + file, 'rb') as f_in:
                with open(data_path + file.split('.')[0], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)

    for file in files:
        print('Removing ', file)
        os.remove(data_path + file)

    print('Files extracted')


download()
extract()
