import os
import urllib.request

data_path = 'data/mnist/'
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
