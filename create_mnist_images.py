import os
from skimage.io import imsave
from mnist import get_mnist_data

data_path = 'data/mnist/'
sets = ['train', 'test']
data_dict = get_mnist_data()

for set_name in sets:
    images = data_dict[set_name + '_images']
    labels = data_dict[set_name + '_labels']
    number_of_samples = images.shape[0]
    for i in range(number_of_samples):
        print(set, i)
        image = images[i]
        label = labels[i]

        if not os.path.exists(data_path + set_name + '/' + str(label) + '/'):
            os.makedirs(data_path + set_name + '/' + str(label) + '/')

        file_number = len(os.listdir(data_path + set_name + '/' + str(label) + '/'))
        imsave(data_path + set_name + '/' + str(label) + '/%05d.png' % file_number, image)
