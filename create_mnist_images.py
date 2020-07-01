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
        print(set_name, i)
        image = images[i]
        label = labels[i]

        path = data_path + 'images/' + set_name + '/' + str(label) + '/'

        if not os.path.exists(path):
            os.makedirs(path)

        file_number = len(os.listdir(path))
        imsave(path + '%05d.png' % file_number, image)
