import numpy as np
import os
import sys
import pickle

from PIL import Image


def cifar10_to_png(root, num_query, num_train):
    """
    Extract data from cifar-10 dataset and save to disk(png format).

    Args
        root(str): Path of dataset.
        num_query(int): Number of query images.
        num_train(int): Number of training images.
    """
    # Load dataset
    data_list = ['data_batch_1',
                 'data_batch_2',
                 'data_batch_3',
                 'data_batch_4',
                 'data_batch_5',
                 'test_batch',
                 ]
    base_folder = 'cifar-10-batches-py'

    data = []
    targets = []

    for file_name in data_list:
        file_path = os.path.join(root, base_folder, file_name)
        with open(file_path, 'rb') as f:
            if sys.version_info[0] == 2:
                entry = pickle.load(f)
            else:
                entry = pickle.load(f, encoding='latin1')
            data.append(entry['data'])
            if 'labels' in entry:
                targets.extend(entry['labels'])
            else:
                targets.extend(entry['fine_labels'])

    data = np.vstack(data).reshape(-1, 3, 32, 32)
    data = data.transpose((0, 2, 3, 1))  # convert to HWC
    targets = np.array(targets)

    # Sort by class
    sort_index = targets.argsort()
    data = data[sort_index, :]
    targets = targets[sort_index]

    # Save to disk
    root = os.path.join(root, 'images')
    os.mkdir(root)
    num_classes = 10
    num_data = len(data)
    num_query_per_class = num_query // num_classes
    num_train_per_class = num_train // num_classes
    num_database_per_class = (num_data - num_query - num_train) // num_classes
    num_data_per_class = num_data // num_classes # 6000

    # Query dataset
    os.mkdir(os.path.join(root, 'query'))
    for class_ in range(num_classes):
        os.mkdir(os.path.join(root, 'query', str(class_)))
        for i in range(num_query_per_class):
            img = data[class_ * num_data_per_class + i, :]
            img = Image.fromarray(img)
            img.save(os.path.join(root, 'query', str(class_), 'query_'+str(class_)+'_'+str(i)+'.png'))

    # Training dataset
    os.mkdir(os.path.join(root, 'train'))
    for class_ in range(num_classes):
        os.mkdir(os.path.join(root, 'train', str(class_)))
        for i in range(num_train_per_class):
            img = data[class_ * num_data_per_class + num_query_per_class + i, :]
            img = Image.fromarray(img)
            img.save(os.path.join(root, 'train', str(class_), 'train_'+str(class_)+'_'+str(i)+'.png'))

    # Database dataset(not conclude training dataset)
    os.mkdir(os.path.join(root, 'database'))
    for class_ in range(num_classes):
        os.mkdir(os.path.join(root, 'database', str(class_)))
        for i in range(num_database_per_class):
            img = data[class_ * num_data_per_class + num_query_per_class + num_train_per_class + i, :]
            img = Image.fromarray(img)
            img.save(os.path.join(root, 'database', str(class_), 'database_'+str(class_)+'_'+str(i)+'.png'))


if __name__ == '__main__':
    root = '/home/tree/Dataset/ImageRetrieval/cifar-10'
    num_query = 1000
    num_train = 5000
    cifar10_to_png(root, num_query, num_train)
