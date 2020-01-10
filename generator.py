import subprocess
import os
import os.path as osp
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py
import cv2
mnist_keys = ['train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
              't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte']

data_subset = ['constant', 'full', [1,5]]


def check_mnist_dir(data_dir):

    downloaded = np.all([osp.isfile(osp.join(data_dir, key)) for key in mnist_keys])
    if not downloaded:
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        download_mnist(data_dir)
    else:
        print('MNIST was found')


def download_mnist(data_dir):

    data_url = 'http://yann.lecun.com/exdb/mnist/'
    for k in mnist_keys:
        k += '.gz'
        url = (data_url+k).format(**locals())
        target_path = os.path.join(data_dir, k)
        cmd = ['curl', url, '-o', target_path]
        print('Downloading ', k)
        subprocess.call(cmd)
        cmd = ['gunzip', '-d', target_path]
        print('Unzip ', k)
        subprocess.call(cmd)


def extract_mnist(data_dir):

    num_mnist_train = 60000
    num_mnist_test = 10000

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_image = loaded[16:].reshape((num_mnist_train, 28, 28, 1))

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    train_label = np.asarray(loaded[8:].reshape((num_mnist_train)))

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_image = loaded[16:].reshape((num_mnist_test, 28, 28, 1))

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    test_label = np.asarray(loaded[8:].reshape((num_mnist_test)))

    return np.concatenate((train_image, test_image)), \
        np.concatenate((train_label, test_label))


def sample_coordinate(high, size):
    if high > 0:
        return np.random.randint(high, size=size)
    else:
        return np.zeros(size).astype(np.int)


def generator(config):
    # check if mnist is downloaded. if not, download it
    check_mnist_dir(config.mnist_path)

    # extract mnist images and labels
    image, label = extract_mnist(config.mnist_path)
    print('images extracted')
    h, w = config.digit_size

    # rs.shuffle(classes)
    num_class = image.shape[0]
    num_train, num_val, num_test = [
            int(float(ratio)/np.sum(config.train_val_test_ratio)*num_class)
            for ratio in config.train_val_test_ratio]

    dataset = {}
    dataset['train'] = image[:num_train], label[: num_train]
    dataset['val'] = image[num_train:num_train+num_val], label[: num_train]
    dataset['test'] = image[num_train+num_val:], label[: num_train]

    duplicate_detect = set()

    file_name = "scattered_mnist_{}x{}_obj{}x{}.hdf5".format(*config.image_size, *config.digit_size)
    with h5py.File(file_name, "w") as f:
        for set_name in ['train', 'val', 'test']:
            dset_data, dset_label = dataset[set_name]
            num_images = dset_data.shape[0]
            num_image_per_set = config.num_image_per_set
            h5set = f.create_group(set_name)

            for subset in data_subset:

                if subset == 'constant':
                    num_digit_in_image = [config.max_num_digit] * num_image_per_set
                    max_num_digit = config.max_num_digit
                elif subset == 'full':
                    num_digit_in_image = np.random.randint(1, config.max_num_digit, size=num_image_per_set)
                    max_num_digit = config.max_num_digit
                else:
                    low, high = subset
                    num_digit_in_image = np.random.randint(low, high, size=num_image_per_set)
                    subset = '{}-{}'.format(low, high)
                    max_num_digit = high

                print('processing {}.{}...'.format(set_name, subset), end='')

                all_h5set = h5set.create_group(subset)

                image_set = all_h5set.create_dataset('image'.format(set_name),
                                      shape = (num_image_per_set, *config.image_size),
                                      chunks = (32, *config.image_size,), # 32 images per batch
                                      dtype = np.float32,)

                bbox = all_h5set.create_dataset('bbox'.format(set_name),
                                      shape = (num_image_per_set, max_num_digit, 4), # X, Y, H, W
                                        chunks=(32, max_num_digit, 4),  # 32 images per batch
                                      dtype = np.float32,)

                digit_count = all_h5set.create_dataset('digit_count'.format(set_name),
                                      shape = (num_image_per_set, 1),
                                      chunks = (32, 1), # 32 images per batch
                                      dtype = np.float32,)


                for k in range(num_image_per_set):
                    # sample images

                    sampled_num_digit = num_digit_in_image[k] # instead of having a fixed number digits, we sample
                    rand_idx = np.random.randint(0, num_images, size = sampled_num_digit)
                    imgs = np.squeeze(dset_data[rand_idx, ...], axis=-1)
                    background = np.zeros((config.image_size)).astype(np.float32)
                    # sample coordinates
                    yt = sample_coordinate(config.image_size[0]-h, size = sampled_num_digit)
                    xt = sample_coordinate(config.image_size[1]-w, size = sampled_num_digit)
                    # combine images
                    for i in range(sampled_num_digit):
                        img = cv2.resize(imgs[i], dsize=tuple(config.digit_size))
                        background[yt[i]:yt[i]+h, xt[i]:xt[i]+w] += img


                    # write the image
                    background = np.clip(background, 0 , 255)
                    background /= 255 # [0, 255] -> [0, 1]

                    digit_count[k] = sampled_num_digit

                    # writing bounding box stuff
                    ys = np.ones_like(yt) * h
                    xs = np.ones_like(xt) * w
                    boxes = np.array([xt, yt, xs, ys]).T
                    bbox_container = np.ones((max_num_digit, 4)) * -1
                    bbox_container[:sampled_num_digit] = boxes
                    bbox[k] = bbox_container

                    image_set[k] = background

                    # For empty set detection
                    if background.sum() == 0:
                        print('Duplicate found!! Not Truly Random')
                        import ipdb; ipdb.set_trace()

                print('Done!')


    return image, label


def argparser():

    def str2bool(v):
        return v.lower() == 'true'

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mnist_path', type=str, default='./datasets/mnist/',
                        help='path to *.gz files')
    parser.add_argument('--multimnist_path', type=str, default='./datasets/multimnist')
    parser.add_argument('--max_num_digit', type=int, default=2)
    parser.add_argument('--train_val_test_ratio', type=int, nargs='+',
                        default=[64, 16, 20], help='percentage')
    parser.add_argument('--image_size', type=int, nargs='+',
                        default=[64, 64])
    parser.add_argument('--num_image_per_set', type=int, default=10000)
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument('--digit_size', type=int, nargs='+', default=[14, 14])
    config = parser.parse_args()
    return config


def main():

    config = argparser()
    assert len(config.train_val_test_ratio) == 3
    assert sum(config.train_val_test_ratio) == 100
    assert len(config.image_size) == 2
    generator(config)


if __name__ == '__main__':
    main()
