import numpy as np
from keras.datasets import cifar10, cifar100, fashion_mnist, mnist
from skimage.transform import resize


def convert_to8x8(data):
    split = 8
    if data.shape[1] == 28:
        data = resize(data, (data.shape[0], 24, 24), anti_aliasing=True)
    #print(data.shape)
    #plt.imshow(data[0], cmap='gray')
    #plt.show()
    data = np.hsplit(data, split)
    for i, img_row in enumerate(data):
        data[i] = np.dsplit(img_row, split)
        #print([[d.shape for d in data[i]] for i in range(len(data))])
    for i in range(len(data)):
        for j in range(len(data[i])):
            #print(data[i][j].shape)
            data[i][j] = np.sum(data[i][j], axis=(1, 2))
            data[i][j] = np.expand_dims(data[i][j], axis=-1)
            data[i][j] = np.expand_dims(data[i][j], axis=-1)
        data[i] = np.concatenate(data[i], axis=-1)
        #data[i] = np.expand_dims(data[i], axis=-1)
    data = np.concatenate(data, axis=-2)
    #plt.imshow(data[0], cmap='gray')
    #plt.show()
    #print(data.shape)
    data = data / np.amax(data)
    return data


def give_data_back():
    rgb_weights = [0.2989, 0.5870, 0.1140]
    x, y, xt, yt = [], [], [], []
    #for i, fct in enumerate([mnist, fashion_mnist, cifar10, cifar100]):
    for i, fct in enumerate([mnist]):
        (x_train, y_train), (x_test, y_test) = fct.load_data()
        print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
        if len(x_train.shape) == 4:
            #plt.imshow(x_train[0])
            #plt.show()
            x_train = np.dot(x_train, rgb_weights)
            x_test = np.dot(x_test, rgb_weights)
            #print(x_train.shape, x_test.shape)
        #plt.imshow(x_train[0], cmap='gray')
        #plt.show()
        x_train = convert_to8x8(x_train)
        x_test = convert_to8x8(x_test)
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
        """
        if i == 0:
            y_train = np.zeros(y_train.shape)
            y_test = np.zeros(y_test.shape)
        else:
            y_train = np.ones(y_train.shape)
            y_test = np.ones(y_test.shape)
        """
        y_train = np.where(y_train == 7, 0, 1)
        y_test = np.where(y_test == 7, 0, 1)
    """
        x.append(x_train)
        y.append(y_train)
        xt.append(x_test)
        yt.append(y_test)

    x = np.concatenate(x, axis=0)
    y = np.concatenate(y, axis=0)
    xt = np.concatenate(xt, axis=0)
    yt = np.concatenate(yt, axis=0)
    """

    x = x_train
    y = y_train
    xt = x_test
    yt = y_test

    #plt.imshow(x[0], cmap='gray')
    #plt.show()

    x = np.resize(x, (x.shape[0], 8 * 8))
    xt = np.resize(xt, (xt.shape[0], 8 * 8))

    #plt.imshow(np.expand_dims(x[0], axis=0), cmap='gray')
    #plt.show()

    print(x.shape, y.shape, xt.shape, yt.shape)
    return (x, y), (xt, yt)
