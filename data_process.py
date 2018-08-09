import numpy as np
import struct

def decode_idx3_ubyte(idx3_ubyte_file):
    ''' Turn the binary file of training data into numpy array

    :param idx3_ubyte_file: The path to training data of ubyte format
    :return images: The training data of numpy array format
    '''
    bin_data = open(file=idx3_ubyte_file, mode='rb').read()

    offset = 0
    fmt_header = '>IIII'
    magic_number, images_number, num_rows, num_cols = struct.unpack_from(fmt_header, buffer=bin_data, offset=offset)
    print("\nmagic:%d, count:%d, size:%dX%d" % (magic_number, images_number, num_rows, num_cols))

    image_size = num_cols * num_rows
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    images = np.empty(shape=(images_number, num_rows, num_cols))
    for i in range(images_number):
        if ( i + 1 ) % 10000 == 0:
            print ("done %d" % ( i + 1 ) + " pictures")
        images[i] = np.array(struct.unpack_from(fmt_image, buffer=bin_data, offset=offset)).reshape((num_rows, num_cols))
        offset += struct.calcsize(fmt_image)

    return images

def decode_idx1_ubyte(idx1_ubyte_file):
    '''Turn the binary file of training data's label into numpy array

    :param idx1_ubyte_file: The path to training data's label of ubyte format
    :return labels: The training data's label of numpy array
    '''
    bin_data = open(file=idx1_ubyte_file, mode='rb').read()

    offset = 0
    fmt_header = '>ii'
    magic_number, images_number = struct.unpack_from(fmt_header, buffer=bin_data, offset=offset)
    print("\nmagic: %d, count: %d" % (magic_number, images_number))

    offset += struct.calcsize(fmt_header)
    fmt_image = 'B'
    labels = np.empty(images_number)
    for i in range(images_number):
        if ( i + 1 ) % 10000 == 0:
            print ("done %d" % ( i + 1 ) + " labels")
        labels[i] = struct.unpack_from(fmt_image, buffer=bin_data, offset=offset)[0]
        offset += struct.calcsize(fmt_image)

    return labels

def load_train_images(idx3_ubyte_file):
    ''' Load the training data's image from idx3_ubyte_file

    :param idx3_ubyte_file: path to the training data's image
    :return images: The training data of numpy array format
    '''
    return decode_idx3_ubyte(idx3_ubyte_file=idx3_ubyte_file)

def load_train_labels(idx1_ubyte_file):
    ''' Load the training data's labels from idx1_ubyte_file

    :param idx1_ubyte_file: path to training data's label
    :return The training data's label of numpy array
    '''
    return decode_idx1_ubyte(idx1_ubyte_file=idx1_ubyte_file)

if __name__ == '__main__':
    train_images_idx3_ubyte_file = './train-images.idx3-ubyte'
    train_labels_idx1_ubyte_file = './train-labels.idx1-ubyte'
    images = load_train_images(idx3_ubyte_file=train_images_idx3_ubyte_file)
    print (images.shape,'\n')
    labels = load_train_labels(idx1_ubyte_file=train_labels_idx1_ubyte_file)
    print (labels.shape)



