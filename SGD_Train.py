import numpy as np
import datetime

from Train import Train_hyperparam
from NN import NN
from data_process import load_train_labels
from data_process import load_train_images

def one_hot_transformer(labels_numpy):
    y_truth = np.zeros([60000,10])
    for row, position in enumerate(labels_numpy.astype(np.int)):
        y_truth[row, position] = 1
    return y_truth

class SGD_Train(Train_hyperparam):
    def __init__(self):
        pass

    def sgd_train(self, fnn, images_vectors, labels_numpy):
        '''
        This method will organize the training process.

        :param fnn: A instance of FNN class.
        :param images_vectors: A numpy array with shape (60000, 784)
        :param labels_numpy: A numpy array with shape (60000, )
        :return: None
        '''
        accuracy = 0
        Accuracy = 0
        wrong_persent = 100
        iteration = 0
        starttime = datetime.datetime.now()
        while Accuracy < 100 - self.train_hyperparam['stop_criterion']:

            images_batch, labels_batch = fnn.batcher(images_vectors, labels_numpy)
            #pred_category, wrong_persent = fnn.forward(images_batch, labels_batch, if_train=True)
            fnn.forward(images_batch, labels_batch, if_train=True)

            fnn.backward(labels_batch,self.train_hyperparam['step_size'])

            iteration += 1

            if iteration > self.train_hyperparam['max_iteration']:
                print ("The training process may failed, we have trained for %d iterations.\n" % self.train_hyperparam['max_iteration'])
                break

            if iteration % (60000 // fnn.model_hyperparam['batch_size']) == 0:
                fnn.model_hyperparam['dropout_percent'] = 0
                accuracy = self.evaluate(fnn, images_vectors, labels_numpy)
                Accuracy = Accuracy * 0.9 + accuracy * 0.1
                print('This is the %dth iterations, and the accuracy on the test data set is: %f%%' %
                        (iteration // (60000 // fnn.model_hyperparam['batch_size']),Accuracy))
                fnn.model_hyperparam['dropout_percent'] = 0.05
                Accuracy = Accuracy*0.9 + accuracy * 0.1

        endtime = datetime.datetime.now()
        print ("\nThe iterations take about %d seconds\n" % (endtime - starttime).seconds)
        print ('\nThe training process finished !\n')

    def evaluate(self, fnn, images_vectors, labels_numpy):
        account = 0
        for i in range(50000 // fnn.model_hyperparam['batch_size']):
            pred_labels, _ = fnn.forward(images_vectors[i * fnn.model_hyperparam['batch_size']:
                                                     (i + 1) *fnn.model_hyperparam['batch_size']],
                                      labels_numpy[i * fnn.model_hyperparam['batch_size']:
                                                   (i + 1) *fnn.model_hyperparam['batch_size']],
                                      if_train=True)
            account += np.nonzero(pred_labels - np.nonzero(labels_numpy[i * fnn.model_hyperparam['batch_size']:
                                                   (i + 1) *fnn.model_hyperparam['batch_size']])[1])[0].shape[0]
        accuracy = (100 - 100 * (account / 50000))
        return accuracy

if __name__ == '__main__':
    ###################
    # path to the training data's images and labels
    ###################
    train_images_idx3_ubyte_file = './train-images.idx3-ubyte'
    train_labels_idx1_ubyte_file = './train-labels.idx1-ubyte'

    ##################
    # Here we go
    ##################
    images_numpy = load_train_images(idx3_ubyte_file=train_images_idx3_ubyte_file)
    labels_numpy = load_train_labels(idx1_ubyte_file=train_labels_idx1_ubyte_file)

    labels_numpy = one_hot_transformer(labels_numpy)

    print('\nThe shape of all data images are:', images_numpy.shape)
    print('\nThe shape of all data labels are:', labels_numpy.shape)

    images_vectors = images_numpy.reshape((60000, -1)) / 255
    fnn = NN()
    train = SGD_Train()
    train.train_hyperparam['stop_criterion'] = 2
    fnn.model_hyperparam['batch_size'] = 128
    fnn.model_hyperparam['layer1_dim'] = 134
    fnn.model_hyperparam['layer2_dim'] = 34
    fnn.model_hyperparam['layer4_dim'] = 10
    fnn.model_hyperparam['dropout_percent'] = 0.05
    print('\nThe hyperparameters of this fully connected neuron network are:\n',fnn.model_hyperparam)
    print('The hyperparameters of training process are:\n',train.train_hyperparam)
    train.sgd_train(fnn=fnn ,images_vectors=images_vectors[0:60000] ,labels_numpy=labels_numpy[0:60000])

    Accuracy = 0
    account = 0
    fnn.model_hyperparam['dropout_percent'] = 0
    for i in range(10000 // fnn.model_hyperparam['batch_size']):
        pred_labels, _ = fnn.forward(images_vectors[50000 + i * fnn.model_hyperparam['batch_size']:50000 + (i + 1) * fnn.model_hyperparam['batch_size']],
                                     labels_numpy[50000 + i * fnn.model_hyperparam['batch_size']:50000 + (i + 1) * fnn.model_hyperparam['batch_size']], if_train=True)
        account += np.nonzero(pred_labels - np.nonzero(labels_numpy[50000 + i * fnn.model_hyperparam['batch_size']:
                                                                    50000 + (i + 1) * fnn.model_hyperparam['batch_size']])[1])[0].shape[0]
    print('The accuracy on the whole data set is %f %%:\n' % (100 - 100 * (account / 10000)))