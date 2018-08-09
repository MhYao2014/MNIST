import numpy as np

from data_process import load_train_labels
from data_process import load_train_images


class FNN_hyperparam():
    def __init__(self):
        pass

    model_hyperparam = {
        'vari_dim': 784,
        'layer1_dim': 128,
        'layer2_dim': 64,
        'layer3_dim': 30,
        'layer4_dim': 10,
        'batch_size': 512,
    }

class FNN(FNN_hyperparam):
    def __init__(self):
        self.model_param, self.middle_results = self.initialize()
        pass

    def initialize(self):
        '''
        This function will initialize all the parameters (matrix and bias vectors) of forward neural network with random number.

        :return: None
        '''
        W_input2layer1 = np.random.randn(784,
                                         self.model_hyperparam['layer1_dim'])
        b_input2layer1 = np.random.randn(1,self.model_hyperparam['layer1_dim'])

        W_layer12layer2 = np.random.randn(self.model_hyperparam['layer1_dim'],
                                          self.model_hyperparam['layer2_dim'])
        b_layer12layer2 = np.random.randn(1,self.model_hyperparam['layer2_dim'])

        W_layer22layer3 = np.random.randn(self.model_hyperparam['layer2_dim'],
                                          self.model_hyperparam['layer3_dim'])
        b_layer22layer3 = np.random.randn(1,self.model_hyperparam['layer3_dim'])

        W_layer32layer4 = np.random.randn(self.model_hyperparam['layer3_dim'],
                                          self.model_hyperparam['layer4_dim'])
        b_layer32layer4 = np.random.randn(1,self.model_hyperparam['layer4_dim'])

        return ([W_input2layer1, b_input2layer1, W_layer12layer2, b_layer12layer2, W_layer22layer3, b_layer22layer3, W_layer32layer4, b_layer32layer4],{})

    def batcher(self, images_vectors, labels_numpy):
        '''
        This method will randomly take "batch_size" samples out of the whole training data (images_batch, labels_batch) pair.

        :param images_vectors: A numpy array with shape (60000, 784)
        :param labels_numpy: A numpy array with shape (60000, )
        :return images_batch: A numpy array with shape (batch_size, 784)
        :return labels_batch: A numpy array with shape (batch_size, )
        '''
        index = np.random.choice(50000, self.model_hyperparam['batch_size'])
        images_batch = images_vectors[index]
        labels_batch = labels_numpy[index]

        return images_batch, labels_batch

    def sigmoid(self, input):
        '''
        Compute the sigmoid function for the input here.

        :param input: A scalar or numpy array.
        :return output: A scalar or numpy array sigmoid(input)
        '''
        output = 1.0 / (1.0 + np.exp(- input))

        return output

    def softmax(self,input):
        """
        Compute the softmax function for each row of the input x.

        :param input: A D dimensional vector or N X D dimensional numpy matrix.
        :return input: Softmax(input)
        """
        orig_shape = input.shape
        if len(input.shape) > 1:
            minus_max_row = lambda a: a - np.max(a)
            input = np.apply_along_axis(minus_max_row, 1, input)
            input = np.exp(input)
            denomi_row = lambda a: 1.0 / np.sum(a)
            denomi = np.apply_along_axis(denomi_row, 1, input)
            input = input * denomi.reshape(-1,1)
        else:
            input_max = np.max(input)
            input = input - input_max
            numerator = np.exp(input)
            denomi = 1.0 / np.sum(numerator)
            input = numerator.dot(denomi)

        assert input.shape == orig_shape

        return input

    def forward(self,images_batch, labels_batch=None, if_train=False):
        '''
        This method will calculate the forward process; if "if_train" is True,
        it will return more than just the predicted catagory, but also the loss and the middle result.

        :return images_batch: A numpy array with shape (batch_size, 784)
        :return labels_batch: A numpy array with shape (batch_size, )
        :param if_train: When "True", this method will return (pred_catagory, loss, middle_result);
                         Otherwise, this method will return (pred_catagory, loss=None, middle_result=None)
        :return: (pred_category, loss=None, middle_result=None)
        '''
        layer1_before = np.matmul(images_batch, self.model_param[0]) + self.model_param[1]
        self.middle_results['layer1_before'] = layer1_before
        layer1 = self.sigmoid(layer1_before)
        self.middle_results['layer1']=layer1

        layer2_before = np.matmul(layer1, self.model_param[2]) + self.model_param[3]
        self.middle_results['layer2_before'] = layer2_before
        layer2 = self.sigmoid(layer2_before)
        self.middle_results['layer2']=layer2

        layer3_before = np.matmul(layer2, self.model_param[4]) + self.model_param[5]
        self.middle_results['layer3_before'] = layer3_before
        layer3 = self.sigmoid(layer3_before)
        self.middle_results['layer3'] = layer3

        layer4_before = np.matmul(layer3, self.model_param[6]) + self.model_param[7]
        self.middle_results['layer4_before'] = layer4_before
        layer4 = self.softmax(layer4_before)
        self.middle_results['layer4']=layer4
        if if_train:
            logit_pred = -np.log(layer4)
            loss = np.choose(labels_batch.astype(np.int),logit_pred.T).sum() / labels_batch.shape[0]
            self.middle_results['loss'] = np.choose(labels_batch.astype(np.int),logit_pred.T)

            pred_category = np.argmax(layer4,axis=1)
            right = np.where((pred_category - labels_batch.astype(np.int)) == 0)[0].shape[0]
            wrong_percent =100 - (right / labels_batch.shape[0] * 100)
        else:
            loss = 0
            pred_category = np.argmax(layer4, axis=1)
            wrong_percent = 0


        return (pred_category, wrong_percent)

    def update_param(self, model_param):
        '''
        This method will take the new model parameters and update them into fnn.model_param

        :param model_param:
        :return:
        '''
        self.model_param = model_param

        return


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

    print('\nThe shape of all data images are:', images_numpy.shape)
    print('\nThe shape of all data labels are:', labels_numpy.shape)

    images_vectors = images_numpy.reshape((60000, -1))

    fnn = FNN()
    images_batch, labels_batch = fnn.batcher(images_vectors=images_vectors,
                                             labels_numpy=labels_numpy)
    pred_cate, loss = fnn.forward(images_batch, labels_batch, if_train=True)

    print('\nThe predicted category are:\n', pred_cate)
    print('\nThe grand-truth category are:\n', labels_batch.astype(np.int))
    print('\n The average loss is:', loss)
    print('\n',fnn.middle_results['loss'].shape)


























































