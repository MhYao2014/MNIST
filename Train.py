import datetime
import numpy as np

from data_process import load_train_labels
from data_process import load_train_images
from FNN import FNN


class Train_hyperparam():
    def __int__(self):

        pass

    train_hyperparam = {
        'step_size': 0.3,
        'stop_criterion': 7,
        'max_iteration': 100000000,
    }

class Train(Train_hyperparam):

    def __int__(self):

        pass


    def get_grade(self, middle_result, model_param, labels_batch, images_batch):
        '''
        This method will calculate the grade of all parameters in fnn.

        :param middle_result: A dictionary and each value is a numpy array. The middle result of fnn's forward process.
        :param model_param: A list and each entry of this list is a numpy array. All the parameters in fnn.
        :param labels_batch: A numpy array with shape (batch_size, ).
        :return grade_param: A list and each entry of this list is a numpy array. All the gradient of all parameters in fnn.
        '''
        y_truth = np.zeros(middle_result['layer4'].shape)
        for row, position in enumerate(labels_batch.astype(np.int)):
            y_truth[row, position] = 1
        layer4_before_grad = middle_result['layer4'] - y_truth
        W_layer32layer4_grad = np.matmul(middle_result['layer3'].T, layer4_before_grad) / labels_batch.shape[0]
        b_layer32layer4_grad = np.sum(layer4_before_grad,axis=0) / labels_batch.shape[0]

        layer3_grad = np.matmul(layer4_before_grad, model_param[6].T)
        layer3_before_grad = (layer3_grad * ( middle_result['layer3'] - middle_result['layer3']**2 ))
        W_layer22layer3_grad = np.matmul(middle_result['layer2'].T, layer3_before_grad) / labels_batch.shape[0]
        b_layer22layer3_grad = np.sum(layer3_before_grad,axis=0) / labels_batch.shape[0]

        layer2_grad = np.matmul(layer3_before_grad, model_param[4].T)
        layer2_before_grad = (layer2_grad * (middle_result['layer2'] - middle_result['layer2'] ** 2))
        W_layer12layer2_grad = np.matmul(middle_result['layer1'].T, layer2_before_grad) / labels_batch.shape[0]
        b_layer12layer2_grad = np.sum(layer2_before_grad,axis=0) / labels_batch.shape[0]

        layer1_grad = np.matmul(layer2_before_grad, model_param[2].T)
        layer1_before_grad = (layer1_grad * (middle_result['layer1'] - middle_result['layer1'] ** 2))
        W_input2layer1_grad = np.matmul(images_batch.T, layer1_before_grad) / labels_batch.shape[0]
        b_input2layer1_grad = np.sum(layer1_before_grad,axis=0) / labels_batch.shape[0]


        grade_param = [W_input2layer1_grad, b_input2layer1_grad, W_layer12layer2_grad, b_layer12layer2_grad, W_layer22layer3_grad, b_layer22layer3_grad, W_layer32layer4_grad, b_layer32layer4_grad]
        return grade_param

    def update_grade(self, model_param, grade_param):
        '''
        This method will perform the fixed step size gradient descent

        :param model_param: A list and each entry of this list is a numpy array. All the parameters in fnn.
        :param grade_param: A list and each entry of this list is a numpy array. All the gradient of all parameters in fnn.
        :return: model_param: A list and each entry of this list is a numpy array. All the parameters in fnn.
        '''
        for index, param in enumerate(model_param):
            model_param[index] = model_param[index] - self.train_hyperparam['step_size']*grade_param[index]
        return model_param

    def sgd_training(self, fnn, images_vectors, labels_numpy):
        '''
        This method will organize the training process.

        :param fnn: A instance of FNN class.
        :param images_vectors: A numpy array with shape (60000, 784)
        :param labels_numpy: A numpy array with shape (60000, )
        :return: None
        '''
        loss_emperical = 100
        iteration = 0
        starttime = datetime.datetime.now()
        while loss_emperical > self.train_hyperparam['stop_criterion']:

            images_batch, labels_batch = fnn.batcher(images_vectors, labels_numpy)

            pred_category, loss_batch_average = fnn.forward(images_batch, labels_batch, if_train=True)

            grade_param = self.get_grade(fnn.middle_results, fnn.model_param, labels_batch, images_batch)

            new_model_param = self.update_grade(fnn.model_param, grade_param)

            fnn.update_param(new_model_param)

            loss_emperical = 0.9 * loss_emperical + 0.1 * loss_batch_average

            iteration += 1

            if iteration > self.train_hyperparam['max_iteration']:
                print ("The training process may failed, we have trained for %d iterations.\n" % self.train_hyperparam['max_iteration'])
                break
            # if iteration >= 15000 and iteration<30000:
            #     self.train_hyperparam['step_size'] = 0.1
            # elif iteration >=30000 and iteration < 45000:
            #     self.train_hyperparam['step_size'] = 0.05
            if iteration % (50000//fnn.model_hyperparam['batch_size']) == 0:
                account = 0
                for i in range(10000//fnn.model_hyperparam['batch_size']):
                    _, accuracy = fnn.forward(images_vectors[50000 + i * fnn.model_hyperparam['batch_size']:50000 + (i + 1) * fnn.model_hyperparam['batch_size']],
                                              labels_numpy[50000 + i * fnn.model_hyperparam['batch_size']:50000 + (i + 1) * fnn.model_hyperparam['batch_size']], if_train=True)
                    account += np.nonzero(_ - labels_numpy[50000 + i * fnn.model_hyperparam['batch_size']:50000 + (i + 1) * fnn.model_hyperparam['batch_size']])[0].shape[0]
                print('This is the %dth iterations, and the accuracy on the test data set is: %f%%' % (iteration//(50000//fnn.model_hyperparam['batch_size']), (100 - 100 * (account / 10000))))
                loss_emperical = 100 * (account / 10000)
                if loss_emperical < 8:
                    fnn.model_hyperparam['step_size'] = 0.015
                else:
                    fnn.model_hyperparam['step_size'] = 0.2
                #print("This is the %dth iteration, and the accuracy is: %d%%" % (iteration, 100 - loss_emperical))

        endtime = datetime.datetime.now()
        print("\nThe iterations take about %d seconds\n" % (endtime - starttime).seconds)
        print ('\nThe training process finished !\n')

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
    train = Train()
    train.sgd_training(fnn=fnn ,images_vectors=images_vectors[0:60000] ,labels_numpy=labels_numpy[0:60000])

    Accuracy = 0
    account = 0
    for i in range(20):
        _, accuracy = fnn.forward(images_vectors[50000 + i * 500:50000 + (i + 1) * 500],
                                  labels_numpy[50000 + i * 500:50000 + (i + 1) * 500], if_train=True)
        account += np.nonzero(_ - labels_numpy[50000 + i * 500:50000 + (i + 1) * 500])[0].shape[0]
    print('The accuracy on the whole data set is %f %%:\n' % (100 - 100 * (account / 10000)))






























































