import numpy as np
from FNN import FNN
from Neuron import Neuron

class NN(FNN):
    def __init__(self):
        self.net = self.build_net()

    def build_net(self):
        h_input = Neuron(name='h_input',
                         variable_dim=self.model_hyperparam['vari_dim'],
                         hidden_dim=self.model_hyperparam['layer1_dim'],
                         acti_function='sigmoid')
        h_1 = Neuron(name='h_1',
                     variable_dim=self.model_hyperparam['layer1_dim'],
                     hidden_dim=self.model_hyperparam['layer2_dim'],
                     acti_function='sigmoid')
        h_output = Neuron(name='h_output',
                          variable_dim=self.model_hyperparam['layer2_dim'],
                          hidden_dim=self.model_hyperparam['layer4_dim'],
                          acti_function='softmax')

        net = {
            'h_input': h_input,
            'h_1': h_1,
            'h_output': h_output,
        }

        return net

    def forward(self, images_batch, labels_batch, if_train=False):
        h_input_state = self.net['h_input'].forward(vari=images_batch, dropout_percent=self.model_hyperparam['dropout_percent'])
        h_1_state = self.net['h_1'].forward(vari=h_input_state, dropout_percent=self.model_hyperparam['dropout_percent'])
        h_output_state = self.net['h_output'].forward(vari=h_1_state, dropout_percent=self.model_hyperparam['dropout_percent'])
        #print(h_output_state)

        if if_train:
            # logit_pred = -np.log(h_output_state)
            # loss = np.choose(labels_batch.astype(np.int),logit_pred.T).sum() / labels_batch.shape[0]

            pred_category = np.argmax(h_output_state,axis=1)
            temp = np.nonzero(labels_batch)[1]
            wrong_count = np.nonzero(pred_category - np.nonzero(labels_batch)[1])[0].shape[0]
            wrong_percent = wrong_count / labels_batch.shape[0] * 100
        else:
            # loss = 0
            pred_category = np.argmax(h_output_state, axis=1)
            wrong_percent = 0

        return pred_category, wrong_percent

    def backward(self, labels_batch,step_size):
        # print (self.net['h_output'].middle_result)
        y_truth = labels_batch
        grade_for_h_output = -y_truth*(1/self.net['h_output'].middle_result['hidden_state_h_output'])
        grade_for_h_1 = self.net['h_output'].backward(grade_by_before=grade_for_h_output)
        grade_for_h_input = self.net['h_1'].backward(grade_by_before=grade_for_h_1)
        grade_for_vari = self.net['h_input'].backward(grade_by_before=grade_for_h_input)

        self.net['h_output'].update_grad(step_size=step_size)
        self.net['h_1'].update_grad(step_size=step_size)
        self.net['h_input'].update_grad(step_size=step_size)

if __name__ == '__main__':
    h_input = Neuron(name='h_input', variable_dim=784, hidden_dim=30, acti_function='softmax')
    print(h_input)
    fnn = NN()