import numpy as np

class Neuron(object):
    def __init__(self, name, variable_dim, hidden_dim, acti_function):
        self.name = name
        self.acti_function = acti_function
        self.model_param, self.param_grad = self.initialize(variable_dim, hidden_dim)
        self.middle_result = {}
        self.function_dict = self.build_function_dict()

    def build_function_dict(self):

        function_dict = {}

        def sigmoid(input):
            '''
            Compute the sigmoid function for the input here.

            :param input: A scalar or numpy array.
            :return output: A scalar or numpy array sigmoid(input)
            '''
            output = 1.0 / (1.0 + np.exp(- input))

            return output

        def sigmoid_grad(input):
            output = input - input**2
            return output

        def softmax(input):
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
                input = input * denomi.reshape(-1, 1)
            else:
                input_max = np.max(input)
                input = input - input_max
                numerator = np.exp(input)
                denomi = 1.0 / np.sum(numerator)
                input = numerator.dot(denomi)

            assert input.shape == orig_shape

            return input

        def softmax_grad(input):
            one_sample_jocabian = lambda row: np.diag(row) - np.matmul(row.reshape(-1,1),row.reshape(1,-1))
            output = np.apply_along_axis(func1d=one_sample_jocabian, axis=1, arr=input)
            return output

        def relu(input):
            row, colum = np.where(input < 0, axis=0)
            return output

        function_dict['sigmoid'] = sigmoid
        function_dict['sigmoid_grad'] = sigmoid_grad
        function_dict['softmax'] = softmax
        function_dict['softmax_grad'] = softmax_grad

        return function_dict

    def forward(self,vari, dropout_percent):
        self.middle_result['vari_'+self.name] = vari

        Z = np.matmul(vari, self.model_param['W_'+self.name]) + self.model_param['b_'+self.name]
        self.middle_result['Z_'+self.name] = Z

        hidden_state = self.function_dict[self.acti_function](Z)
        if self.acti_function != 'softmax':
            dropout_mask = np.random.binomial([np.ones(hidden_state[0].shape)], 1 - dropout_percent)[0] * \
                           (1.0 / (1 - dropout_percent))
            hidden_state *= dropout_mask

        self.middle_result['hidden_state_' + self.name] = hidden_state
        return hidden_state

    def backward(self,grade_by_before):
        if self.acti_function == 'softmax':
            grade_for_Z = np.tensordot(grade_by_before.reshape(grade_by_before.shape[0],1,-1),
                                       self.function_dict[self.acti_function + '_grad'](self.middle_result['hidden_state_'+self.name]),
                                       axes=([2],[2]))[range(grade_by_before.shape[0]),0,range(grade_by_before.shape[0]),:]
        else:
            grade_for_Z = grade_by_before * self.function_dict[self.acti_function + '_grad'](self.middle_result['hidden_state_'+self.name])
        self.param_grad['W_grad_'+self.name] = np.matmul(self.middle_result['vari_'+self.name].T, grade_for_Z) / grade_for_Z.shape[0]
        self.param_grad['b_grad_' + self.name] = np.sum(grade_for_Z,axis=0) / grade_for_Z.shape[0]
        grade_for_vari = np.matmul(grade_for_Z, self.model_param['W_'+self.name].T)
        return grade_for_vari

    def update_grad(self,step_size):
        self.model_param['W_'+self.name] = self.model_param['W_'+self.name] - step_size*self.param_grad['W_grad_'+self.name]
        self.model_param['b_' + self.name] = self.model_param['b_' + self.name] - step_size * self.param_grad['b_grad_' + self.name]

    def initialize(self,variable_dim, hidden_dim):

        model_param = {
            'W_'+self.name: np.random.randn(variable_dim, hidden_dim),
            'b_'+self.name: np.random.randn(1,hidden_dim),
        }

        param_grad = {
            'W_grad_'+self.name: np.zeros(model_param['W_'+self.name].shape),
            'b_grad_'+self.name: np.zeros(model_param['b_'+self.name].shape),
        }

        return model_param, param_grad