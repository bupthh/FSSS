import numpy as np
import datetime

from build_dataset import BuildDataset


class FeatureSelection(object):

    alpham = 1
    rho = 0.1
    t = 2

    max_iter = 2000
    min_error = 1e-6

    selected_feature_ratio = 0.2
    selected_feature_file_url = ''
    selected_feature_output_file_url = ''

    def __init__(self, attr_list):
        for attr_name, attr_value in attr_list.items():
            setattr(self, attr_name, attr_value)

    @staticmethod
    def armijo_search(f, df, x, alpham, rho, t, parameters):
        flag = 0
        x_value = parameters[x]
        a = 0
        b = alpham
        fk = f(parameters)
        gk_mat = df(parameters)[x]
        gk = gk_mat.getA().flatten()
        phi0 = fk
        dphi0 = np.dot(gk, -gk)
        alpha = b * np.random.uniform(0, 1)

        while flag == 0:
            parameters[x] = x_value + alpha * (-gk_mat)
            new_fk = f(parameters)
            phi = new_fk
            if phi - phi0 <= rho * alpha * dphi0:
                if phi - phi0 >= (1 - rho) * alpha * dphi0:
                    flag = 1
                else:
                    a = alpha
                    b = b
                    if b < alpham:
                        alpha = (a + b) / 2
                    else:
                        alpha = t * alpha
            else:
                a = a
                b = alpha
                alpha = (a + b) / 2

        parameters[x] = x_value

        return alpha

    def steepest_gradient_descent(self, f, df, parameters):
        i = 0
        f_old = f(parameters)

        while True:
            grad_values = df(parameters)
            grad_w = grad_values['W']
            alpha_w = FeatureSelection.armijo_search(f, df, 'W', self.alpham, self.rho, self.t, parameters)
            parameters['W'] += alpha_w * (-grad_w)

            i = i + 1
            f_new = f(parameters)
            error = abs(f_new - f_old) / abs(f_old)
            print("i={} f_new={} f_old={} error={}".format(i, f_new, f_old, error))

            if error <= self.min_error or i >= self.max_iter:
                break
            else:
                f_old = f_new

    @staticmethod
    def get_selected_features_index(feature_mat):
        w_square_by_row = np.linalg.norm(feature_mat, ord=2, axis=1)
        return np.argsort(-w_square_by_row)

    def fsss(self, parameters, data_set):
        if data_set.dag_on:
            loss_fun = data_set.build_dag_loss_function
            derivative_fun = data_set.build_dag_derivative_w
        else:
            loss_fun = data_set.build_loss_function
            derivative_fun = data_set.build_derivative_w

        self.steepest_gradient_descent(loss_fun, derivative_fun, parameters)

        return self.get_selected_features_index(parameters['W'])

    def select_features(self, data_set):
        if data_set.dag_on:
            paras = data_set.build_dag_parameters()
        else:
            paras = data_set.build_parameters()

        old_w_mat = np.linalg.norm(paras['W'], ord='fro') ** 2
        start_time = datetime.datetime.now()
        try:
            f_mat_index = self.fsss(paras, data_set)
        finally:
            end_time = datetime.datetime.now()
            print("runtime={}".format((end_time - start_time).seconds))
            print("old_w={}\nnew_w={}".format(old_w_mat, np.linalg.norm(paras['W'], ord='fro') ** 2))
            with open(self.selected_feature_file_url, 'a') as sf:
                np.savetxt(sf, np.mat(f_mat_index + 1), fmt='%s')

    def build_test_file(self, data_set):
        test_mat = data_set.build_x_mat_from_file(data_set.test_file_url)
        selected_feature_num = int(data_set.features_num * self.selected_feature_ratio)

        with open(self.selected_feature_file_url, 'r') as sf:
            line = sf.readline()
            feature_array = np.fromstring(line, dtype=int, sep=' ')
            selected_features = sorted((feature_array - 1).tolist()[:selected_feature_num])
            BuildDataset.construct_file_based_selected_features(test_mat, data_set.test_file_url, selected_features,
                                                                self.selected_feature_output_file_url)


if __name__ == '__main__':
    bd_attr_dict = dict()
    bd_attr_dict.update({
        'features_num': 478,  # 80    """ imclef07d  seq"""
        'labels_num': 379,  # 29
        'hier_map_d': 'd:/seq_code',  # 'd:/imclef07d.hier_map'
        'hier_file_url': 'd:/seq.hf',  # 'd:/imclef07d.hf'
        'input_file_url': 'd:/seq_train',  # 'd:/imclef07d_train'
        'b_file_url': 'd:/seq_b',  # 'd:/imclef07d_b'
        'test_file_url': 'd:/seq_test',  # 'd:/imclef07d_test'
        'alpha': 100,
        'beta': 1,
        'theta': 1,
        'gamma': 0.1,
        'noisy_on': True,
        'noisy_ratio': 0.1,
        'dag_on': True,
        'DAG_map_d': 'd:/seq_code_dag',
        'DAG_hier_file_url': 'd:/seq_DAG.hf'
    })
    bd = BuildDataset(bd_attr_dict)

    fs_attr_dict = dict()
    fs_attr_dict.update({
        'max_iter': 100,
        'min_error': 1e-6,
        'selected_feature_ratio': 0.3,
        'selected_feature_file_url': 'd:/hyx.txt',
        'selected_feature_output_file_url': 'd:/hhh.txt'

    })
    fs = FeatureSelection(fs_attr_dict)
    fs.select_features(bd)
    fs.build_test_file(bd)
