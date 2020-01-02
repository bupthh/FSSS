import numpy as np
import numpy.matlib
import random as rdm


class BuildDataset(object):

    features_num = 0
    labels_num = 0
    hier_map_d = ''
    hier_file_url = ''
    input_file_url = ''
    b_file_url = ''
    test_file_url = ''

    hier_parent_index = []
    sibling_num_sets = [0]
    position_sets = [0]

    dag_on = False
    DAG_map_d = ''
    DAG_hier_file_url = ''
    DAG_parent_index_set = []
    DAG_parent_num_sets = []

    alpha = 0.5
    beta = 1
    theta = 1
    gamma = 0.1
    epsilon = 1e-5

    missing_on = False
    missing_ratio = 0.2

    noisy_on = True
    noisy_ratio = 0.2

    def __init__(self, attr_list):
        for attr_name, attr_value in attr_list.items():
            setattr(self, attr_name, attr_value)

        if self.dag_on:
            self.build_dag_parent_index_set()
        else:
            self.build_hier_parent_index()

    def build_hier_parent_index(self):
        if type(self.hier_map_d) is str:
            with open(self.hier_map_d) as hf:
                self.hier_map_d = hf.read().splitlines()
                # print(self.hier_map_d)

        hier_parent_map = dict()
        with open(self.hier_file_url, 'r') as hf:
            for line in hf:
                v, k = line.strip('\n').split()
                hier_parent_map[k] = v

        for child in self.hier_map_d:
            self.hier_parent_index.append(self.hier_map_d.index(hier_parent_map[child])
                                          if hier_parent_map[child] in self.hier_map_d
                                          else -1)

    def build_dag_parent_index_set(self):
        if type(self.DAG_map_d) is str:
            with open(self.DAG_map_d) as hf:
                self.hier_map_d = hf.read().splitlines()
                print(self.hier_map_d)

        hier_parent_map = dict()
        with open(self.DAG_hier_file_url, 'r') as hf:
            for line in hf:
                parent, child = line.strip('\n').split()
                hier_parent_map.setdefault(child, []).append(parent)
        print("hier_parent_map:={}".format(hier_parent_map))

        for child in self.hier_map_d:
            parent_set = hier_parent_map.get(child)
            if parent_set is None:
                parent_index_set = []
            else:
                parent_index_set = [index for index, parent in enumerate(self.hier_map_d)
                                    if parent in parent_set]
            self.DAG_parent_index_set.append(parent_index_set)
            self.DAG_parent_num_sets.append(len(parent_index_set))

        print("DAG_parent_index_set={}".format(self.DAG_parent_index_set))

    def build_x_mat_from_file(self, file_url):
        x_array = np.loadtxt(file_url, usecols=range(1, self.features_num + 1), dtype=str)
        x_mat = np.matlib.zeros((x_array.shape[0], self.features_num))
        it = np.nditer(x_array, flags=['multi_index'], order='C')

        while not it.finished:
            feature_index, feature_value = str(it.value).split(':')
            feature_index = int(feature_index) - 1
            feature_value = float(feature_value)
            example_row = it.multi_index[0]
            x_mat[example_row, feature_index] = feature_value
            it.iternext()

        return x_mat

    def build_y_mat_from_file(self, file_url, row, col):
        y_mat = np.zeros((row, col), dtype=int)
        labels_y = np.loadtxt(file_url, usecols=0, dtype=str)
        it = np.nditer(labels_y, order='C')
        row = 0

        while not it.finished:
            for label in str(it.value).split(','):
                y_mat[row][self.hier_map_d.index(label)] = 1
            it.iternext()
            row += 1

        if self.missing_on:
            y_mat = self.build_missing_labels(y_mat, self.missing_ratio)

        if self.noisy_on:
            y_mat = self.build_noisy_labels(y_mat, self.noisy_on)

        return np.mat(y_mat)

    def build_b_mat_from_file(self, file_url):
        embed_mat = np.mat(np.loadtxt(file_url))
        embed_mul_mat = embed_mat * embed_mat.T
        return embed_mul_mat / np.sum(embed_mul_mat)

    def build_s_mat(self, w_mat):
        w_mul_mat = w_mat.T * w_mat
        return w_mul_mat / np.sum(w_mul_mat)

    def build_d_mat(self, w_mat):
        w_mul_mat = w_mat * w_mat.T
        return np.mat(np.diag(np.reciprocal(2*np.power(np.diag(w_mul_mat) + self.epsilon, 0.5))))

    def build_w_parent_mat(self, w_mat):
        split_mat = np.hsplit(w_mat, self.labels_num)
        w_parent_mat = np.matlib.rand((self.features_num, 0))
        for child_index, parent_index in enumerate(self.hier_parent_index):
            parent_index = child_index if parent_index == -1 else parent_index
            w_parent_mat = np.concatenate((w_parent_mat, split_mat[parent_index]), axis=1)
        return w_parent_mat

    def build_p_mat(self, labels_num):
        p_mat = np.matlib.zeros((labels_num, labels_num))
        for child_index, parent_index in enumerate(self.hier_parent_index):
            parent_index = child_index if parent_index == -1 else parent_index
            p_mat[parent_index, child_index] = 1
        return p_mat

    def build_dag_m_f_mat(self, labels_num):
        m_mat = np.matlib.zeros((labels_num, sum(self.DAG_parent_num_sets)))
        f_mat = np.matlib.zeros((labels_num, sum(self.DAG_parent_num_sets)))
        parent_pos = 0
        for child_index, parents in enumerate(self.DAG_parent_index_set):
            for parent_index in parents:
                m_mat[child_index, parent_pos] = 1
                f_mat[parent_index, parent_pos] = 1
                parent_pos += 1
        return m_mat, f_mat

    def build_w_sibling_mat(self, w_mat):
        split_mat = np.hsplit(w_mat, self.labels_num)
        w_sibling_mat = np.matlib.rand((self.features_num, 0))

        for child in range(self.labels_num):
            w_child_sibling_mat = np.matlib.rand((self.features_num, 0))
            child_siblings = [index for index, parent_index in enumerate(self.hier_parent_index)
                              if index != child and parent_index == self.hier_parent_index[child]]
            for sibling_index in child_siblings:
                w_child_sibling_mat = np.concatenate((w_child_sibling_mat, split_mat[sibling_index]), axis=1)
            w_sibling_mat = np.concatenate((w_sibling_mat, w_child_sibling_mat), axis=1)

        return w_sibling_mat

    def build_q_mat(self, labels_num):
        w_child_sibling_set = []
        sibling_nums = 0
        position = 0

        for child in range(labels_num):
            child_siblings = [index for index, parent_index in enumerate(self.hier_parent_index)
                              if index != child and parent_index == self.hier_parent_index[child]]
            w_child_sibling_set.append(child_siblings)
            sibling_nums += len(child_siblings)
            self.sibling_num_sets.append(len(child_siblings))
            position += len(child_siblings)
            self.position_sets.append(position)
        q_mat = np.matlib.zeros((labels_num, sibling_nums))
        sibling_pos = 0

        for child_index, siblings in enumerate(w_child_sibling_set):
            for sibling_index in siblings:
                q_mat[sibling_index, sibling_pos] = 1
                sibling_pos += 1

        return q_mat

    def build_dag_q_mat(self, labels_num):
        w_child_sibling_set = []
        sibling_nums = 0
        position = 0

        for child in range(labels_num):
            parents = BuildDataset.DAG_parent_index_set[child]
            child_all_sibling_nums = 0
            for parent in parents:
                child_siblings = [index for index, parent_index_set in enumerate(self.DAG_parent_index_set)
                                  if index != child and parent in parent_index_set
                                  and index not in parents and child not in parent_index_set]
                w_child_sibling_set.append(child_siblings)
                sibling_nums += len(child_siblings)
                child_all_sibling_nums += len(child_siblings)
            self.sibling_num_sets.append(child_all_sibling_nums)
            position += child_all_sibling_nums
            self.position_sets.append(position)
        q_mat = np.matlib.zeros((labels_num, sibling_nums))
        sibling_pos = 0

        for child_index, siblings in enumerate(w_child_sibling_set):
            for sibling_index in siblings:
                q_mat[sibling_index, sibling_pos] = 1
                sibling_pos += 1

        return q_mat

    def build_h_mat(self):
        h_mat = np.zeros((sum(self.sibling_num_sets), self.labels_num))
        it = np.nditer(h_mat, flags=['multi_index'], op_flags=['readwrite'], order='C')

        while not it.finished:
            i, j = it.multi_index
            if self.position_sets[j] <= i < self.position_sets[j+1]:
                it[0] = 1
            it.iternext()

        return np.mat(h_mat)

    def build_loss_function(self, parameters):
        x_mat = parameters['X']
        w_mat = parameters['W']
        y_mat = parameters['Y']
        s_mat = self.build_s_mat(parameters['W'])
        b_mat = parameters['B']
        p_mat = parameters['P']
        q_mat = parameters['Q']
        h_mat = parameters['H']
        alpha = parameters['alpha']
        beta = parameters['beta']
        theta = parameters['theta']
        gamma = parameters['gamma']

        l1 = np.linalg.norm((x_mat * w_mat - y_mat), ord='fro') ** 2
        l2 = np.linalg.norm((s_mat - b_mat), ord='fro') ** 2
        l3 = np.linalg.norm(np.linalg.norm(w_mat, ord=2, axis=1) ** 0.5, ord=1)
        l4 = np.linalg.norm((w_mat - w_mat * p_mat), ord='fro') ** 2
        l5 = np.trace(w_mat.T * w_mat * q_mat * h_mat)

        return l1 + alpha * l2 + beta * l3 + theta * l4 + gamma * l5

    def build_derivative_w(self, parameters):
        x_mat = parameters['X']
        w_mat = parameters['W']
        y_mat = parameters['Y']
        s_mat = self.build_s_mat(parameters['W'])
        b_mat = parameters['B']
        d_mat = self.build_d_mat(parameters['W'])
        p_mat = parameters['P']
        q_mat = parameters['Q']
        h_mat = parameters['H']
        alpha = parameters['alpha']
        beta = parameters['beta']
        theta = parameters['theta']
        gamma = parameters['gamma']

        ones_mat = np.matlib.ones((self.labels_num, self.labels_num))
        i_mat = np.matlib.identity(self.labels_num)
        df_result = {}

        dl1 = 2 * x_mat.T * (x_mat * w_mat - y_mat)
        dl2 = 4 * alpha * w_mat * (s_mat - b_mat + np.trace((b_mat - s_mat) * s_mat) * ones_mat) / np.sum(w_mat.T * w_mat)
        dl3 = 2 * beta * d_mat * w_mat
        ip_mat = i_mat - p_mat
        dl4 = 2 * theta * w_mat * ip_mat * ip_mat.T
        qh_mat = q_mat * h_mat
        dl5 = gamma * w_mat * (qh_mat + qh_mat.T)

        dw = dl1 + dl2 + dl3 + dl4 + dl5
        df_result['W'] = dw

        return df_result

    def build_parameters(self):
        parameters = dict()
        parameters['X'] = self.build_x_mat_from_file(self.input_file_url)
        parameters['W'] = np.matlib.rand((self.features_num, self.labels_num))
        parameters['Y'] = self.build_y_mat_from_file(self.input_file_url, parameters['X'].shape[0], self.labels_num)
        parameters['B'] = self.build_b_mat_from_file(self.b_file_url)
        parameters['P'] = self.build_p_mat(self.labels_num)
        parameters['Q'] = self.build_q_mat(self.labels_num)
        parameters['H'] = self.build_h_mat()
        parameters['alpha'] = self.alpha
        parameters['beta'] = self.beta
        parameters['theta'] = self.theta
        parameters['gamma'] = self.gamma

        return parameters

    def build_dag_loss_function(self, parameters):
        x_mat = parameters['X']
        w_mat = parameters['W']
        y_mat = parameters['Y']
        s_mat = self.build_s_mat(parameters['W'])
        b_mat = parameters['B']
        m_mat = parameters['M']
        f_mat = parameters['F']
        q_mat = parameters['Q']
        h_mat = parameters['H']
        alpha = parameters['alpha']
        beta = parameters['beta']
        theta = parameters['theta']
        gamma = parameters['gamma']

        l1 = np.linalg.norm((x_mat * w_mat - y_mat), ord='fro') ** 2
        l2 = np.linalg.norm((s_mat - b_mat), ord='fro') ** 2
        l3 = np.linalg.norm(np.linalg.norm(w_mat, ord=2, axis=1) ** 0.5, ord=1)
        l4 = np.linalg.norm((w_mat * (m_mat - f_mat)), ord='fro') ** 2
        l5 = np.trace(w_mat.T * w_mat * q_mat * h_mat)

        return l1 + alpha * l2 + beta * l3 + theta * l4 + gamma * l5

    def build_dag_derivative_w(self, parameters):
        x_mat = parameters['X']
        w_mat = parameters['W']
        y_mat = parameters['Y']
        s_mat = self.build_s_mat(parameters['W'])
        b_mat = parameters['B']
        d_mat = self.build_d_mat(parameters['W'])
        m_mat = parameters['M']
        f_mat = parameters['F']
        q_mat = parameters['Q']
        h_mat = parameters['H']
        alpha = parameters['alpha']
        beta = parameters['beta']
        theta = parameters['theta']
        gamma = parameters['gamma']

        ones_mat = np.matlib.ones((self.labels_num, self.labels_num))
        df_result = {}

        dl1 = 2 * x_mat.T * (x_mat * w_mat - y_mat)
        dl2 = 4 * alpha * w_mat * (s_mat - b_mat + np.trace((b_mat - s_mat) * s_mat) * ones_mat) / np.sum(
            w_mat.T * w_mat)
        dl3 = 2 * beta * d_mat * w_mat
        dl4 = 2 * theta * w_mat * (m_mat - f_mat) * (m_mat - f_mat).T
        qh_mat = q_mat * h_mat
        dl5 = gamma * w_mat * (qh_mat + qh_mat.T)

        dw = dl1 + dl2 + dl3 + dl4 + dl5
        df_result['W'] = dw

        return df_result

    def build_dag_parameters(self):
        parameters = dict()
        parameters['X'] = self.build_x_mat_from_file(self.input_file_url)
        parameters['W'] = np.matlib.rand((self.features_num, self.labels_num))
        parameters['Y'] = self.build_y_mat_from_file(self.input_file_url, parameters['X'].shape[0], self.labels_num)
        parameters['B'] = self.build_b_mat_from_file(self.b_file_url)
        parameters['M'], parameters['F'] = self.build_dag_m_f_mat(self.labels_num)
        parameters['Q'] = self.build_dag_q_mat(self.labels_num)
        parameters['H'] = self.build_h_mat()
        parameters['alpha'] = self.alpha
        parameters['beta'] = self.beta
        parameters['theta'] = self.theta
        parameters['gamma'] = self.gamma

        return parameters

    @staticmethod
    def print_parameters(parameters, name=None):
        with open('d:/parameters', 'w') as pf:
            if name is not None:
                for item in name:
                    pf.write(item + '={}\n'.format(parameters[item]))
            else:
                for k, v in parameters.items():
                    pf.write(k + '={}\n'.format(v))

    @staticmethod
    def arrange_feature_index(mat, rows, cols):
        for i in range(rows):
            for j in range(1, cols):
                mat[i][j] = str(j) + ':' + mat[i][j].split(':')[1]

    @staticmethod
    def construct_file_based_selected_features(examples, input_file_url, features, output_file_url):
        labels_y = np.loadtxt(input_file_url, usecols=0, dtype=str)
        selected_examples = examples[:, features].astype(str)
        it_x = np.nditer(selected_examples, flags=['multi_index'], op_flags=['readwrite'], order='C')

        while not it_x.finished:
            it_x[0] = str(1 + it_x.multi_index[1]) + ':' + str(it_x[0])
            it_x.iternext()

        file_data = np.concatenate((labels_y.reshape(labels_y.shape[0], 1), selected_examples), axis=1)

        with open(output_file_url, 'w') as f:
            np.savetxt(f, file_data, fmt='%s')

    @staticmethod
    def build_noisy_labels(labels, ratio):
        labels_row_num, labels_features_num = labels.shape
        noisy_rows_num = int(labels_row_num * ratio)
        noisy_rows_set = set()

        while len(noisy_rows_set) < noisy_rows_num:
            noisy_rows_set.add(rdm.randint(0, labels_row_num - 1))

        for label_row in noisy_rows_set:
            feature_index = rdm.randint(0, labels_features_num - 1)
            labels[label_row, feature_index] = 1 - labels[label_row, feature_index]

        return labels

    @staticmethod
    def build_missing_labels(labels, ratio):
        labels_row_num, labels_features_num = labels.shape
        missing_rows_num = int(labels_row_num * ratio)
        missing_rows_set = set()

        while len(missing_rows_set) < missing_rows_num:
            missing_rows_set.add(rdm.randint(0, labels_row_num - 1))

        for label_row in missing_rows_set:
            candidate_features = np.where(labels[label_row].flatten() == 1)[0]
            feature_index = rdm.randint(0, len(candidate_features) - 1)
            labels[label_row, candidate_features[feature_index]] = 0

        return labels


if __name__ == '__main__':
    attr_dict = dict()
    attr_dict.update({
        'features_num': 80,
        'labels_num': 29,
        # 'hier_map_d': '',
        # 'hier_file_url': 'd:/imclef07d.hf',
        # 'input_file_url': 'd:/imclef07d_train',
        # 'b_file_url': 'd:/imclef07d_b',
        # 'test_file_url': 'd:/imclef07d_test',
        'alpha': 100,
        'beta': 1,
        'theta': 1,
        'gamma': 0.1,
        'epsilon': 1e-5,
        'missing_on': True,
        'missing_ratio': 0.2
    })

    bd = BuildDataset(attr_dict)
    # paras = bd.build_parameters()
    # bd.print_parameters(paras, name=['Q'])
