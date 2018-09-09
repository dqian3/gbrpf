#!/usr/bin/python
'''
An experiment in applying Gradient Descent to the problem of finding a good
coeffecients for our linear combination. Couldn't get it to work.
'''

import sys
import time
try:
    # For python2
    from itertools import izip as zip
    LARGE_NUMBER = sys.maxint
except ImportError:
    # For python3
    LARGE_NUMBER = sys.maxsize

import numpy as np
from gradient import accel_grad_desc
from sklearn.preprocessing import normalize
np.random.seed(0)

class Dataset(object):
    def __init__(self, X, y):
        for i in range(X.shape[1]):
            min_feat = 9999999
            max_feat = -9999999
            for j in range(X.shape[0]):
                if X[j][i] > max_feat:
                    max_feat = X[j][i]
                if X[j][i] < min_feat:
                    min_feat = X[j][i]
            for j in range(X.shape[0]):
                X[j][i] -= min_feat
                X[j][i] /= (max_feat - min_feat)
        print(X)
        self.X = X
        self.y = y


class TreeNode(object):
    def __init__(self):
        self.is_leaf = False
        self.left_child = None
        self.right_child = None
        self.split_coef_vector = None # Defines the linear combination for our projection tree.
        self.split_val = None
        self.weight = None

    def _calc_split_gain(self, G, H, G_l, H_l, G_r, H_r, lambd):
        """
        Loss reduction
        (Refer to Eq7 of Reference[1])
        """
        def calc_term(g, h):
            return np.square(g) / (h + lambd)
        return calc_term(G_l, H_l) + calc_term(G_r, H_r) - calc_term(G, H)

    def _calc_leaf_weight(self, grad, hessian, lambd):
        """
        Calculate the optimal weight of this leaf node.
        (Refer to Eq5 of Reference[1])
        """
        return np.sum(grad) / (np.sum(hessian) + lambd)

    def _calc_linear_comb(self, coef_vector, instance):
        return np.dot(coef_vector, instance)

    def build(self, instances, grad, hessian, shrinkage_rate, depth, param):
        """
        Exact Greedy Alogirithm for Split Finidng
        (Refer to Algorithm1 of Reference[1])
        """
        assert instances.shape[0] == len(grad) == len(hessian)
        if depth > param['max_depth']:
            self.is_leaf = True
            self.weight = self._calc_leaf_weight(grad, hessian, param['lambda']) * shrinkage_rate
            return
        G = np.sum(grad)
        H = np.sum(hessian)
        best_gain = 0.
        best_feature_id = None
        best_coef_vector = None
        is_diagonal = False

        best_val = 0.
        best_left_instance_ids = None
        best_right_instance_ids = None

        # Do one-hot vectors (one feature each, same as normal gbt)
        for feature_id in range(instances.shape[1]):
            coef_vector = np.full(instances.shape[1], 0)    # One-hot
            coef_vector[feature_id] = 1

            # Stores results of lin comb of every row
#            dot_products = np.array([self._calc_linear_comb(coef_vector, instance) for instance in instances])
#             (negative_best_gain, sorted_instance_ids, j) = f(coef_vector, instances)
            (curr, current_coef_vector, negative_current_gain) = accel_grad_desc(1, coef_vector, 0.00001, self.f, instances, grad, hessian, shrinkage_rate, depth, param)
            current_gain = -negative_current_gain
            if current_gain > best_gain:
                best_gain = current_gain
                best_coef_vector = current_coef_vector

        (negative_best_gain, sorted_instance_ids, j, dot_products) = self.evaluate(best_coef_vector, instances, grad, hessian, shrinkage_rate, depth, param)
        assert -negative_best_gain == best_gain
        best_val = dot_products[sorted_instance_ids[j]]
        best_left_instance_ids = sorted_instance_ids[:j+1]
        best_right_instance_ids = sorted_instance_ids[j+1:]
                
#            G_l, H_l = 0., 0.
#            #sorted_instance_ids = instances[:,feature_id].argsort()
#            sorted_instance_ids = dot_products.argsort()
#            for j in range(sorted_instance_ids.shape[0]):
#                G_l += grad[sorted_instance_ids[j]]
#                H_l += hessian[sorted_instance_ids[j]]
#                G_r = G - G_l
#                H_r = H - H_l
#                current_gain = self._calc_split_gain(G, H, G_l, H_l, G_r, H_r, param['lambda'])
#                if current_gain > best_gain:
#                    best_gain = current_gain
#                    #best_feature_id = feature_id
#                    best_coef_vector = coef_vector
#                    best_val = dot_products[sorted_instance_ids[j]]
#                    best_left_instance_ids = sorted_instance_ids[:j+1]
#                    best_right_instance_ids = sorted_instance_ids[j+1:]

        # =======================================THIS IS OUR STUFF=========================================
        # Try random linear combinations instead
        for i in range(0):#instances.shape[1]):
            coef_vector = np.random.rand(instances.shape[1])
            #coef_vector = np.array([round(x) for x in coef_vector])
            dot_products = np.array([self._calc_linear_comb(coef_vector, instance) for instance in instances])
            G_l, H_l = 0., 0.
            #sorted_instance_ids = instances[:,feature_id].argsort()
            sorted_instance_ids = dot_products.argsort()
            for j in range(sorted_instance_ids.shape[0]):

                G_l += grad[sorted_instance_ids[j]]
                H_l += hessian[sorted_instance_ids[j]]
                G_r = G - G_l
                H_r = H - H_l
                current_gain = self._calc_split_gain(G, H, G_l, H_l, G_r, H_r, param['lambda'])
                if current_gain > best_gain:
                    best_gain = current_gain
                    #best_feature_id = feature_id
                    best_coef_vector = coef_vector
                    best_val = dot_products[sorted_instance_ids[j]]
                    best_left_instance_ids = sorted_instance_ids[:j+1]
                    best_right_instance_ids = sorted_instance_ids[j+1:]

        # =======================================THIS IS OUR STUFF=========================================
        
        if best_gain < param['min_split_gain']:
            self.is_leaf = True
            self.weight = self._calc_leaf_weight(grad, hessian, param['lambda']) * shrinkage_rate
        else:
            self.split_coef_vector = best_coef_vector
            self.split_val = best_val

            self.left_child = TreeNode()
            self.left_child.build(instances[best_left_instance_ids],
                                  grad[best_left_instance_ids],
                                  hessian[best_left_instance_ids],
                                  shrinkage_rate,
                                  depth+1, param)

            self.right_child = TreeNode()
            self.right_child.build(instances[best_right_instance_ids],
                                   grad[best_right_instance_ids],
                                   hessian[best_right_instance_ids],
                                   shrinkage_rate,
                                   depth+1, param)

    def f(self, coef_vector, instances, grad, hessian, shrinkage_rate, depth, param):
        return self.evaluate(coef_vector, instances, grad, hessian, shrinkage_rate, depth, param)[0]

    def evaluate(self, coef_vector, instances, grad, hessian, shrinkage_rate, depth, param):
      best_gain = 0.
      best_feature_id = None
      is_diagonal = False

      G = np.sum(grad)
      H = np.sum(hessian)

      best_val = 0.
      best_j = -1
#      best_left_instance_ids = None
#      best_right_instance_ids = None

      dot_products = np.array([self._calc_linear_comb(coef_vector, instance) for instance in instances])
      G_l, H_l = 0., 0.
      sorted_instance_ids = dot_products.argsort()
      for j in range(sorted_instance_ids.shape[0]):

        G_l += grad[sorted_instance_ids[j]]
        H_l += hessian[sorted_instance_ids[j]]
        G_r = G - G_l
        H_r = H - H_l
        current_gain = self._calc_split_gain(G, H, G_l, H_l, G_r, H_r, param['lambda'])
        if current_gain > best_gain:
          best_gain = current_gain
          #best_feature_id = feature_id
          best_val = dot_products[sorted_instance_ids[j]]
          best_j = j
#          best_left_instance_ids = sorted_instance_ids[:j+1]
#          best_right_instance_ids = sorted_instance_ids[j+1:]
      return (-best_gain, sorted_instance_ids, best_j, dot_products)

    def predict(self, x):
        if self.is_leaf:
            return self.weight
        else:
            if self._calc_linear_comb(self.split_coef_vector, x) <= self.split_val:
                return self.left_child.predict(x)
            else:
                return self.right_child.predict(x)


class Tree(object):
    ''' Classification and regression tree for tree ensemble '''
    def __init__(self):
        self.root = None

    def build(self, instances, grad, hessian, shrinkage_rate, param):
        assert len(instances) == len(grad) == len(hessian)
        self.root = TreeNode()
        current_depth = 0
        self.root.build(instances, grad, hessian, shrinkage_rate, current_depth, param)

    def predict(self, x):
        return self.root.predict(x)


class GBT(object):
    def __init__(self):
        self.params = {'gamma': 0.,
                       'lambda': 1.,
                       'min_split_gain': 0.1,
                       'max_depth': 5,
                       'learning_rate': 0.3,
                       }
        self.best_iteration = None

    def _calc_training_data_scores(self, train_set, models):
        if len(models) == 0:
            return None
        X = train_set.X
        scores = np.zeros(len(X))
        for i in range(len(X)):
            scores[i] = self.predict(X[i], models=models)
        return scores

    def _calc_l2_gradient(self, train_set, scores):
        labels = train_set.y
        hessian = np.full(len(labels), 2)
        if scores is None:
            grad = np.random.uniform(size=len(labels))
        else:
            grad = np.array([2 * (labels[i] - scores[i]) for i in range(len(labels))])
        return grad, hessian

    def _calc_gradient(self, train_set, scores):
        """For now, only L2 loss is supported"""
        return self._calc_l2_gradient(train_set, scores)

    def _calc_l2_loss(self, models, data_set):
        errors = []
        for x, y in zip(data_set.X, data_set.y):
            errors.append(y - self.predict(x, models))
        return np.mean(np.square(errors))

    def _calc_loss(self, models, data_set):
        """For now, only L2 loss is supported"""
        return self._calc_l2_loss(models, data_set)

    def _build_learner(self, train_set, grad, hessian, shrinkage_rate):
        learner = Tree()
        learner.build(train_set.X, grad, hessian, shrinkage_rate, self.params)
        return learner

    def train(self, params, train_set, num_boost_round=20, valid_set=None, early_stopping_rounds=5):
        self.params.update(params)
        models = []
        shrinkage_rate = 1.
        best_iteration = None
        best_val_loss = LARGE_NUMBER
        train_start_time = time.time()

        print("Training until validation scores don't improve for {} rounds."
              .format(early_stopping_rounds))
        for iter_cnt in range(num_boost_round):
            iter_start_time = time.time()
            scores = self._calc_training_data_scores(train_set, models)
            grad, hessian = self._calc_gradient(train_set, scores)
            learner = self._build_learner(train_set, grad, hessian, shrinkage_rate)
            if iter_cnt > 0:
                shrinkage_rate *= self.params['learning_rate']
            models.append(learner)
            train_loss = self._calc_loss(models, train_set)
            val_loss = self._calc_loss(models, valid_set) if valid_set else None
            val_loss_str = '{:.10f}'.format(val_loss) if val_loss else '-'
            print("Iter {:>3}, Train's L2: {:.10f}, Valid's L2: {}, Elapsed: {:.2f} secs"
                  .format(iter_cnt, train_loss, val_loss_str, time.time() - iter_start_time))
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_iteration = iter_cnt
            if iter_cnt - best_iteration >= early_stopping_rounds:
                print("Early stopping, best iteration is:")
                print("Iter {:>3}, Train's L2: {:.10f}".format(best_iteration, best_val_loss))
                break

        self.models = models
        self.best_iteration = best_iteration
        print("Training finished. Elapsed: {:.2f} secs".format(time.time() - train_start_time))

    def predict(self, x, models=None, num_iteration=None):
        if models is None:
            models = self.models
        assert models is not None
        return np.sum(m.predict(x) for m in models[:num_iteration])