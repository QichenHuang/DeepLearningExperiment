import pickle
import numpy as np
import os

class AdaBoostClassifier:
    '''A simple AdaBoost Classifier.'''

    def __init__(self, weak_classifier, n_weakers_limit):
        '''Initialize AdaBoostClassifier

        Args:
            weak_classifier: The class of weak classifier, which is recommend to be sklearn.tree.DecisionTreeClassifier.
            n_weakers_limit: The maximum number of weak classifier the model can use.
        '''
        self.weak_classifier = weak_classifier
        self.n_weakers_limit = n_weakers_limit
        self.classifier_list = []
        self.a = []

    def is_good_enough(self):
        '''Optional'''
        pass

    def fit(self,X,y):
        '''Build a boosted classifier from the training set (X, y).

        Args:
            X: An ndarray indicating the samples to be trained, which shape should be (n_samples,n_features).
            y: An ndarray indicating the ground-truth labels correspond to X, which shape should be (n_samples,1).
        '''
        # 读取模型参数
        model_file = 'AdaBoost_Model.pkl'
        if os.path.exists(model_file):
            self.classifier_list,self.a = self.load(model_file)
            return
        # 样本权重初始化
        n_samples,_ = X.shape
        w = np.array([1/n_samples]*n_samples)
        for _ in range(self.n_weakers_limit):
            # 基分类器创建及训练
            classifier = self.weak_classifier(max_leaf_nodes=3)
            self.classifier_list.append(classifier)
            classifier.fit(X,y,w)
            # 准确率
            accuracy = classifier.score(X,y)
            # 分类器权重
            at = 0.5 * np.log(accuracy/(1-accuracy))
            self.a.append(at)
            # 更新样本权重
            predict_y = classifier.predict(X)
            w = w*np.exp(-at*y*predict_y)
            w = w/np.sum(w)
        #保存模型参数
        self.save((self.classifier_list,self.a),model_file)

    def predict_scores(self, X):
        '''Calculate the weighted sum score of the whole base classifiers for given samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).

        Returns:
            An one-dimension ndarray indicating the scores of differnt samples, which shape should be (n_samples,1).
        '''
        ensemble_predict = [self.a[i]*self.classifier_list[i].predict(X) for i in range(self.n_weakers_limit)]
        ensemble_predict = np.array(ensemble_predict)
        result = np.apply_along_axis(np.sum,0,ensemble_predict)
        return result

    def predict(self, X, threshold=0):
        '''Predict the catagories for geven samples.

        Args:
            X: An ndarray indicating the samples to be predicted, which shape should be (n_samples,n_features).
            threshold: The demarcation number of deviding the samples into two parts.

        Returns:
            An ndarray consists of predicted labels, which shape should be (n_samples,1).
        '''
        predict_score = self.predict_scores(X)
        predict_score[predict_score < threshold] = -1
        predict_score[predict_score >= threshold] = 1 
        return predict_score
    def score(self,X,y):
        predict_y = self.predict(X)
        accuracy = y*predict_y
        accuracy[accuracy==-1] = 0
        return accuracy.mean()
        
    @staticmethod
    def save(model, filename):
        with open(filename, "wb") as f:
            pickle.dump(model, f)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)
