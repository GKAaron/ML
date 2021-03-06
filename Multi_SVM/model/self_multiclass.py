import numpy as np
from sklearn import svm


class MulticlassSVM:

    def __init__(self, mode):
        if mode != 'ovr' and mode != 'ovo' and mode != 'crammer-singer':
            raise ValueError('mode must be ovr or ovo or crammer-singer')
        self.mode = mode

    def fit(self, X, y):
        if self.mode == 'ovr':
            self.fit_ovr(X, y)
        elif self.mode == 'ovo':
            self.fit_ovo(X, y)
        elif self.mode == 'crammer-singer':
            self.fit_cs(X, y)

    def fit_ovr(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovr_student(X, y)

    def fit_ovo(self, X, y):
        self.labels = np.unique(y)
        self.binary_svm = self.bsvm_ovo_student(X, y)

    def fit_cs(self, X, y):
        self.labels = np.unique(y)
        X_intercept = np.hstack([X, np.ones((len(X), 1))])

        N, d = X_intercept.shape
        K = len(self.labels)

        W = np.zeros((K, d))

        n_iter = 1500
        learning_rate = 1e-8
        for i in range(n_iter):
            W -= learning_rate * self.grad_student(W, X_intercept, y)

        self.W = W

    def predict(self, X):
        if self.mode == 'ovr':
            return self.predict_ovr(X)
        elif self.mode == 'ovo':
            return self.predict_ovo(X)
        else:
            return self.predict_cs(X)

    def predict_ovr(self, X):
        scores = self.scores_ovr_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_ovo(self, X):
        scores = self.scores_ovo_student(X)
        return self.labels[np.argmax(scores, axis=1)]

    def predict_cs(self, X):
        X_intercept = np.hstack([X, np.ones((len(X), 1))])
        return np.argmax(self.W.dot(X_intercept.T), axis=0)

    def bsvm_ovr_student(self, X, y):
        '''
        Train OVR binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with labels as keys,
                        and binary SVM models as values.
        '''
        binary_svm = {}
        for i in self.labels:
           bin_class = svm.LinearSVC(random_state=12345)
           y_label = np.zeros(y.shape)
           y_label[y==i] = 1
           bin_class.fit(X,y_label)
           binary_svm[i] = bin_class
        return binary_svm

    def bsvm_ovo_student(self, X, y):
        '''
        Train OVO binary classfiers.

        Arguments:
            X, y: training features and labels.

        Returns:
            binary_svm: a dictionary with label pairs as keys,
                        and binary SVM models as values.
        '''
        binary_svm = {}
        i = 0
        while i < self.labels.shape[0]:
            j = i+1
            l1 = self.labels[i]
            while j < self.labels.shape[0]:
                l2 = self.labels[j]
                t_label = (y==l1)|(y==l2)
                t_y = y[t_label]
                t_X = X[t_label]
                t_y[t_y==l2] = -1
                t_y[t_y==l1] = 1
                t_y[t_y==-1] = 0
                bin_class = svm.LinearSVC(random_state=12345)
                bin_class.fit(t_X,t_y)
                binary_svm[(l1,l2)] = bin_class
                j += 1
            i+=1
        return binary_svm

    def scores_ovr_student(self, X):
        '''
        Compute class scores for OVR.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        scores = []
        for i in self.binary_svm.keys():
            score = self.binary_svm[i].decision_function(X)
            scores.append(score)
        return np.array(scores).T

    def scores_ovo_student(self, X):
        '''
        Compute class scores for OVO.

        Arguments:
            X: Features to predict.

        Returns:
            scores: a numpy ndarray with scores.
        '''
        scores = np.zeros((X.shape[0],self.labels.shape[0]))
        for l1,l2 in self.binary_svm.keys():
            score = self.binary_svm[(l1,l2)].predict(X)
            scores[:,l1] = scores[:,l1] + score
            score[score==1] = -1
            score[score==0] = 1
            score[score==-1] = 0
            scores[:,l2] = scores[:,l2] + score
        return scores
            
    def loss_student(self, W, X, y, C=1.0):
        '''
        Compute loss function given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The value of loss function given W, X and y.
        '''
        f = X.dot(W.T)
        theta = np.ones(f.shape)
        theta[[i for i in range(X.shape[0])],y] = 0
        f = np.argmax(f+theta,axis=1)
        w = []
        j = 0
        for i in f:
            if i == y[j]:
                w.append(np.zeros(X[i].shape))
            else:
                w.append(W[i]-W[y[j]])
            j += 1
        w = np.array(w)
        task_l = y - f
        task_l[task_l!=0]=1
        loss = 0.5*np.sum(W*W) + C*np.sum(np.sum(X*w,axis=1)+task_l)
        return loss
        
    def grad_student(self, W, X, y, C=1.0):
        '''
        Compute gradient function w.r.t. W given W, X, y.

        For exact definitions, please check the MP document.

        Arugments:
            W: Weights. Numpy array of shape (K, d)
            X: Features. Numpy array of shape (N, d)
            y: Labels. Numpy array of shape N
            C: Penalty constant. Will always be 1 in the MP.

        Returns:
            The graident of loss function w.r.t. W,
            in a numpy array of shape (K, d).
        '''
        f = X.dot(W.T)
        theta = np.ones(f.shape)
        theta[[i for i in range(X.shape[0])],y] = 0
        f = np.argmax(f+theta,axis=1)
        w = np.zeros(W.shape)
        j = 0
        for i in f:
            if i != y[j]:
                w[i] = w[i] + X[j]
                w[y[j]] = w[y[j]] - X[j]
            j += 1
        w_g = W + C*w
        return w_g
