from sklearn import multiclass, svm


def sklearn_multiclass_prediction(mode, X_train, y_train, X_test):
    '''
    Use Scikit Learn built-in functions multiclass.OneVsRestClassifier
    and multiclass.OneVsOneClassifier to perform multiclass classification.

    Arguments:
        mode: one of 'ovr', 'ovo' or 'crammer'.
        X_train, X_test: numpy ndarray of training and test features.
        y_train: labels of training data, from 0 to 9.

    Returns:
        y_pred_train, y_pred_test: a tuple of 2 numpy ndarrays,
                                   being your prediction of labels on
                                   training and test data, from 0 to 9.
    '''
    if mode == 'ovr':
        classifier = multiclass.OneVsRestClassifier(svm.LinearSVC(
        random_state=12345))
        classifier.fit(X_train,y_train)
        y_pred_train = classifier.predict(X_train)
        y_pred_test = classifier.predict(X_test)
    elif mode == 'ovo':
        classifier = multiclass.OneVsOneClassifier(svm.LinearSVC(
        random_state=12345))
        classifier.fit(X_train,y_train)
        y_pred_train = classifier.predict(X_train)
        y_pred_test = classifier.predict(X_test)
    else:
        classifier = svm.LinearSVC(multi_class='crammer_singer',
        random_state=12345)
        classifier.fit(X_train,y_train)
        y_pred_train = classifier.predict(X_train)
        y_pred_test = classifier.predict(X_test)
    return y_pred_train, y_pred_test
