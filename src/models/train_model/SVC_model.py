from sklearn.svm import LinearSVC

def SVC_model(X_train,y_train,hyperparameter:dict):
    model=LinearSVC(**hyperparameter)
    model.fit(X_train,y_train)

    return model