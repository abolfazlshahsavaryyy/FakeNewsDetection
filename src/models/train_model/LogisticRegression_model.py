from sklearn.linear_model import LogisticRegression

def LogisticRegression_model(X_train,y_train,hyperparameter):
    model=LogisticRegression(hyperparameter)
    model.fit(X_train,y_train)

    return model