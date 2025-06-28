from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
def predict_logistic_regression(model:LogisticRegression,x_test,y_test):
    y_pred=model.predict(x_test)
    return accuracy_score(y_pred,y_test)
    