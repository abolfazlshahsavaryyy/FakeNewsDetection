from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def predict_svc(model:LinearSVC,x_test,y_test):

    y_pred=model.predict(x_test)

    return accuracy_score(y_pred,y_test)