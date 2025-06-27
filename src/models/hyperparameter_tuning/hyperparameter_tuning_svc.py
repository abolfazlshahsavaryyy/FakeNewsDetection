from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

def manual_grid_search_linear(x_train, y_train, x_valid, y_valid, param_grid):
    best_model = None
    best_score = 0
    best_params = {}

    for C in param_grid['C']:
        for max_iter in param_grid.get('max_iter', [1000]):
            model = LinearSVC(C=C, max_iter=max_iter)
            model.fit(x_train, y_train)
            y_pred = model.predict(x_valid)
            score = accuracy_score(y_valid, y_pred)

            if score > best_score:
                best_score = score
                best_model = model
                best_params = {'C': C, 'max_iter': max_iter}

    return best_model, best_params, best_score
