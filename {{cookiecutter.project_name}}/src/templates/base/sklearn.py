from abc import abstractmethod

from sklearn.model_selection import GridSearchCV

import utils as u


class SklearnModel:
    def __init__(self, model, model_type, name, default_params={}):
        u.set_random_state()

        self.model = model
        self.type = model_type
        self.name = name
        self.default_params = default_params

    def train(self, train_npz, scores_npz, params_file):

        X, y, featnames = u.read_data(train_npz, scores_npz)
        param_grid = u.read_parameters(params_file, self.type, self.name)
        param_grid.update(self.default_params)

        is_grid = any([len(v) > 1 for _, v in param_grid.items()])

        if is_grid:
            self.clf = GridSearchCV(self.model(), param_grid, scoring="roc_auc")
            self.clf.fit(X, y)
            self.best_hyperparams = {k: v for k, v in param_grid.items()}

        else:
            self.best_hyperparams = {k: v[0] for k, v in param_grid.items()}
            self.clf = self.model(**self.best_hyperparams)
            self.clf.best_estimator_ = self.clf

            self.clf.fit(X, y)

        scores = self.score_features()
        scores = u.sanitize_vector(scores)
        selected = self.select_features(scores)

        u.save_scores_npz(featnames, selected, scores, self.best_hyperparams)
        u.save_scores_tsv(featnames, selected, scores, self.best_hyperparams)

    def predict_proba(self, test_npz, scores_npz):

        X_test, _, _ = u.read_data(test_npz, scores_npz)

        y_proba = self.clf.predict_proba(X_test)
        u.save_proba_npz(y_proba, self.best_hyperparams)

    def predict(self, test_npz, scores_npz):

        X_test, _, _ = u.read_data(test_npz, scores_npz)

        y_pred = self.clf.predict(X_test)
        u.save_preds_npz(y_pred, self.best_hyperparams)

    @abstractmethod
    def score_features(self):
        raise NotImplementedError

    @abstractmethod
    def select_features(self, scores):
        raise NotImplementedError
