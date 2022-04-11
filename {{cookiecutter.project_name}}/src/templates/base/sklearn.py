from abc import abstractmethod

from sklearn.model_selection import GridSearchCV

import utils as u


class SklearnModel:
    def __init__(self, model, model_type, name, params):
        u.set_random_state()

        self.model = model
        self.type = model_type
        self.name = name

        params = u.read_json_parameters(params)
        self.fixed_params = params["fixed_parameters"]
        self.cv_params = params["cv_parameters"]

    def train(self, train_npz, scores_npz):

        X, y, featnames = u.read_data(train_npz, scores_npz)

        model = self.model(**self.fixed_params)
        if self.cv_params:
            self.clf = GridSearchCV(model, self.cv_params, scoring="roc_auc")
            self.clf.fit(X, y)
            self.best_hyperparams = {
                k: self.clf.best_params_[k] for k in self.cv_params.keys()
            }
        else:
            self.clf = model
            self.clf.fit(X, y)
            self.best_hyperparams = {}

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
