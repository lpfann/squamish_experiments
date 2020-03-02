import cvxpy as cvx
import fri.model.ordinal_regression
import numpy as np
from sklearn.utils import check_X_y


class RegularizedLinearOrdinalRegression(fri.model.ordinal_regression.OrdinalRegression_SVM):
    '''
    Ordinal regressor with variable l1 ratio to enable ElasticNet, Lasso and Ridge
    and different score functions
    '''
    def __init__(self, **parameters):
        super().__init__(**parameters)

    @classmethod
    def hyperparameter(cls):
        return ["C","l1_ratio"]

    def fit(self, X, y, **kwargs):
        (n, d) = X.shape
        C = self.hyperparam["C"]
        l1_ratio = self.hyperparam["l1_ratio"]

        self.classes_ = np.unique(y)
        original_bins = sorted(self.classes_)
        n_bins = len(original_bins)
        bins = np.arange(n_bins)
        get_old_bin = dict(zip(bins, original_bins))

        w = cvx.Variable(shape=(d), name="w")
        # For ordinal regression we use two slack variables, we observe the slack in both directions
        slack_left = cvx.Variable(shape=(n), name="slack_left")
        slack_right = cvx.Variable(shape=(n), name="slack_right")
        # We have an offset for every bin boundary
        b_s = cvx.Variable(shape=(n_bins - 1), name="bias")
        norm = l1_ratio * cvx.pnorm(w, 1) + (1 - l1_ratio) * cvx.pnorm(w, 2)**2
        objective = cvx.Minimize(norm + C * cvx.sum(slack_left + slack_right))
        constraints = [
            slack_left >= 0,
            slack_right >= 0
        ]

        # Add constraints for slack into left neighboring bins
        for i in range(n_bins - 1):
            indices = np.where(y == get_old_bin[i])
            constraints.append(X[indices] * w - slack_left[indices] <= b_s[i] - 1)

        # Add constraints for slack into right neighboring bins
        for i in range(1, n_bins):
            indices = np.where(y == get_old_bin[i])
            constraints.append(X[indices] * w + slack_right[indices] >= b_s[i - 1] + 1)

        # Add explicit constraint, that all bins are ascending
        for i in range(n_bins - 2):
            constraints.append(b_s[i] <= b_s[i + 1])

        # Solve problem.
        solver_params = self.solver_params
        problem = cvx.Problem(objective, constraints)
        problem.solve(**solver_params)

        w = w.value
        b_s = b_s.value
        slack_left = np.asarray(slack_left.value).flatten()
        slack_right = np.asarray(slack_right.value).flatten()
        self.coef_ = w
        self.model_state = {
            "w": w,
            "b_s": b_s,
            "slack": (slack_left, slack_right)
        }

        loss = np.sum(slack_left + slack_right)
        w_l1 = np.linalg.norm(w, ord=1)
        self.constraints = {
            "loss": loss,
            "w_l1": w_l1
        }
        return self

    def score(self, X, y, return_error=False, **kwargs):

        X, y = check_X_y(X, y)

        prediction = self.predict(X)
        score = fri.model.ordinal_regression.ordinal_scores(y, prediction, "mze", return_error=return_error)

        return score