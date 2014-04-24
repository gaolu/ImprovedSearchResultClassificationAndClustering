class L1LinearSVC(LinearSVC):

    def fit(self, X, y):

        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.

        self.transformer_ = LinearSVC(penalty='l1', dual=False,
                tol=1e-3)
        X = self.transformer_.fit_transform(X, y)
        return LinearSVC.fit(self, X, y)

    def predict(self, X):
        X = self.transformer_.transform(X)
        return LinearSVC.predict(self, X)