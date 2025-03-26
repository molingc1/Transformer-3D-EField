import numpy as np
from sklearn.preprocessing import PowerTransformer

class PiecewiseScaler:
    def __init__(self, threshold=None, low_transformer=None):
        """
        threshold: split value (default: 90th percentile in fit)
        low_transformer: transformer for values < threshold (default: Yeo-Johnson)
        """
        self.threshold = threshold
        self.low_transformer = low_transformer if low_transformer is not None else PowerTransformer(method='yeo-johnson')

    def fit(self, X):
        """
        Fit transformer on 1D array X
        """
        if self.threshold is None:
            self.threshold = np.percentile(X, 90)
            # print("Threshold (90th percentile):", self.threshold)

        low_mask = X < self.threshold
        high_mask = ~low_mask

        if np.sum(low_mask) > 0:
            self.low_transformer.fit(X[low_mask].reshape(-1, 1))
            low_trans = self.low_transformer.transform(X[low_mask].reshape(-1, 1)).flatten()
            self.low_trans_max = low_trans.max()
            # print("Low transformed max:", self.low_trans_max)
        else:
            self.low_trans_max = 0.0

        self.max_val = X[high_mask].max() if np.sum(high_mask) > 0 else self.threshold
        return self

    def transform(self, X):
        """
        Piecewise transform:
        - low values: Yeo-Johnson
        - high values: linear mapping
        """
        X_trans = np.empty_like(X, dtype=float)
        low_mask = X < self.threshold
        high_mask = ~low_mask

        if np.sum(low_mask) > 0:
            X_low = X[low_mask].reshape(-1, 1)
            X_trans[low_mask] = self.low_transformer.transform(X_low).flatten()
            # print("Low region range: [{:.4f}, {:.4f}]".format(X_trans[low_mask].min(), X_trans[low_mask].max()))
        else:
            # print("No low values.")
            pass

        if np.sum(high_mask) > 0:
            X_high = X[high_mask]
            denom = (self.max_val - self.threshold) if (self.max_val != self.threshold) else 1e-8
            X_trans[high_mask] = self.low_trans_max + (X_high - self.threshold) / denom
            # print("High region range: [{:.4f}, {:.4f}]".format(X_trans[high_mask].min(), X_trans[high_mask].max()))
        else:
            # print("No high values.")
            pass

        return X_trans

    def inverse_transform(self, X_trans):
        """
        Inverse transform based on fitted stats
        """
        X_inv = np.empty_like(X_trans, dtype=float)
        low_mask = X_trans < self.low_trans_max
        high_mask = ~low_mask

        if np.sum(low_mask) > 0:
            X_inv[low_mask] = self.low_transformer.inverse_transform(X_trans[low_mask].reshape(-1, 1)).flatten()

        if np.sum(high_mask) > 0:
            denom = (self.max_val - self.threshold) if (self.max_val != self.threshold) else 1e-8
            X_inv[high_mask] = self.threshold + (X_trans[high_mask] - self.low_trans_max) * denom

        return X_inv
