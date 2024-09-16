import numpy as np
import pandas as pd

def sigmoid(x, L, x0, k, b):
    return L / (1 + np.exp(-k * (x - x0))) + b

def sigmoid_rev(x, L, x0, k=1, b=0):
    return L / (1 + np.exp(k * (x - x0))) + b

def sample_range_u(x, L, x0, k, b, x0_rev, k_rev, b_rev, beta=1, beta_rev=1):
    """
    Calculates normal range based utility per sample
    
    :param x: Input value
    :param L: Maximum value of the sigmoid curve
    :param x0: Midpoint of the sigmoid curve
    :param k: Steepness of the curve
    :param b: Y-axis shift
    :param x0_rev: Midpoint of the reverse sigmoid curve
    :param k_rev: Steepness of the reverse curve
    :param b_rev: Y-axis shift for the reverse curve
    :param beta: Weight for the sigmoid curve
    :param beta_rev: Weight for the reverse sigmoid curve
    :return: Calculated utility value
    """
    return sigmoid(x, L, x0, k, b) * beta + sigmoid_rev(x, L, x0_rev, k_rev, b_rev) * beta_rev

def weights_range_u(sig, L, x0, k, b, x0_rev, k_rev, b_rev, beta=1, beta_rev=1):
    """
    Calculates  normal range based weights
    """
    return np.array([sample_range_u(sig[i], L=L, x0=x0, k=k, b=0, x0_rev=x0_rev, k_rev=k_rev, b_rev=b_rev) for i in range(len(sig))])

def generate_sliding_window(s, k):
    return [s[i - k:i] for i in range(k, len(s) + 1)] if len(s) > k else []

def slope_fit(xs, s):
    try:
        return np.polyfit(xs, s, deg=1)[0]
    except Exception as e:
        print(f"Error in slope_fit: {e}")
        print(f"xs: {xs}")
        print(f"s: {s}")
    return None

class URangeCost:
    def __init__(self, L, x0_low, x0_high, k_low=1, b_low=0, k_high=1, b_high=0):
        """
        Initialize utility cost behavior
        Initiates params which control high and low sigmoid shapes according to the formula:
        L / (1 + np.exp(-k*(x-x0))) + b
        """
        self.L = L
        self.x0_low = x0_low
        self.k_low = k_low
        self.b_low = b_low
        self.x0_high = x0_high
        self.k_high = k_high
        self.b_high = b_high

    def low_loss(self, y_true, y_pred, weights=None):
        weights = np.ones(len(y_true)) if weights is None else weights
        return np.power((sigmoid_rev(y_true, self.L, self.x0_low, self.k_low, self.b_low) -
                         sigmoid_rev(y_pred, self.L, self.x0_low, self.k_low, self.b_low)) * weights, 2)

    def high_loss(self, y_true, y_pred, weights=None):
        weights = np.ones(len(y_true)) if weights is None else weights
        return np.power((sigmoid(y_true, self.L, self.x0_high, self.k_high, self.b_high) -
                         sigmoid(y_pred, self.L, self.x0_high, self.k_high, self.b_high)) * weights, 2)

    def range_u_loss(self, y_true, y_pred, weights=None, raw=False):
        weights = np.ones(len(y_true)) if weights is None else weights
        loss = self.high_loss(y_true, y_pred, weights) + self.low_loss(y_true, y_pred, weights)
        return (np.mean(loss), loss) if raw else np.mean(loss)

    def low_loss_truth(self, y_true, weights=None):
        weights = np.ones(len(y_true)) if weights is None else weights
        return np.power(sigmoid_rev(y_true, self.L, self.x0_low, self.k_low, self.b_low) * weights, 2)

    def high_loss_truth(self, y_true, weights=None):
        weights = np.ones(len(y_true)) if weights is None else weights
        return np.power(sigmoid(y_true, self.L, self.x0_high, self.k_high, self.b_high) * weights, 2)

    def range_u_loss_truth(self, y_true, y_pred, weights=None, raw=False):
        return self.range_u_loss(y_true, y_pred, weights, raw)

class TrendCost:
    def __init__(self, high=1, low=1, k=4):
        self.high = high
        self.low = low
        self.k = k

    def low_loss_trend(self, delta):
        return self.low * np.maximum(delta, 0) ** 2

    def high_loss_trend(self, delta):
        return self.high * np.minimum(delta, 0) ** 2

    def trend_loss_sig(self, y_true, y_pred, weights, debug_mode=False):
        trend_true = self.calc_trend(y_true, k=self.k)
        trend_pred = self.calc_trend(y_true, y_pred)
        delta = (trend_true - trend_pred) * weights[self.k - 1:]
        loss = self.low_loss_trend(delta) + self.high_loss_trend(delta)
        return (loss, trend_true, trend_pred) if debug_mode else loss

    def trend_loss(self, y, pred, k=None, ids=None, weights=None, raw=False, debug_mode=False):
        self.k = k or 4
        ids = np.zeros(len(y)) if ids is None else ids
        weights = np.ones(len(y)) if weights is None else weights

        df = pd.DataFrame({'ids': ids, 'y_true': y, 'y_pred': pred, 'weights': weights})
        grp = df.groupby('ids')

        trends = []
        trends_true = []
        trends_pred = []

        for _, group in grp:
            y_true = group['y_true'].values
            y_pred = group['y_pred'].values
            w = group['weights'].values
            if debug_mode:
                l, trend_true, trend_pred = self.trend_loss_sig(y_true, y_pred, w, debug_mode=True)
                trends.append(l)
                trends_true.append(trend_true)
                trends_pred.append(trend_pred)
            else:
                trends.append(self.trend_loss_sig(y_true, y_pred, w))

        trends_array = np.hstack(trends)
        rmse = np.sqrt(np.mean(trends_array**2))
        
        if raw:
            if debug_mode:
                return rmse, trends_array**2, trends_true, trends_pred
            return rmse, trends_array**2
        return rmse

    def trend_dev_loss(self, y, pred, k=None, weights=None, ids=None, raw=False):
        self.k = k or 3
        ids = np.zeros(len(y)) if ids is None else ids
        weights = np.ones(len(y)) if weights is None else weights

        df = pd.DataFrame({'ids': ids, 'y_true': y, 'y_pred': pred, 'weights': weights})
        grp = df.groupby('ids')

        trends = []
        for _, group in grp:
            y_true = group['y_true'].values
            y_pred = group['y_pred'].values
            w = group['weights'].values

            trend_true = self.calc_trend(y_true, k=self.k)
            trend_expected = self.calc_trend(y_true, k=self.k - 1)[:-1]

            trend_weights = (trend_true - trend_expected)**2
            errors = ((y_true - y_pred) ** 2)[self.k - 1:] * w[self.k - 1:]

            trends.append(trend_weights * errors)

        trends_array = np.hstack(trends)
        rmse = np.sqrt(np.mean(trends_array))
        
        return (rmse, trends_array) if raw else rmse

    def calc_trend(self, y_true, y_pred=None, k=3):
        k = k or self.k
        y_true = y_true.values if isinstance(y_true, pd.Series) else y_true

        if y_pred is None:
            chunks = generate_sliding_window(y_true, k=k)
            return np.array([slope_fit(np.arange(len(chunk)), chunk) for chunk in chunks])

        if self.k <= 3:
            print("Warning: k is too small")
            return None

        chunks_y = generate_sliding_window(y_true, k=k-3)
        chunks_pred = generate_sliding_window(y_pred[k-3:], k=k)
        unified = [np.append(chunks_y[i], chunks_pred[i]) for i in range(len(chunks_pred))]

        return np.array([slope_fit(np.arange(len(chunk)), chunk) for chunk in unified])

