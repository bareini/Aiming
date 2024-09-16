import numpy as np
import torch

def sigmoid(x, L, x0, k, b):
    return L / (1 + torch.exp(-k * (x - x0))) + b

def sigmoid_rev(x, L, x0, k=1, b=0):
    return L / (1 + torch.exp(k * (x - x0))) + b

def sample_range_u(x, L, x0, k, b, x0_rev, k_rev, b_rev, beta=1, beta_rev=1):
    """
    Calculates clinical normal range based utility per sample

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
    Calculates clinical normal range based weights
    """
    return torch.tensor([sample_range_u(sig[i], L=L, x0=x0, k=k, b=0, x0_rev=x0_rev, k_rev=k_rev, b_rev=b_rev) for i in range(len(sig))])

def generate_sliding_window(s, k):
    return [s[:, i - k:i] for i in range(k, s.shape[1] + 1)] if s.shape[1] > k else []

def slope_fit_polyfit(xs, s, device=None):
    try:
        if torch.cuda.is_available():
            return np.polyfit(xs, s.detach().cpu().numpy(), deg=1)[0]
        else:
            return np.polyfit(xs, s.detach().numpy(), deg=1)[0]
    except Exception as e:
        print(f"Error in slope_fit_polyfit: {e}")
        print(f"xs: {xs}")
        print(f"s: {s}")
    return None

def slope_fit(X, Y):
    """
    Analytical calculation of slope
    :param X: Independent variable
    :param Y: Dependent variable
    :return: Naive slope
    """
    try:
        X = X.float()
        return (torch.mean(X * Y, dim=1) - torch.mean(X, dim=1) * torch.mean(Y, dim=1)) / \
               (torch.mean(X * X, dim=1) - torch.mean(X, dim=1) * torch.mean(X, dim=1)).float()
    except Exception as e:
        print(f"Error in slope_fit: {e}")
        print(f"X: {X}")
        print(f"Y: {Y}")
    return None

class URangeLoss:
    def __init__(self, L, x0_low, x0_high, k_low=1, b_low=0, k_high=1, b_high=0, device=None):
        """
        Initialize utility loss behavior
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
        self.device = device

    def low_loss(self, y_true, y_pred, weights=None):
        weights = torch.ones(len(y_true), device=self.device) if weights is None else weights
        return torch.pow((sigmoid_rev(y_true, self.L, self.x0_low, self.k_low, self.b_low) -
                          sigmoid_rev(y_pred, self.L, self.x0_low, self.k_low, self.b_low)) * weights, 2)

    def high_loss(self, y_true, y_pred, weights=None):
        weights = torch.ones(y_true.shape, device=self.device) if weights is None else weights
        return torch.pow((sigmoid(y_true, self.L, self.x0_high, self.k_high, self.b_high) -
                          sigmoid(y_pred, self.L, self.x0_high, self.k_high, self.b_high)) * weights, 2)

    def range_u_loss(self, y_true, y_pred, weights=None, raw=False):
        weights = torch.ones(y_true.shape, device=self.device) if weights is None else weights
        loss = self.high_loss(y_true, y_pred, weights) + self.low_loss(y_true, y_pred, weights)
        return (torch.mean(loss), loss) if raw else torch.mean(loss)

    def low_loss_truth(self, y_true, weights=None):
        weights = torch.ones(y_true.shape, device=self.device) if weights is None else weights
        return torch.pow(sigmoid_rev(y_true, self.L, self.x0_low, self.k_low, self.b_low) * weights, 2)

    def high_loss_truth(self, y_true, weights=None):
        weights = torch.ones(y_true.shape, device=self.device) if weights is None else weights
        return torch.pow(sigmoid(y_true, self.L, self.x0_high, self.k_high, self.b_high) * weights, 2)

    def range_u_loss_truth(self, y_true, y_pred, weights=None, raw=False):
        return self.range_u_loss(y_true, y_pred, weights, raw)

class TrendLoss:
    def __init__(self, high=1, low=1, k=4, device=None):
        self.high = high
        self.low = low
        self.k = k
        self.device = device
        self.true_trend_dict = {}

    def low_loss_trend(self, delta):
        zeros = torch.zeros(delta.shape, device=self.device)
        delta = torch.maximum(delta, zeros)
        return self.low * delta ** 2

    def high_loss_trend(self, delta):
        zeros = torch.zeros(delta.shape, device=self.device)
        delta = torch.minimum(delta, zeros)
        return self.high * delta ** 2

    def trend_loss_sig(self, y_true, y_pred, weights, true_trend=None, debug_mode=False):
        trend_true = self.calc_trend(y_true, k=self.k) if true_trend is None else true_trend
        trend_pred = self.calc_trend(y_true, y_pred)
        delta = (trend_true - trend_pred) * weights[:, self.k-1:]

        loss = self.low_loss_trend(delta) + self.high_loss_trend(delta)
        return (loss, trend_true, trend_pred) if debug_mode else loss

    def trend_loss(self, y_true, y_pred, ids=None, weights=None, raw=False, true_trend=None, debug_mode=False):
        self.k = 4
        weights = torch.ones(y_true.shape, device=self.device) if weights is None else weights

        if debug_mode:
            loss, trend_true, trend_pred = self.trend_loss_sig(y_true, y_pred, weights, debug_mode=True, true_trend=true_trend)
            if raw:
                return torch.sqrt(torch.mean(loss**2)), loss**2, trend_true, trend_pred
            return torch.sqrt(torch.mean(loss**2))
        else:
            loss = self.trend_loss_sig(y_true, y_pred, weights, true_trend=true_trend)
            if raw:
                return torch.sqrt(torch.mean(loss**2)), loss**2
            return torch.sqrt(torch.mean(loss**2))

    def trend_dev_loss(self, y_true, y_pred, weights=None, trend_true=None, trend_expected=None, raw=False, k=3):
        weights = torch.ones(y_true.shape, device=self.device) if weights is None else weights

        trend_true = self.calc_trend(y_true, k=self.k) if trend_true is None else trend_true
        trend_expected = self.calc_trend(y_true, k=self.k - 1)[:, :-1] if trend_expected is None else trend_expected

        trend_weights = (trend_true - trend_expected)**2
        errors = ((y_true - y_pred) ** 2)[:, self.k-1:] * weights[:, self.k-1:]

        loss = trend_weights * errors

        if raw:
            return torch.mean(loss), loss
        return torch.mean(loss)

    def calc_trend(self, y_true, y_pred=None, k=3):
        k = self.k if k is None else k

        if y_pred is None:
            chunks = generate_sliding_window(y_true, k=k)
            trends = []
            for chunk in chunks:
                x_range = torch.arange(chunk.shape[1], device=self.device).repeat(y_true.shape[0], 1)
                slopes = slope_fit(x_range, chunk)
                trends.append(slopes)
        else:
            if self.k <= 3:
                print("Warning: k is too small")
                return None
            chunks_y = generate_sliding_window(y_true, k=k-3)
            chunks_pred = generate_sliding_window(y_pred[:, k-3:], k=k)
            unified = [torch.cat((chunks_y[i], chunks_pred[i]), dim=1) for i in range(len(chunks_pred))]

            trends = []
            for chunk in unified:
                x_range = torch.arange(chunk.shape[1], device=self.device).repeat(y_true.shape[0], 1)
                trends.append(slope_fit(x_range, chunk))

        return torch.vstack(trends).T
