import numpy as np
import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
import xgboost as xgb
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import metrics_u

# Constants
BASE_SEQ_LEN = 496
PREF_PARAMS_TRAIN = ["rmse", "rmse_onset1", "rmse_onset2", "rmse_trend_events",
                     "rmse_normal_range", "u_range", "u_trend", "u_div"]
PREF_PARAMS_TEST = ["rmse", "rmse_onset1", "rmse_onset2", "rmse_trend_events",
                    "rmse_normal_range", "u_range", "u_trend", "u_div", "u_raw",
                    "u_trend_raw", "u_div_raw", "y", "pred", "trend_y_test", "div_y_test",
                    "trend_pred_test", "div_pred_test"]
def prepare_for_trend(arr, k):
    new_arr = arr.reshape(-1, BASE_SEQ_LEN)
    return new_arr[:, k:].flatten()

def load_dataset(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def initialize_models():
    return {
        "LinearRegression": LinearRegression(),
        "Ridge": RidgeCV(alphas=[1e-2, 1e-1, 1, 3, 5]),
        "Lasso": LassoCV(alphas=[1e-2, 1e-1, 1, 3, 5]),
        "XGBoost Reg": xgb.XGBRegressor(n_estimators=40, max_depth=3),
        "XGBoost Reg 2": xgb.XGBRegressor(n_estimators=30, max_depth=6),
        "GB": GradientBoostingRegressor(loss="quantile", alpha=0.5)
    }

def initialize_performance_dict(model_names):
    return {model_name: {
        "train": {metric: [] for metric in PREF_PARAMS_TRAIN},
        "test": {metric: [] for metric in PREF_PARAMS_TEST}
    } for model_name in model_names}

def calculate_normal_range(y_train):
    mean = y_train.mean()
    std = y_train.std()
    high_th = mean + std
    low_th = mean - std
    return mean, high_th, low_th

def evaluate_model(model, X_train, y_train, X_test, y_test,
                   onset1_train, onset1_test, onset2_train,
                   onset2_test, trend_events_train, trend_events_test,
                   normal_range_train, normal_range_test,
                   train_ids, test_ids):

    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    results = {}
    for name, y_true, y_pred, onset1, onset2, trend_events, normal_range, ids in [
        ("train", y_train, pred_train, onset1_train, onset2_train, trend_events_train, normal_range_train, train_ids),
        ("test", y_test, pred_test, onset1_test, onset2_test, trend_events_test, normal_range_test, test_ids)
    ]:
        results[name] = {
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "rmse_onset1": np.sqrt(mean_squared_error(y_true[onset1 > 0], y_pred[onset1 > 0])),
            "rmse_onset2": np.sqrt(mean_squared_error(y_true[onset2 > 0], y_pred[onset2 > 0])),
            "rmse_trend_events": np.sqrt(mean_squared_error(y_true[trend_events > trend_events.std()], y_pred[trend_events > trend_events.std()])),
            "rmse_normal_range": np.sqrt(mean_squared_error(y_true[normal_range], y_pred[normal_range]))
        }

        u_range = metrics_u.URangeCost(L=10, x0_high=y_train.mean(), k_high=0.08,
                                        b_high=0, x0_low=y_train.mean(), k_low=0.1, b_low=0)
        u_trend = metrics_u.TrendCost()

        results[name]["u_range"], results[name]["u_raw"] = u_range.range_u_loss(y_true, y_pred, raw=True)
        results[name]["u_trend"], results[name]["u_trend_raw"] = u_trend.trend_loss(y_true, y_pred, raw=True, ids=ids)
        results[name]["u_div"], results[name]["u_div_raw"] = u_trend.trend_dev_loss(y_true, y_pred, raw=True, ids=ids)

    return results, pred_train, pred_test

def main():
    datasets = load_dataset('syn_base_dataset.pkl')
    models = initialize_models()
    sim_perf_dict = initialize_performance_dict(models.keys())

    for model_name, model in models.items():
        for i in range(5):

            # get data and events ground truth heat maps for the evaluation
            X_train, y_train = datasets["trains"][i], datasets["train_targets"][i]
            X_test, y_test = datasets["tests"][i], datasets["test_targets"][i]
            onset1_train, onset1_test = datasets["onset1s_train"][i], datasets["onset1s_test"][i]
            onset2_train, onset2_test = datasets["onset2s_train"][i], datasets["onset2s_test"][i]
            trend_events_train, trend_events_test = datasets["trend_events_train"][i], datasets["trend_events_test"][i]
            train_ids, test_ids = datasets["train_ids"][i], datasets["test_ids"][i]

            mean, high_th, low_th = calculate_normal_range(y_train)
            normal_range_train = (y_train > high_th) | (y_train < low_th)
            normal_range_test = (y_test > high_th) | (y_test < low_th)

            results, pred_train, pred_test = evaluate_model(model, X_train, y_train, X_test, y_test, onset1_train, onset1_test, onset2_train, onset2_test, trend_events_train, trend_events_test, normal_range_train, normal_range_test, train_ids, test_ids)

            for phase in ["train", "test"]:
                for metric, value in results[phase].items():
                    sim_perf_dict[model_name][phase][metric].append(value)

            k = 4
            sim_perf_dict[model_name]["test"]["y"].append(y_test)
            sim_perf_dict[model_name]["test"]["pred"].append(pred_test)
            sim_perf_dict[model_name]["test"]["trend_y_test"].append(prepare_for_trend(y_test, k))
            sim_perf_dict[model_name]["test"]["div_y_test"].append(prepare_for_trend(y_test, k))
            sim_perf_dict[model_name]["test"]["trend_pred_test"].append(prepare_for_trend(pred_test, k))
            sim_perf_dict[model_name]["test"]["div_pred_test"].append(prepare_for_trend(pred_test, 3))

            print(f"########################## {i} ##########################")
            print(f"--------{model_name}--------")
            for phase in ["train", "test"]:
                print(f"{phase.capitalize()} results:")
                for metric, value in results[phase].items():
                    if isinstance(value, (int, float)):
                        print(f"{metric}: {value:.4f}")
                print("--------------------------")

    with open('results/syn_base_pref_dict.pkl', 'wb') as f:
        pickle.dump(sim_perf_dict, f)

if __name__ == "__main__":
    main()
