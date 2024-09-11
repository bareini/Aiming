import numpy as np
import pickle

# Constants
NUM_PATIENTS = 1000
EVENT_SIZE = 7
EVENT_SIZE2 = 7
TREND_SIZE = 1.5
UNCERTAIN = True

def add_duration(events, duration_interval=10):
    """
    Add duration to events.

    Args:
    events (np.array): Binary vector indicating rare event locations.
    duration_interval (int): Max possible duration of the event in steps.

    Returns:
    np.array: Rare events vector including duration.
    """
    idx = 0
    while idx < len(events):
        if events[idx] == 1:
            duration_interval = min(duration_interval, len(events) - idx)
            event_duration = np.random.choice(np.arange(duration_interval))
            events[idx:idx + event_duration] = 1
            idx += event_duration
        idx += 1
    return events

def add_back_trend(events, max_steps=20):
    """
    Add a trend back to baseline signal instead of a sudden drop.

    Args:
    events (np.array): Binary vector indicating rare event locations.
    max_steps (int): The number of steps required to get back.

    Returns:
    np.array: Events with added back trend.
    """
    events = events.astype(float)
    idx, ev_flag = 0, 0
    while idx < len(events):
        if (ev_flag == 0) and (events[idx] != 0):
            ev_flag = events[idx]
        elif (ev_flag != 0) and (events[idx] == 0):
            max_steps = min(max_steps, len(events) - idx)
            slope = np.random.choice(np.arange(max_steps))
            events[idx:idx+slope] = 1 - (np.arange(slope) + 1) / slope
            idx += slope
            ev_flag = 0
        else:
            idx += 1
    return events

def gen_trend(max_len=499):
    """
    Generate an evolving trend event.

    Args:
    max_len (int): Max possible length of the event.

    Returns:
    np.array: Generated trend mask.
    """
    trend_mask = np.zeros(max_len)
    duration = np.random.randint(1, 35)
    trend_arr = 0.01 * np.arange(duration)**2
    trend_arr[trend_arr > 15] = 10
    start = np.random.randint(max_len - duration)
    trend_mask[start:(start + duration)] = trend_arr
    trend_mask[start + duration] = trend_mask[start + duration - 1] - 2
    return trend_mask

def generate_sliding_window(s, p, dot_val=True):
    """
    Generate a sliding window per sequence sample.

    Args:
    s (np.array): Input sequence.
    p (int): Window size.
    dot_val (bool): Whether to include values in the window.

    Returns:
    np.array: Stacked sliding windows.
    """
    chunks = []
    if len(s) > p:
        for i in range(p+1, len(s)):
            chunk = s[i-p-1:i+1].values if dot_val else s[i-p-1:i]
            chunks.append(chunk)
    return np.vstack(chunks)

def generate_data():
    """
    Generate synthetic patient data.

    Returns:
    dict: Dictionary containing generated data.
    """
    a = np.random.randint(18, 80, size=NUM_PATIENTS)
    g = np.random.randint(0, 2, size=NUM_PATIENTS)
    L = 0.1 * a + 2 * g
    L2 = 0.25 * a + 0.1 * g

    x = np.linspace(-np.pi/2, 100*np.pi, 501)
    x1 = np.sin(0.25*x) - np.sin(0.1*x)
    x2 = np.sin(0.5*x) - np.sin(0.05*x)
    x3 = 0.5 * np.sin(0.1*1)

    base_sig1 = (np.tile(L, (len(x1), 1)).T +
                 np.tile(x1, (NUM_PATIENTS, 1)) +
                 np.random.normal(0, 0.25, (NUM_PATIENTS, len(x1))))

    base_sig2 = (np.tile(L2, (len(x1), 1)).T +
                 np.tile(x2, (NUM_PATIENTS, 1)) +
                 np.random.normal(0, 0.1, (NUM_PATIENTS, len(x2))))

    base_sig3 = (np.tile(x3, (NUM_PATIENTS, 1)) +
                 np.roll(base_sig1, 1) +
                 0.1 * np.roll(base_sig2, 2))
    base_sig3 = base_sig3[:, 2:] + np.random.normal(0, 0.1, base_sig3[:, 2:].shape)

    # Scenario 1
    onset1 = np.random.binomial(1, 0.0007, (NUM_PATIENTS, len(x1)))
    rare_event1 = np.vstack([add_duration(onset1[i, :]) for i in range(NUM_PATIENTS)])
    base_sig1 += EVENT_SIZE * rare_event1

    onset1 = np.roll(onset1, 1)[:, 2:]
    rare_event1 = np.roll(rare_event1, 1)[:, 2:]

    if UNCERTAIN:
        certainty = np.random.binomial(1, 0.75, (NUM_PATIENTS, rare_event1.shape[1]))
        rare_event1 *= certainty

    rare_event1 = np.vstack([add_back_trend(rare_event1[i, :]) for i in range(NUM_PATIENTS)])
    base_sig3 -= EVENT_SIZE * rare_event1

    # Scenario 2
    onset2 = np.random.binomial(1, 0.0005, (NUM_PATIENTS, len(x2)))
    rare_event2 = np.vstack([add_duration(onset2[i, :]) for i in range(NUM_PATIENTS)])
    base_sig2 += EVENT_SIZE * rare_event2

    onset2 = np.roll(onset2, 2)[:, 2:]
    rare_event2 = np.roll(rare_event2, 2)[:, 2:]

    if UNCERTAIN:
        certainty = np.random.binomial(1, 0.6, (NUM_PATIENTS, rare_event2.shape[1]))
        rare_event2 *= certainty

    rare_event2 = np.vstack([add_back_trend(rare_event2[i, :]) for i in range(NUM_PATIENTS)])
    base_sig3 += EVENT_SIZE2 * rare_event2

    # Scenario 3
    trend_event_embed = np.vstack([add_back_trend(gen_trend()) for _ in range(NUM_PATIENTS)])
    trend_event = trend_event_embed
    base_sig3 += TREND_SIZE * trend_event_embed

    return {
        "base_sig1": base_sig1[:, 2:],
        "base_sig2": base_sig2[:, 2:],
        "base_sig3": base_sig3,
        "onset1": onset1,
        "onset2": onset2,
        "trend_event": trend_event
    }

def prepare_datasets(data, back_steps=2, is_valid=False):
    """
    Prepare datasets for training, testing, or validation.

    Args:
    data (dict): Generated data.
    back_steps (int): Number of steps to look back.
    is_valid (bool): Whether to prepare validation set.

    Returns:
    dict: Prepared datasets.
    """
    patient_list = []
    for i in range(NUM_PATIENTS):
        patient_data = np.hstack([
            generate_sliding_window(data["base_sig1"][i, :], back_steps, dot_val=False),
            generate_sliding_window(data["base_sig2"][i, :], back_steps, dot_val=False),
            generate_sliding_window(data["base_sig3"][i, :], back_steps, dot_val=False)
        ])
        patient_list.append(patient_data)

    roll_num = back_steps + 1
    y = np.roll(data["base_sig3"], -roll_num, axis=1)[:, :-roll_num]
    onset1 = np.roll(data["onset1"], -roll_num, axis=1)[:, :-roll_num]
    onset2 = np.roll(data["onset2"], -roll_num, axis=1)[:, :-roll_num]
    trend_event = np.roll(data["trend_event"], -roll_num, axis=1)[:, :-roll_num] > 0
    patient_arr = np.array(patient_list)

    if is_valid:
        return {
            "data_val": [patient_arr.reshape(-1, 9)],
            "targets_val": [y.flatten()],
            "onset1s_val": [onset1.flatten()],
            "onset2s_val": [onset2.flatten()],
            "trend_events_val": [trend_event.flatten()],
            "val_ids": [np.repeat(np.arange(y.shape[0]), y.shape[1])]
        }

    # Train-test split
    test_size = NUM_PATIENTS // 5
    split_idx = [np.arange(test_size*i, test_size*(i+1)) for i in range(5)]
    
    datasets = {
        "trains": [], "tests": [], "train_targets": [], "test_targets": [],
        "onset1s_train": [], "onset2s_train": [], "trend_events_train": [],
        "onset1s_test": [], "onset2s_test": [], "trend_events_test": [],
        "train_ids": [], "test_ids": []
    }

    for i in range(5):
        subset_ids = np.concatenate([split_idx[j] for j in range(5) if j != i])
        
        datasets["train_ids"].append(np.repeat(np.arange(y[subset_ids].shape[0]), y[subset_ids].shape[1]))
        datasets["trains"].append(patient_arr[subset_ids].reshape(-1, 9))
        datasets["train_targets"].append(y[subset_ids].flatten())
        datasets["tests"].append(patient_arr[split_idx[i]].reshape(-1, 9))
        datasets["test_targets"].append(y[split_idx[i]].flatten())
        datasets["test_ids"].append(np.repeat(np.arange(y[split_idx[i]].shape[0]), y[split_idx[i]].shape[1]))
        datasets["onset1s_train"].append(onset1[subset_ids].flatten())
        datasets["onset2s_train"].append(onset2[subset_ids].flatten())
        datasets["trend_events_train"].append(trend_event[subset_ids].flatten())
        datasets["onset1s_test"].append(onset1[split_idx[i]].flatten())
        datasets["onset2s_test"].append(onset2[split_idx[i]].flatten())
        datasets["trend_events_test"].append(trend_event[split_idx[i]].flatten())

    return datasets

if __name__ == "__main__":
    data = generate_data()
    datasets = prepare_datasets(data)
    
    with open('sim_data.pkl', 'wb') as f:
        pickle.dump(datasets, f)
