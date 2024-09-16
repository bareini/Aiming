import torch
import numpy as np
import pickle as pkl
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
import metrics_u

from syn_dataset import SimDataset
from syn_models import SIMLSTM

def get_device(cuda_id):
    if not isinstance(cuda_id, str):
        cuda_id = str(cuda_id)
    return torch.device(f"cuda:{cuda_id}" if torch.cuda.is_available() else "cpu")

def calc_batch_r2(out, target, weight):
    r2 = sum(r2_score(out[i, :], target[i], sample_weight=weight[i]) for i in range(out.shape[0]))
    return np.mean(r2)

def cross_entropy(softmax, y_target):
    return -torch.sum(torch.log(softmax) * y_target, dim=1)

def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()

def disable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.eval()

# Constants
HIDDEN_DIM = 6
INPUT_DIM = 9
LEARNING_RATES = [0.05, 0.01, 0.005, 0.002]
NUM_EPOCHS = 4000
BATCH_SIZE = 104
CUDA_ID = 3

MODEL_NAME = 'sim_lstm'

device = get_device(CUDA_ID)

def train(epoch, train_loader, model, optimizer, device):
    true_value_list, preds_list = [], []
    ids_list, onsets_1, onsets_2, trend_events = [], [], [], []

    model.train()
    train_loss = 0
    loss_fun = torch.nn.MSELoss()

    for data, yt, onset1, onset2, trend_event, idx in train_loader:
        data, yt, onset1, onset2, trend_event, idx = [x.to(device) for x in [data, yt, onset1, onset2, trend_event, idx]]

        onsets_1.append(onset1.flatten().squeeze())
        onsets_2.append(onset2.flatten().squeeze())
        trend_events.append(trend_event.flatten().squeeze())
        ids_list.append(idx.flatten().squeeze())

        optimizer.zero_grad()
        out = model(input=data)
        target_tab = yt.float()

        loss = loss_fun(out.squeeze(), target_tab)
        loss.backward()
        train_loss += loss.item()

        out_tab = out.detach().cpu().numpy().squeeze()
        target_tab = target_tab.detach().cpu().numpy().squeeze()

        true_value_list.append(target_tab.flatten().squeeze())
        preds_list.append(out_tab.flatten().squeeze())

    print(f'Train MSE epoch {epoch}: {train_loss}, learning rate: {optimizer.param_groups[0]["lr"]}')

    return (train_loss, true_value_list, preds_list, onsets_1, onsets_2, trend_events, ids_list)

def test(test_loader, model, device):
    true_value_list, preds_list = [], []
    ids_list, onsets_1, onsets_2, trend_events = [], [], [], []

    test_loss = 0
    loss_fun = torch.nn.MSELoss()

    model.eval()
    with torch.no_grad():
        for data, yt, onset1, onset2, trend_event, idx in test_loader:
            data, yt, onset1, onset2, trend_event, idx = [x.to(device) for x in [data, yt, onset1, onset2, trend_event, idx]]

            onsets_1.append(onset1.flatten().squeeze())
            onsets_2.append(onset2.flatten().squeeze())
            trend_events.append(trend_event.flatten().squeeze())

            out = model(input=data)
            target_tab = yt.float()

            loss = loss_fun(out.squeeze(), target_tab)
            test_loss += loss.item()

            out_tab = out.cpu().numpy().squeeze()
            target_tab = target_tab.cpu().numpy().squeeze()

            preds_list.append(out_tab.flatten().squeeze())
            true_value_list.append(target_tab.flatten().squeeze())
            ids_list.append(idx.flatten())

    print(f'Test Average MSE: {test_loss}')

    return (test_loss, true_value_list, preds_list, onsets_1, onsets_2, trend_events, ids_list)

def main():
    print('Started training')

    results = {
        'range_loss_list': [], 'range_loss_list_raw': [],
        'trend_loss_list': [], 'trend_loss_list_raw': [],
        'loss_u_div_list': [], 'loss_u_div_list_raw': [],
        'range_loss_list_test': [], 'range_loss_list_raw_test': [],
        'trend_loss_list_test': [], 'trend_loss_list_raw_test': [],
        'loss_u_div_list_test': [], 'loss_u_div_list_raw_test': [],
        'onset1_list_train': [], 'onset2_list_train': [], 'trend_event_list_train': [],
        'rmse_onset1_train': [], 'rmse_onset2_train': [], 'rmse_trend_event_train': [],
        'rmse_range_train': [], 'rmse_train': [], 'rmse_test': [],
        'normal_range_list_train': [], 'normal_range_list_test': [],
        'onset1_list_test': [], 'onset2_list_test': [], 'trend_event_list_test': [],
        'y': [], 'pred': [], 'id_list': []
    }

    for i in range(5):
        print(f'============== Dataset {i} ==============')
        train_dataset = SimDataset(dataidx=i, mode='train')
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        test_dataset = SimDataset(dataidx=i, mode='test')
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

        loss_u_range = metrics_u.URangeLoss(L=10, x0_high=train_dataset.mean, k_high=0.08, b_high=0,
                                            x0_low=train_dataset.mean, k_low=0.1, b_low=0)
        loss_u_trend = metrics_u.TrendLoss()

        model = SIMLSTM(input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATES[0], betas=(0.95, 0.99), eps=1e-08)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 50, 500, 1000, 1500], gamma=0.5)

        for epoch in range(NUM_EPOCHS):
            train_results = train(epoch, train_loader, model, optimizer, device)
            optimizer.step()
            scheduler.step()

        true_value_list_train = np.concatenate(train_results[1])
        preds_list_train = np.concatenate(train_results[2])
        ids_list = np.concatenate(train_results[6])

        mean, std = true_value_list_train.mean(), true_value_list_train.std()
        high_th, low_th = mean + std, mean - std
        normal_range_train = (true_value_list_train > high_th) | (true_value_list_train < low_th)

        rmse = np.sqrt(mean_squared_error(true_value_list_train, preds_list_train))
        results['rmse_train'].append(rmse)

        onsets_1_train = np.concatenate(train_results[3])
        onsets_2_train = np.concatenate(train_results[4])
        trend_events_train = np.concatenate(train_results[5])

        results['onset1_list_train'].append(onsets_1_train)
        results['onset2_list_train'].append(onsets_2_train)
        results['trend_event_list_train'].append(trend_events_train)
        results['normal_range_list_train'].append(normal_range_train)

        # Calculate train RMSE on events
        results['rmse_range_train'].append(np.sqrt(mean_squared_error(true_value_list_train[normal_range_train],
                                                                      preds_list_train[normal_range_train])))
        results['rmse_onset1_train'].append(np.sqrt(mean_squared_error(true_value_list_train[onsets_1_train],
                                                                       preds_list_train[onsets_1_train])))
        results['rmse_onset2_train'].append(np.sqrt(mean_squared_error(true_value_list_train[onsets_2_train],
                                                                       preds_list_train[onsets_2_train])))
        results['rmse_trend_event_train'].append(np.sqrt(mean_squared_error(true_value_list_train[trend_events_train],
                                                                            preds_list_train[trend_events_train])))

        # Calculate loss_u loss train
        l, l_raw = loss_u_range.range_u_loss(true_value_list_train, preds_list_train, raw=True)
        results['range_loss_list'].append(l)
        results['range_loss_list_raw'].append(l_raw)

        l, l_raw = loss_u_trend.trend_loss(y=true_value_list_train, pred=preds_list_train, ids=ids_list, raw=True)
        results['trend_loss_list'].append(l)
        results['trend_loss_list_raw'].append(l_raw)

        l, l_raw = loss_u_trend.trend_dev_loss(y=true_value_list_train, pred=preds_list_train, ids=ids_list, raw=True)
        results['loss_u_div_list'].append(l)
        results['loss_u_div_list_raw'].append(l_raw)

        # Test phase
        test_results = test(test_loader, model, device)

        true_value_list_test = np.concatenate(test_results[1])
        results['y'].append(true_value_list_test)
        preds_list_test = np.concatenate(test_results[2])
        results['pred'].append(preds_list_test)

        ids_list = np.concatenate(test_results[6])
        results['id_list'].append(ids_list)

        rmse = np.sqrt(mean_squared_error(true_value_list_test, preds_list_test))
        results['rmse_test'].append(rmse)
        print(f"Test RMSE: {rmse}")

        normal_range_test = (true_value_list_test > high_th) | (true_value_list_test < low_th)

        onsets_1_test = np.concatenate(test_results[3])
        onsets_2_test = np.concatenate(test_results[4])
        trend_events_test = np.concatenate(test_results[5])

        results['onset1_list_test'].append(onsets_1_test)
        results['onset2_list_test'].append(onsets_2_test)
        results['trend_event_list_test'].append(trend_events_test > trend_events_test.std())
        results['normal_range_list_test'].append(normal_range_test)

        # Calculate test RMSE per event
        if normal_range_test.shape[0] > 0:
            results['rmse_range_test'].append(np.sqrt(mean_squared_error(true_value_list_test[normal_range_test],
                                                                         preds_list_test[normal_range_test])))
        else:
            results['rmse_range_test'].append(-1)

        results['rmse_onset1_test'].append(np.sqrt(mean_squared_error(true_value_list_test[onsets_1_test],
                                                                      preds_list_test[onsets_1_test])))
        results['rmse_onset2_test'].append(np.sqrt(mean_squared_error(true_value_list_test[onsets_2_test],
                                                                      preds_list_test[onsets_2_test])))
        results['rmse_trend_event_test'].append(np.sqrt(mean_squared_error(true_value_list_test[trend_events_test > trend_events_test.std()],
                                                                           preds_list_test[trend_events_test > trend_events_test.std()])))

        # Calculate loss_u loss test
        l, l_raw = loss_u_range.range_u_loss(true_value_list_test, preds_list_test, raw=True)
        results['range_loss_list_test'].append(l)
        results['range_loss_list_raw_test'].append(l_raw)

        l, l_raw = loss_u_trend.trend_loss(y=true_value_list_test, pred=preds_list_test, ids=ids_list, raw=True)
        results['trend_loss_list_test'].append(l)
        results['trend_loss_list_raw_test'].append(l_raw)

        l, l_raw = loss_u_trend.trend_dev_loss(y=true_value_list_test, pred=preds_list_test, ids=ids_list, raw=True)
        results['loss_u_div_list_test'].append(l)
        results['loss_u_div_list_raw_test'].append(l_raw)

    with open('res.pkl', 'wb') as f:
        pkl.dump(results, f)


if __name__ == "__main__":
    main()
