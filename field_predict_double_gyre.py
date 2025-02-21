import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.io import loadmat
from torch.cuda.amp import GradScaler, autocast
import time
import os
import pandas as pd
import matplotlib.pyplot as plt


# LSTM-TransConv Network
class LSTMConvUpsamplePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_steps):
        super(LSTMConvUpsamplePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fcs = nn.ModuleList([nn.Linear(hidden_size, hidden_size * 8 * 8) for _ in range(output_steps)])
        self.conv2 = nn.Conv2d(hidden_size, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 8, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(8, 2, kernel_size=3, padding=1)
        self.output_steps = output_steps
        self._initialize_weights()

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h_0, c_0))
        lstm_output = out[:, -1, :]  # use the output of the last timestep

        outputs = []
        for t in range(self.output_steps):
            fc_out = self.fcs[t](lstm_output)
            fc_out = fc_out.view(-1, self.lstm.hidden_size, 8, 8)  # reshape to 8x8

            upsampled_out = F.interpolate(fc_out, scale_factor=2, mode='bilinear', align_corners=True)
            conv_out = torch.relu(self.conv2(upsampled_out))
            upsampled_out = F.interpolate(conv_out, scale_factor=2, mode='bilinear', align_corners=True)
            conv_out = torch.relu(self.conv3(upsampled_out))
            upsampled_out = F.interpolate(conv_out, scale_factor=2, mode='bilinear', align_corners=True)
            conv_out = torch.relu(self.conv4(upsampled_out))
            upsampled_out = F.interpolate(conv_out, scale_factor=2, mode='bilinear', align_corners=True)
            conv_out = self.conv5(upsampled_out)

            outputs.append(conv_out)

        outputs = torch.stack(outputs, dim=1)
        return outputs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)


class FlowFieldDataset(Dataset):
    def __init__(self, input_data, output_data, u, v, t_step_previous=10, t_step_ahead=5, device=torch.device('cpu')):
        if not (t_step_previous <= 10 and isinstance(t_step_ahead, int) and isinstance(t_step_previous, int)):
            raise ValueError("Check input.")
        self.input_data = torch.tensor(input_data, dtype=torch.float32).to(device)
        self.output_data = torch.tensor(output_data, dtype=torch.long).to(device)  # 确保索引是 long 类型
        self.u = torch.tensor(u, dtype=torch.float32).to(device)
        self.v = torch.tensor(v, dtype=torch.float32).to(device)
        self.t_step_previous = t_step_previous
        self.t_step_ahead = t_step_ahead
        self.device = device

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_seq = self.input_data[idx, 0:self.t_step_previous]
        t = self.output_data[idx, 0].long()  # 确保 t 是 long 类型

        x, y = input_seq[-1, 0], input_seq[-1, 1]  # 取出最新的xy位置
        flow_field_data = self.make_partial_train_data(t, x, y)

        input_seq_normalized = input_seq.clone()

        input_seq_normalized[:, 0] = 2 * (input_seq_normalized[:, 0] / 2.0) - 1  # x[0,2] -> [-1,1]
        input_seq_normalized[:, 1] = 2 * (input_seq_normalized[:, 1] / 1.0) - 1  # y[0,1] -> [-1,1]
        input_seq_normalized[:, 2] = input_seq_normalized[:, 2] / 2.0  # u[-2,2] -> [-1,1]
        input_seq_normalized[:, 3] = input_seq_normalized[:, 3] / 2.0  # v[-2,2] -> [-1,1]

        return input_seq_normalized, flow_field_data

    @staticmethod
    def env_cycle(t_step):
        return t_step % 100

    def make_partial_train_data(self, t, x, y):
        center_x = int(x // 0.005)
        center_y = int(y // 0.005)
        center_x = int(np.clip(center_x, 0, 400))
        center_y = int(np.clip(center_y, 0, 200))

        _output = []
        for i in range(self.t_step_ahead):
            u_data = self.extract_patch(self.u[self.env_cycle(t + i)], center_x, center_y)
            v_data = self.extract_patch(self.v[self.env_cycle(t + i)], center_x, center_y)
            _output.append(torch.cat((u_data, v_data), dim=0))
        _output = torch.stack(_output)
        return _output

    def extract_patch(self, field, center_x, center_y, patch_size=128):
        field = field.unsqueeze(0).unsqueeze(0)
        field_padded = torch.nn.functional.pad(field, (64, 64, 64, 64), mode='reflect')
        field_padded = field_padded.squeeze(0).squeeze(0)  # 变回2D张量
        patch = field_padded[center_x + 1:center_x + 1 + patch_size, center_y + 1:center_y + 1 + patch_size]
        if patch.size() != torch.Size([128, 128]):
            raise ValueError("check xy")
        return patch.unsqueeze(0)


# 训练函数
def train(model, train_loader, criterion, optimizer, scaler, device, epoch, num_epochs, log_dir, grad_clip_value=0.5):
    model.train()
    remaining_epochs = num_epochs - epoch - 1
    total_loss = 0
    start_time = time.time()
    batch_losses = []
    grad_stats = []

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        with autocast():
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_value)
        scaler.step(optimizer)
        scaler.update()
        loss_cpu = loss.item()
        total_loss += loss_cpu
        batch_losses.append({'Epoch': epoch + 1, 'Batch': batch_idx + 1, 'Loss': loss_cpu})

        if batch_idx % 10 == 0:
            elapsed_time = time.time() - start_time
            remaining_batches = len(train_loader) - batch_idx
            estimated_time = elapsed_time / (batch_idx + 1) * remaining_batches
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch {batch_idx}/{len(train_loader)}, Loss: {loss_cpu:.8f}, "
                  f"Epoch Elapsed: {elapsed_time / 60:.2f}min, Epoch Remaining: {estimated_time / 60:.2f}min, "
                  f"Total Remaining: {((elapsed_time + estimated_time) * remaining_epochs + estimated_time) / 3600:.2f}h")

            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_max = param.grad.abs().max().item()
                    grad_stats.append({'Epoch': epoch + 1, 'Batch': batch_idx + 1, 'Layer': name, 'Grad_Max': grad_max})

    batch_loss_df = pd.DataFrame(batch_losses)
    batch_loss_df.to_csv(os.path.join(log_dir, 'train_batch_losses.csv'), mode='a',
                         header=not os.path.exists(os.path.join(log_dir, 'train_batch_losses.csv')), index=False)

    grad_stats_df = pd.DataFrame(grad_stats)
    grad_stats_df.to_csv(os.path.join(log_dir, 'grad_stats.csv'), mode='a',
                         header=not os.path.exists(os.path.join(log_dir, 'grad_stats.csv')), index=False)

    return total_loss / len(train_loader)


def test(model, test_loader, criterion, device, epoch, num_epochs, log_dir):
    model.eval()
    remaining_epochs = num_epochs - epoch - 1
    total_loss = 0
    start_time = time.time()
    batch_losses = []
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            loss_cpu = loss.item()
            total_loss += loss_cpu
            batch_losses.append({'Epoch': epoch + 1, 'Batch': batch_idx + 1, 'Loss': loss_cpu})
            if batch_idx % 10 == 0:  # 每10个batch打印一次
                elapsed_time = time.time() - start_time  # 经过的时间
                remaining_batches = len(test_loader) - batch_idx
                estimated_time = elapsed_time / (batch_idx + 1) * remaining_batches
                print(
                    f"Test Epoch [{epoch + 1}/{num_epochs}], Batch {batch_idx}/{len(test_loader)}, Loss: {loss_cpu:.8f}, "
                    f"Epoch Elapsed: {elapsed_time / 60:.2f}min, Epoch Remains: {estimated_time / 60:.2f}min, "
                    f"Total Remaining: {((elapsed_time + estimated_time) * remaining_epochs + estimated_time) / 3600:.2f}h")

    batch_loss_df = pd.DataFrame(batch_losses)
    batch_loss_df.to_csv(os.path.join(log_dir, 'test_batch_losses.csv'), mode='a',
                         header=not os.path.exists(os.path.join(log_dir, 'test_batch_losses.csv')), index=False)

    return total_loss / len(test_loader)


def load_model(model_path, input_size, hidden_size, num_layers, output_steps, device):
    model = LSTMConvUpsamplePredictor(input_size, hidden_size, num_layers, output_steps).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def plot_comparison(index, dataset, model, device):
    input_seq, true_output = dataset[index]
    # --------------
    # input_seq = [[1.5, 0.5, 1.1542024162330739e-16, -3.4626072486992216e-16], [1.491, 0.5, 1.1384093869387944e-16, -0.32787377206540047], [1.4820000000000002, 0.496721262279346, 0.02801153575871326, -0.6751733170617445], [1.4732801153575874, 0.48996952910872854, 0.07844151094818144, -1.0156135441694443], [1.4650645304670693, 0.47981339366703407, 0.12074090026597659, -1.2923971757841086], [1.457271939469729, 0.46688942190919297, 0.15190847845289387, -1.5524541945353822], [1.449791024254258, 0.45136487996383917, 0.19312297862126543, -1.7467796665517525], [1.4427222540404707, 0.43389708329832166, 0.24758925347063873, -1.84425980541393], [1.4361981465751772, 0.4154544852441824, 0.2824716873018823, -1.8863180169787794], [1.430022863448196, 0.3965913050743946, 0.33905209862331803, -1.8627233096214266]]
    # input_seq = torch.tensor(input_seq, dtype=torch.float32).to(device)
    # --------
    input_seq = input_seq.unsqueeze(0).to(device)
    true_output = true_output.cpu().numpy()
    im_true = np.array([])
    with torch.no_grad():
        predicted_output = model(input_seq).cpu().numpy()

    fig, axes = plt.subplots(2, dataset.t_step_ahead + 1, figsize=(20, 5),
                             gridspec_kw={'width_ratios': [1] * dataset.t_step_ahead + [0.05]})
    for t in range(dataset.t_step_ahead):
        im_true = axes[0, t].imshow(np.rot90(true_output[t, 0, :, :]), cmap='jet', vmin=-2.2, vmax=2.2)
        axes[0, t].set_title(f'True u, t+{t}')
        im_pred = axes[1, t].imshow(np.rot90(predicted_output[0, t, 0, :, :]), cmap='jet', vmin=-2.2, vmax=2.2)
        axes[1, t].set_title(f'Pred u, t+{t}')

    fig.colorbar(im_true, cax=axes[0, -1], shrink=0.95)
    plt.tight_layout()
    plt.show()

    fig, axes = plt.subplots(2, dataset.t_step_ahead + 1, figsize=(20, 5),
                             gridspec_kw={'width_ratios': [1] * dataset.t_step_ahead + [0.05]})
    for t in range(dataset.t_step_ahead):
        im_true = axes[0, t].imshow(np.rot90(true_output[t, 1, :, :]), cmap='jet', vmin=-2.2, vmax=2.2)
        axes[0, t].set_title(f'True v, t+{t}')
        im_pred = axes[1, t].imshow(np.rot90(predicted_output[0, t, 1, :, :]), cmap='jet', vmin=-2.2, vmax=2.2)
        axes[1, t].set_title(f'Pred v, t+{t}')

    fig.colorbar(im_true, cax=axes[0, -1], shrink=0.95)
    plt.tight_layout()
    plt.show()


def train_main(_id, _lr=1e-6, num_epochs=10, batch_size=1024, CUDA_ID=0):
    train_data = loadmat('./training_data.mat')
    input_data = np.array(train_data['inputs'])  # 200000*10*4
    output_data = np.array(train_data['outputs'])  # 200000*1

    field_data = loadmat('./DoubleGyre_grid_0.0to2.0of401_0.0to1.0of201.mat')
    u_grid = np.array(field_data['u_grid'])
    v_grid = np.array(field_data['v_grid'])

    print("============================================================================================")
    # set device to cpu or cuda
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{CUDA_ID}')
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)) + f" @ CUDA: {CUDA_ID}")
    else:
        print("Device set to : cpu")
    print("============================================================================================")

    input_size = 4  # x, y, u, v
    hidden_size = 256
    num_layers = 2
    output_steps = 10  # timestep for Look-Ahead

    model = LSTMConvUpsamplePredictor(input_size, hidden_size, num_layers, output_steps).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=_lr)
    scaler = GradScaler()

    X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

    train_dataset = FlowFieldDataset(X_train, y_train, u_grid, v_grid, t_step_ahead=output_steps, device=device)
    test_dataset = FlowFieldDataset(X_test, y_test, u_grid, v_grid, t_step_ahead=output_steps, device=device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model_id = f'ID_{_id}'
    model_dir = f'DG_pred_models/{model_id}'
    log_dir = f'DG_pred_logs/{model_id}'

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, criterion, optimizer, scaler, device, epoch, num_epochs, log_dir)
        test_loss = test(model, test_loader, criterion, device, epoch, num_epochs, log_dir)
        print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}')

        torch.save(model.state_dict(), os.path.join(model_dir, f'model_epoch_{epoch + 1}.pth'))

        with open(os.path.join(log_dir, 'epoch_losses.txt'), 'a') as f:
            f.write(
                f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.8f}, Test Loss: {test_loss:.8f}\n')

    torch.save(model.state_dict(), os.path.join(model_dir, 'model_final.pth'))


def test_main(data_idx=10086):
    # 加载数据
    train_data = loadmat('./training_data.mat')
    input_data = np.array(train_data['inputs'])  # n*10*4
    output_data = np.array(train_data['outputs'])  # n*1

    # 加载流场
    field_data = loadmat('./flow_field/DoubleGyre_grid_0.0to2.0of401_0.0to1.0of201.mat')
    u_grid = np.array(field_data['u_grid'])
    v_grid = np.array(field_data['v_grid'])

    # print("============================================================================================")
    # set device to cpu or cuda
    device = torch.device('cpu')
    # if torch.cuda.is_available():
    #     device = torch.device('cuda:0')
    #     torch.cuda.empty_cache()
    #     print("Device set to : " + str(torch.cuda.get_device_name(device)))
    # else:
    #     print("Device set to : cpu")
    # print("============================================================================================")

    input_size = 4  # x, y, u, v
    hidden_size = 256
    num_layers = 2
    output_steps = 10

    model_path = f'pretrained_models/field_predict_models/DG_pred_model.pth'  # 这里需要替换成实际的模型路径
    model = load_model(model_path, input_size, hidden_size, num_layers, output_steps, device)

    dataset = FlowFieldDataset(input_data, output_data, u_grid, v_grid, t_step_ahead=output_steps, device=device)

    plot_comparison(data_idx, dataset, model, device)  # data_idx for the specific entry of dataset that is compared


if __name__ == '__main__':
    # start_id = 48
    # train_times = 1
    # lrs = [10e-3]
    #
    # for i in range(train_times):
    #     train_main(_id=(start_id + i), _lr=lrs[i],
    #                num_epochs=200, batch_size=1024, CUDA_ID=2)

    test_main(data_idx=10010)
    test_main(data_idx=10086)
    test_main(data_idx=12345)
    test_main(data_idx=180000)
