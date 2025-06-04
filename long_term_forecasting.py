import pandas as pd
import os
import time
from torch import optim
import numpy as np
import matplotlib.pyplot as plt
import BiMamba4TS
import torch
from utils.loss import get_loss
from tqdm import tqdm

models_dict = {
    'BiMamba4TS': BiMamba4TS
}

optimizer_catagory = {
    'adam': optim.Adam,
    'sgd': optim.SGD
}

loss_funcs = {
    "mse": torch.nn.MSELoss(),
    "mae": torch.nn.L1Loss(),
    "huber": torch.nn.HuberLoss(reduction='mean', delta=1.0)
}


class LTF_Trainer:
    def __init__(self, args, task, setting, corr=None) -> None:
        self.args = args
        self.setting = setting
        self.r_path = args.results
        self.c_path = args.checkpoints
        self.p_path = args.predictions
        self.device = torch.device(args.device)

        # 确保路径存在
        os.makedirs(self.c_path, exist_ok=True)
        os.makedirs(self.r_path, exist_ok=True)
        os.makedirs(self.p_path, exist_ok=True)

        print(f"Checkpoints path: {self.c_path}")
        print(f"Results path: {self.r_path}")
        print(f"Predictions path: {self.p_path}")

        if hasattr(args, 'SRA') and args.SRA:
            model = models_dict[self.args.model].Model(self.args, corr=corr).to(self.device)
        else:
            model = models_dict[self.args.model].Model(self.args).to(self.device)

        if args.use_multi_gpu:
            self.model = torch.nn.DataParallel(model, device_ids=self.args.device_ids)
        else:
            self.model = model

    def train(self, data):
        print('>>>>>> start training : {} >>>>>>'.format(self.setting))

        train_loader = data.get('combined_train_loader', None)  # 训练集加载器
        vali_loader = data.get('combined_val_loader', None)  # 验证集加载器
        train_steps = len(train_loader)
        vali_steps = len(vali_loader) if vali_loader else 0

        optimizer = optimizer_catagory[self.args.opt](self.model.parameters(), lr=self.args.learning_rate)

        # 定义损失函数
        accel_loss_func = loss_funcs["mse"]
        friction_loss_func = loss_funcs["mae"]

        train_loss_list, vali_accel_loss_list, vali_friction_loss_list, epoch_list = [], [], [], []
        for epoch in range(self.args.train_epochs):
            epoch_list.append(epoch + 1)
            epoch_time = time.time()  # 记录训练开始时间
            train_loss = []
            min_batch_loss = float('inf')
            best_outputs, best_batch_y = None, None

            self.model.train()

            # 创建训练进度条
            with tqdm(total=train_steps + vali_steps, desc=f"Epoch {epoch + 1}/{self.args.train_epochs}", unit='batch',
                      leave=True) as progress_bar:
                # 训练过程
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    optimizer.zero_grad()

                    batch_x = batch_x.float().to(self.args.device)
                    batch_y = batch_y.float().to(self.args.device)

                    dec_inp = torch.zeros_like(batch_y[:, -self.args.label_len:, :]).float()
                    dec_inp = torch.cat([batch_x, dec_inp], dim=1).float()

                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs, _ = outputs
                    f_dim = -1 if self.args.task == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]

                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.device)

                    # 分别计算加速度损失和摩擦损失
                    accel_loss = accel_loss_func(outputs, batch_y)
                    friction_loss = friction_loss_func(outputs, batch_y)

                    # 合并损失
                    total_loss = accel_loss + friction_loss

                    train_loss.append(total_loss.item())
                    total_loss.backward()
                    optimizer.step()

                    if total_loss.item() < min_batch_loss:
                        min_batch_loss = total_loss.item()
                        best_outputs = outputs.cpu().detach().numpy()
                        best_batch_y = batch_y.cpu().detach().numpy()

                    # 更新训练进度条
                    progress_bar.set_postfix({'train_loss': np.mean(train_loss),
                                              'accel_loss': accel_loss.item(),
                                              'friction_loss': friction_loss.item()})
                    progress_bar.update(1)

                train_loss = np.average(train_loss)
                train_loss_list.append(train_loss)

                save_dir = os.path.join(self.args.datasave, f'epoch{epoch + 1}')
                os.makedirs(save_dir, exist_ok=True)
                batch_save_path = os.path.join(save_dir, 'min_loss_batch.csv')

                df = pd.DataFrame({
                    'outputs': best_outputs.flatten(),
                    'batch_y': best_batch_y.flatten()
                })
                df.to_csv(batch_save_path, index=False)

                model_save_path = f"{self.r_path}/model_epoch_{epoch + 1}.pth"
                torch.save(self.model.state_dict(), model_save_path)

                # 验证集损失
                if vali_loader:
                    vali_accel_loss, vali_friction_loss = self.validate(vali_loader, accel_loss_func,
                                                                        friction_loss_func, progress_bar)
                    vali_accel_loss_list.append(vali_accel_loss)
                    vali_friction_loss_list.append(vali_friction_loss)

            epoch_duration = time.time() - epoch_time
            print(f"Epoch {epoch + 1}: Train Loss: {np.mean(train_loss):.4f} | "
                  f"Train Accel Loss: {accel_loss:.4f}, Train Friction Loss: {friction_loss:.4f} | "
                  f"Validation Accel Loss: {vali_accel_loss:.4f}, Validation Friction Loss: {vali_friction_loss:.4f} | "
                  f"Training Time: {epoch_duration:.2f} seconds")

        # 保存损失结果
        loss_save_path = os.path.join(self.r_path, 'loss.csv')
        loss_data = pd.DataFrame({
            'epoch': epoch_list,
            'train_loss': train_loss_list,
            'vali_accel_loss': vali_accel_loss_list,
            'vali_friction_loss': vali_friction_loss_list
        })
        loss_data.to_csv(loss_save_path, index=False)

    def validate(self, vali_loader, accel_loss_func, friction_loss_func, progress_bar):
        total_loss = []  # 总损失
        total_accel_loss = []  # 加速度损失
        total_friction_loss = []  # 摩擦损失
        self.model.eval()

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float().to(self.args.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.label_len:, :]).float()
                dec_inp = torch.cat([batch_x, dec_inp], dim=1).float()

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs, _ = outputs

                f_dim = -1 if self.args.task == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.device)

                # 计算损失（与训练相同）
                accel_loss = accel_loss_func(outputs, batch_y)
                friction_loss = friction_loss_func(outputs, batch_y)
                total_loss.append((accel_loss + friction_loss).item())
                total_accel_loss.append(accel_loss.item())
                total_friction_loss.append(friction_loss.item())

                # 更新验证进度条
                progress_bar.set_postfix({
                    'accel_loss': np.mean(total_accel_loss),
                    'friction_loss': np.mean(total_friction_loss),
                    'total_loss': np.mean(total_loss)
                })
                progress_bar.update(1)

        avg_total_loss = np.average(total_loss)
        avg_accel_loss = np.average(total_accel_loss)
        avg_friction_loss = np.average(total_friction_loss)

        return avg_accel_loss, avg_friction_loss

    def test(self, data):
        print('>>>>>>>start testing : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(self.setting))
        # 加载已经训练好的模型权重
        self.model.load_state_dict(torch.load(os.path.join(self.r_path, 'model_epoch_3.pth')))
        self.model.eval()

        test_loader = data.get('combined_test_loader', None)

        preds = []
        trues = []
        mse_list, rmse_list, mae_list, mape_list = [], [], [], []
        batch_losses = []  # List to store batch losses

        # 打开 CSV 文件并准备写入每个 batch 的预测和真实值
        batch_save_path = os.path.join(self.p_path, 'predictions_vs_trues_batch.csv')
        with open(batch_save_path, mode='w', newline='') as f:
            csv_writer = pd.writer(f)
            csv_writer.writerow(['batch', 'true_value', 'predicted_value'])  # 写入表头

            # 使用 tqdm 来显示测试过程的进度条
            with tqdm(total=len(test_loader), desc="Testing", unit='batch', leave=True) as progress_bar:
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    batch_x = batch_x.float().to(self.args.device)
                    batch_y = batch_y.float().to(self.args.device)

                    # 构建解码器输入
                    dec_inp = torch.zeros_like(batch_y[:, -self.args.label_len:, :]).float()
                    dec_inp = torch.cat([batch_x, dec_inp], dim=1).float()

                    # 前向传播
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    outputs, _ = outputs

                    # 选择输出预测维度
                    f_dim = -1 if self.args.task == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.args.device)

                    # 分离预测和真实值
                    preds_batch = outputs.detach().cpu()
                    trues_batch = batch_y.detach().cpu()

                    # 计算损失
                    mse, rmse, mae, mape = get_loss(preds_batch, trues_batch)
                    mse_list.append(float(mse))
                    rmse_list.append(float(rmse))
                    mae_list.append(float(mae))
                    mape_list.append(float(mape))

                    # Calculate batch loss and append to the list
                    batch_loss = get_loss(preds_batch, trues_batch)[0]  # MSE loss is returned as the first element
                    batch_losses.append(batch_loss)

                    preds.append(preds_batch.numpy())
                    trues.append(trues_batch.numpy())

                    # Save batch predictions and true values to CSV
                    for j in range(preds_batch.shape[0]):  # Loop over the batch size
                        for k in range(preds_batch.shape[1]):  # Loop over the sequence length (pred_len)
                            # Write batch index, true value, and predicted value to CSV
                            csv_writer.writerow(
                                [f'{i + 1}_{j + 1}_{k + 1}', trues_batch[j, k, 0].item(), preds_batch[j, k, 0].item()])

                    print(
                        f'Batch {i + 1}/{len(test_loader)}, MSE: {mse:.6f}, RMSE: {rmse:.6f}, MAE: {mae:.6f}, MAPE: {mape:.6f}')

                    # 绘制 preds 和 trues 的对比图
                    plt.figure(figsize=(10, 6))
                    plt.plot(trues_batch[0, :, 0].numpy(), label='True Values', color='blue')  # 绘制真实值
                    plt.plot(preds_batch[0, :, 0].numpy(), label='Predicted Values', color='red')  # 绘制预测值
                    plt.title(f'Prediction vs True at Batch {i + 1}')
                    plt.xlabel('Time Steps')
                    plt.ylabel('Values')
                    plt.legend()
                    # plt.show()

            # 将所有批次的预测和真实值保存到 CSV 文件
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            save_path = os.path.join(self.p_path, 'predictions_vs_trues.csv')
            df = pd.DataFrame({
                'preds': preds.flatten(),
                'trues': trues.flatten()
            })
            df.to_csv(save_path, index=False)
            print(f'Predictions and true values saved to {save_path}')

        # 打印并保存最终结果
        mse, rmse, mae, mape = np.average(mse_list), np.average(rmse_list), np.average(mae_list), np.average(mape_list)
        result_path = os.path.join(self.p_path, "result.txt")
        print('mse:{:.6f}, mae:{:.6f}, rmse:{:.6f}, mape:{:.6f}'.format(mse, mae, rmse, mape))
        with open(result_path, 'a') as f:
            f.write(self.setting + "\n")
            f.write('mse:{:.6f}, mae:{:.6f}, rmse:{:.6f}, mape:{:.6f} ==== lr={} seed={}\n\n'.format(
                mse, mae, rmse, mape, self.args.learning_rate, self.args.seed
            ))

        # Save the batch-wise loss results
        loss_save_path = os.path.join(self.p_path, 'batch_losses.csv')
        loss_df = pd.DataFrame({
            'batch_loss': batch_losses
        })
        loss_df.to_csv(loss_save_path, index=False)
        print(f'Batch losses saved to {loss_save_path}')

        return round(float(mse), 6)

    '''
    def predict(self, pred_loader):
        print('>>>>>>>start predicting : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(self.setting))
        self.model.load_state_dict(torch.load(os.path.join(self.r_path, "model_epoch_31.pth"), map_location="cuda:0"))

        preds = []
        trues = []
        inputs = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.args.device)
                batch_y = batch_y.float().to(self.args.device)

                batch_x_mark = batch_x_mark.float().to(self.args.device)
                batch_y_mark = batch_y_mark.float().to(self.args.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                if hasattr(self.args, 'use_gcn') and self.args.use_gcn:
                    outputs, _, _ = outputs
                else:
                    outputs, _ = outputs

                pred = outputs.detach().cpu().numpy()
                true = batch_y[:, -self.args.pred_len:, :].detach().cpu().numpy()
                input = batch_x[:, -self.args.seq_len:, :].detach().cpu().numpy()
                preds.append(pred)
                trues.append(true)
                inputs.append(input)

        preds = np.array(preds)
        trues = np.array(trues)
        inputs = np.array(inputs)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        inputs = inputs.reshape(-1, inputs.shape[-2], inputs.shape[-1])

        folder_path = self.p_path
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + f'/{self.args.model}_pred.npy', preds)
        np.save(folder_path + f'/{self.args.model}_true.npy', trues)
        np.save(folder_path + f'/{self.args.model}_in.npy', inputs)

        print('Prediction done...')
'''
