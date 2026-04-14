import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
from glob import glob
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random, time
import itertools
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from utils import *
from datetime import datetime
from torch.autograd import Variable
from IPython.display import clear_output
try:
    from urllib.request import URLopener
except ImportError:
    from urllib import URLopener

# 引入四个模型
from model.unet import UNet
from model.unetformer import UNetFormer
from model.ukan import UKAN
from model.ukanformer import UKANFormer


def initialize_network(model, num_classes, weights=None):
    if model == 'UNet':
        net = UNet(n_classes=num_classes).cuda()
    elif model == 'UNetFormer':
        net = UNetFormer(num_classes=num_classes).cuda()
    elif model == 'UKAN':
        net = UKAN(num_classes=num_classes).cuda()
    elif model == 'UKANFormer':
        net = UKANFormer(num_classes=num_classes).cuda()
    return net

# 上来就保存模型文件
def train(net, optimizer, epochs, scheduler=None, weights=None, save_epoch=1):
    global MODE
    weights = weights.cuda() if weights else None
    iter_ = 0
    MIoU_best = 0.0
    print(f"Total batches in train loader: {len(train_loader)}")
    
    # 打印每个批次的数据形状
    for batch_idx, (data, target) in enumerate(train_loader):
        # target shape就是label shape
        print(f"Batch {batch_idx}: Data shape {data.shape}, Target shape {target.shape}")
        break  

    # 修改了模型的保存方式
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if DATASET == 'Coral':
        save_path = f'./results/{net.__class__.__name__}/exp_Coral/exp_Coral_{timestamp}'

    print('net')
    print(net)
    
    os.makedirs(save_path, exist_ok=True) 

    for e in range(1, epochs + 1):
        # --- 训练阶段 ---
        net.train()  # 确保模型处于训练模式
        epoch_start_time = time.time()
        print(f'Begin training epoch {e}/{epochs}...')
    
        # 遍历训练数据
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = net(data)
            loss = loss_calc(output, target, weights)
            loss.backward()
            optimizer.step()

            # 每隔 50 个 batch 打印训练信息
            if iter_ % 50 == 0:
                clear_output(wait=True)
                pred = np.argmax(output.data.cpu().numpy()[0], axis=0)
                gt = target.data.cpu().numpy()[0]
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Train (epoch {e}/{epochs}) [{batch_idx}/{len(train_loader)}] "
                    f"LR: {current_lr:.8f}, Loss: {loss.item():.6f}, Acc: {accuracy(pred, gt):.6f}")
        
            iter_ += 1
            del data, target, loss  # 及时释放显存
            torch.cuda.empty_cache()

        # --- 验证阶段（每个 epoch 必须执行）---
        net.eval()  # 切换为评估模式
        print("Evaluating...")
        with torch.no_grad():
            MIoU = test(net, test_ids, mode='Test', all=False, stride=Stride_Size) 
        print(f"Epoch {e} | Validation mIoU: {MIoU:.4f}")

        # --- 更新学习率（基于 mIoU）---
        scheduler.step(MIoU)  # 关键：传入当前 epoch 的 mIoU

        # --- 保存模型（save_epoch=1 时每个 epoch 保存）---
        if e % save_epoch == 0:
            if MIoU > MIoU_best:
                print(f"Saving model with improved mIoU: {MIoU:.4f}")
                if DATASET == 'Coral':
                    torch.save(net.state_dict(), f'{save_path}/{net.__class__.__name__}_epoch{e}_{MIoU}.pth')
                MIoU_best = MIoU
            torch.cuda.empty_cache()

        # --- 记录 epoch 时间 ---
        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {e} completed in {epoch_duration:.2f} seconds.\n")


MAIN_FOLDER = r""
def test(net, test_ids, mode='Test',all=False, stride=WINDOW_SIZE[0], batch_size=BATCH_SIZE, window_size=WINDOW_SIZE):
    # 动态生成路径
    data_folder = os.path.join(MAIN_FOLDER, mode.lower(), 'images', '{}.png')
    label_folder = os.path.join(MAIN_FOLDER, mode.lower(), 'original', '{}.png')
    eroded_folder = os.path.join(MAIN_FOLDER, mode.lower(), 'eroded', '{}.png')
    # print('data_folder:',data_folder)
    # print('label_folder:',label_folder)
    # print('eroded_folder:',eroded_folder)
    
    # 仅加载测试影像（输入）
    test_images = [1/255 * np.asarray(io.imread(data_folder.format(id)), dtype='float32') for id in test_ids]
    # 加载标签（不参与预测，仅用于评估）
    test_labels = [convert_from_color(np.asarray(io.imread(label_folder.format(id)))) for id in test_ids] 
    eroded_labels = [convert_from_color(np.asarray(io.imread(eroded_folder.format(id)))) for id in test_ids]

    # 检查文件是否存在
    if not os.path.exists(data_folder.format(test_ids[0])):
        raise FileNotFoundError(f"文件不存在: {data_folder.format(test_ids[0])}")
    
    all_preds = []
    all_gts = []

    with torch.no_grad():
        # 遍历测试图像和对应标签
        for img, gt, gt_e in tqdm(zip(test_images, test_labels, eroded_labels), total=len(test_ids), leave=False):
            # 对当前图像进行滑动窗口预测
            pred = sliding_window_predict(net, img, stride, window_size, batch_size)
            all_preds.append(pred)
            all_gts.append(gt_e)  # 使用腐蚀标签作为评估基准
    
    accuracy = metrics(
        np.concatenate([p.ravel() for p in all_preds]),
        np.concatenate([p.ravel() for p in all_gts]).ravel())
    if all:
        return accuracy, all_preds, all_gts
    else:
        return accuracy


if __name__ == '__main__':
    # Initialize model and dataset
    net = initialize_network(MODEL, N_CLASSES, weights=None)

    # Count parameters
    params = 0
    for name, param in net.named_parameters():
        params += param.nelement()
    print(f"Total parameters: {params}")

    # Load the datasets
    print(f"Training: {len(train_ids)}, Testing: {len(test_ids)}, Stride_Size: {Stride_Size}, BATCH_SIZE: {BATCH_SIZE}")

    # epochs
    epochs = 300
    # epochs = 150
    # Prepare optimizer
    base_lr = 2.5 * 10 ** -6
    params_dict = dict(net.named_parameters())
    params = []
    for key, value in params_dict.items():
        params += [{'params': [value], 'lr': base_lr}]

    optimizer = optim.SGD(params, lr=base_lr, momentum=0.9, weight_decay=0.0005)
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',        # 监控指标为mIoU,所以应该是max
        factor=0.5,        
        patience=5,      
        verbose=True,      
        threshold=1e-3,    
        min_lr=1 * 10**-8     
        # 选一个非常小的最低学习率
    )

    # Train or Test mode
    if MODE == 'Train':
        if DATASET == 'Coral':
            train_set = Coral_dataset(train_ids, cache=CACHE)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE)
        train(net, optimizer, epochs, scheduler)
    # 对应修改这里的文件路径
    elif MODE == 'Test':
        if DATASET == 'Coral':
            test_set = Coral_dataset(test_ids, cache=CACHE)
            model_path = r""
            net.load_state_dict(torch.load(model_path), strict=False)
            net.eval()
            MIoU, all_preds, all_gts = test(net, test_ids, all=True, stride=32)
            print("MIoU: ", MIoU)
            for p, id_ in zip(all_preds, test_ids):
                img = convert_to_color(p)
                io.imsave(f'./results/UNet/exp_Coral/exp_Coral_20250520_225145/{MODEL}_tile_{id_}.png', img)
