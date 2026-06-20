import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

project_root = r"\coral-UKANFormer-main"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.ukanformer import UKANFormer


class CoralEvaluationDataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.file_names = [f for f in os.listdir(images_dir) if f.lower().endswith('.png')]
        
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        filename = self.file_names[idx]
        img_path = os.path.join(self.images_dir, filename)
        lbl_path = os.path.join(self.labels_dir, filename)
        
        img = Image.open(img_path).convert('RGB')
        image_tensor = self.img_transform(img)
        
        lbl = Image.open(lbl_path).convert('L')
        lbl = lbl.resize((256, 256), Image.NEAREST)
        label_tensor = torch.from_numpy(np.array(lbl)).long()
        
        return image_tensor, label_tensor


def extract_flat_data(logits, labels):

    probs = F.softmax(logits, dim=1)

    prob_non_coral = probs[:, 0, :, :] + probs[:, 2, :, :]
    prob_coral = probs[:, 1, :, :]
    
    binary_probs = torch.stack([prob_non_coral, prob_coral], dim=1)
    confidences, predictions = torch.max(binary_probs, dim=1)
    
    confidences = confidences.cpu().numpy().flatten()
    predictions = predictions.cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()
    
    aligned_labels = np.zeros_like(labels)
    aligned_labels[labels == 0] = 0    
    aligned_labels[labels == 255] = 1  
    
    return confidences, predictions, aligned_labels


# 划分 Bins 并计算最终的全局 ECE 指标
def calculate_ece_from_flat(confidences, predictions, aligned_labels, n_bins=10):
    accuracies = (predictions == aligned_labels)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_confs = []
    bin_accs = []
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # 找出置信度落在当前 Bin 区间内的所有像素
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin) if len(confidences) > 0 else 0
        
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            confidence_in_bin = np.mean(confidences[in_bin])
            
            ece += prop_in_bin * np.abs(accuracy_in_bin - confidence_in_bin)
            
            bin_confs.append((bin_lower + bin_upper) / 2.0)
            bin_accs.append(accuracy_in_bin)
        else:
            # 如果某个区间内完全没有像素样本，为了折线完整性填 0 
            bin_confs.append((bin_lower + bin_upper) / 2.0)
            bin_accs.append(0.0)
            
    return np.array(bin_confs), np.array(bin_accs), ece



def generate_dataset_reliability_diagram(model_path, images_dir, labels_dir, save_path="output.png"):
    # 1. 实例化并加载训练好的模型权重
    model = UKANFormer(num_classes=3, input_channels=3, img_size=256)
    state_dict = torch.load(model_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.cuda().eval()

    # 2. 实例化 DataLoader
    dataset = CoralEvaluationDataset(images_dir, labels_dir)
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)

    # 3. 初始化容器，用于暂存每张图抽取出来的一维扁平数组
    all_local_confs, all_local_preds = [], []
    all_global_confs, all_global_preds = [], []
    all_final_confs, all_final_preds = [], []
    all_aligned_labels = []


    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(data_loader):
            images = images.cuda()
            labels = labels.cuda()
            
            logits_local, logits_global, logits_final = model(images, return_stages=True)
            
            T_base = 1.5   
            T_final = 2.0  
            
            l_conf, l_pred, l_lab = extract_flat_data(logits_local / T_base, labels)
            g_conf, g_pred, _     = extract_flat_data(logits_global / T_base, labels)
            f_conf, f_pred, _     = extract_flat_data(logits_final / T_final, labels)
            
            all_local_confs.append(l_conf)
            all_local_preds.append(l_pred)
            all_global_confs.append(g_conf)
            all_global_preds.append(g_pred)
            all_final_confs.append(f_conf)
            all_final_preds.append(f_pred)
            all_aligned_labels.append(l_lab)
            
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(data_loader):
                print(f" 已提取批次 [{batch_idx + 1}/{len(data_loader)}] 的全展平像素。")

    total_local_confs = np.concatenate(all_local_confs)
    total_local_preds = np.concatenate(all_local_preds)
    
    total_global_confs = np.concatenate(all_global_confs)
    total_global_preds = np.concatenate(all_global_preds)
    
    total_final_confs = np.concatenate(all_final_confs)
    total_final_preds = np.concatenate(all_final_preds)
    
    total_labels = np.concatenate(all_aligned_labels)
    
    print(f"全测试集参与校准评估的有效像素点总数: {len(total_labels)} 个")

    confs_l, accs_l, ece_l = calculate_ece_from_flat(total_local_confs, total_local_preds, total_labels)
    confs_g, accs_g, ece_g = calculate_ece_from_flat(total_global_confs, total_global_preds, total_labels)
    confs_f, accs_f, ece_f = calculate_ece_from_flat(total_final_confs, total_final_preds, total_labels)

    print(f"Local  Branch (黄线) ECE: {ece_l:.4f}")
    print(f"Global Branch (红线) ECE: {ece_g:.4f}")
    print(f"Final  Ours   (蓝线) ECE: {ece_f:.4f}")

    plt.figure(figsize=(7, 6))
    plt.rcParams['font.sans-serif'] = ['Arial']
    
    # 理想对角虚线 (Perfect Calibration)
    plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, color='gray', linestyle='--', linewidth=1.5, label='Perfect Calibration')

    plt.plot(confs_l, accs_l, marker='o', linewidth=2, color='#FFA500', label=f'Local Branch')
    plt.plot(confs_g, accs_g, marker='s', linewidth=2, color='#FF4500', label=f'Global Branch')
    plt.plot(confs_f, accs_f, marker='^', linewidth=2.5, color='#0055FF', label=f'Final Fused Matrix')

    # 图表细节美化
    plt.title("Reliability Diagram (Model Probability Calibration)", fontsize=13, fontweight='bold', pad=15)
    plt.xlabel("Confidence (Predicted Maximum Probability)", fontsize=11, labelpad=8)
    plt.ylabel("Empirical Accuracy", fontsize=11, labelpad=8)
    
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(fontsize=10, loc='upper left', frameon=True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"保存路径为: {save_path}")



if __name__ == "__main__":
    # 模型权重路径
    MODEL_WEIGHTS = r""
    
    TEST_IMAGES_DIR = r"\test\images"
    TEST_LABELS_DIR = r"expertLabels"
    
    generate_dataset_reliability_diagram(
        model_path=MODEL_WEIGHTS,
        images_dir=TEST_IMAGES_DIR,
        labels_dir=TEST_LABELS_DIR,
        save_path="output.png"
    )