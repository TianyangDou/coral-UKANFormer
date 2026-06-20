import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

project_root = r""

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.ukanformer import UKANFormer

def get_entropy_flat(logits):
    T = 0.30
    scaled_logits = logits / T
    
    probs = F.softmax(scaled_logits, dim=1)
    eps = 1e-7
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
    num_classes = logits.shape[1]
    entropy = entropy / torch.log(torch.tensor(num_classes, dtype=torch.float32))
    
    return entropy[0].cpu().numpy().flatten()


def generate_figure13(model_path, image_path, save_path="output.png"):

    model = UKANFormer(num_classes=2, input_channels=3, img_size=256) 
    state_dict = torch.load(model_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(img).unsqueeze(0).cuda()

    with torch.no_grad():
        logits_local, logits_global, logits_final = model(image_tensor, return_stages=True)
        
        ent_local = get_entropy_flat(logits_local)
        ent_global = get_entropy_flat(logits_global)
        ent_final = get_entropy_flat(logits_final)

    plt.figure(figsize=(9, 6))
    
    plt.rcParams['font.sans-serif'] = ['Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 调整直方图参数，使其呈现平滑的阶梯线/填充效果
    bins = 500 
    
    plt.hist(ent_local, bins=bins, range=(0, 1), density=True, histtype='step', 
             linewidth=2.5, color='#FFA500', label='Local Branch Features')
    plt.hist(ent_global, bins=bins, range=(0, 1), density=True, histtype='step', 
             linewidth=2.5, color='#FF4500', label='Global Attention Features')
    plt.hist(ent_final, bins=bins, range=(0, 1), density=True, histtype='step', 
             linewidth=3.0, color='#0055FF', label='Final Fused Matrix (Ours)')
    
    plt.hist(ent_final, bins=bins, range=(0, 1), density=True, histtype='stepfilled', 
             color='#0055FF', alpha=0.12)

    plt.title("Statistical Distribution of Predictive Entropy (Uncertainty)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Predictive Entropy Value (0 = Absolutes Certainty, 1 = Max Ambiguity)", fontsize=12, labelpad=10)
    plt.ylabel("Probability Density", fontsize=12, labelpad=10)
    
    plt.xlim(-0.02, 1.02)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(fontsize=11, loc='upper right', frameon=True, shadow=False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"保存路径为: {save_path}")

if __name__ == "__main__":
    MODEL_WEIGHTS = r""
    TEST_IMAGE = r"test\images\image.png" 
    
    generate_figure13(MODEL_WEIGHTS, TEST_IMAGE, save_path="output.png")