import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

project_root = r"D:\3_WHU\WHU_coral\coral-UKANFormer-main"

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from model.ukanformer import UKANFormer


def compute_pixel_entropy(logits):

    probs = F.softmax(logits, dim=1) 

    eps = 1e-7
    
    # 计算香农熵 H = -sum(P * log(P))
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1) 
    
    num_classes = logits.shape[1]
    entropy = entropy / torch.log(torch.tensor(num_classes, dtype=torch.float32))
    
    return entropy[0].cpu().numpy()


def generate_figure12(model_path, image_path, save_path="output.png"):

    model = UKANFormer(num_classes=3, input_channels=3, img_size=256) 

    state_dict = torch.load(model_path, map_location='cpu')


    if 'model_state_dict' in state_dict: 
        model.load_state_dict(state_dict['model_state_dict'])
    else:
        model.load_state_dict(state_dict)

    model.cuda()
    model.eval()


    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])
    image_tensor = transform(img).unsqueeze(0).cuda() # 增加 Batch 维度 [1, 3, 256, 256]

    with torch.no_grad():
        
        logits_local, logits_global, logits_final = model(image_tensor, return_stages=True)
        
        
        entropy_local = compute_pixel_entropy(logits_local)
        entropy_global = compute_pixel_entropy(logits_global)
        entropy_final = compute_pixel_entropy(logits_final)


    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    
    # (a) 原始图片显示
    orig_img = np.array(img.resize((256, 256)))
    axes[0].imshow(orig_img)
    axes[0].set_title("(a) Original Image", fontsize=14, pad=10)
    axes[0].axis('off')
    
    # (b) Local Branch 局部卷积分支熵图
    im1 = axes[1].imshow(entropy_local, cmap='jet', vmin=0, vmax=1)
    axes[1].set_title("(b) Local Branch", fontsize=14, pad=10)
    axes[1].axis('off')
    
    # (c) Global Attention Branch 全局自注意力分支熵图
    im2 = axes[2].imshow(entropy_global, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title("(c) Global Attention Branch", fontsize=14, pad=10)
    axes[2].axis('off')
    
    # (d) Final Fused Matrix 最终融合后的预测熵图
    im3 = axes[3].imshow(entropy_final, cmap='jet', vmin=0, vmax=1)
    axes[3].set_title("(d) Final Fused Matrix", fontsize=14, pad=10)
    axes[3].axis('off')
    
    # 蓝色代表确定性高(熵接近0)，红色代表模型极度迷茫(熵接近1)
    cbar_ax = fig.add_axes([0.93, 0.15, 0.012, 0.7]) 
    cbar = fig.colorbar(im3, cax=cbar_ax)
    cbar.set_label('Predictive Entropy (Spatial Uncertainty)', rotation=270, labelpad=18, fontsize=12)
    

    plt.subplots_adjust(wspace=0.1) 
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"保存路径为: {save_path}")


if __name__ == "__main__":

    MODEL_WEIGHTS = r""
    TEST_IMAGE = r"test\images\image.png" 
    
    generate_figure12(MODEL_WEIGHTS, TEST_IMAGE, save_path="output.png")