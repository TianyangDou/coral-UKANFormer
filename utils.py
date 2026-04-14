import numpy as np
from sklearn.metrics import confusion_matrix
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import itertools
from torchvision.utils import make_grid
from torch.autograd import Variable
from PIL import Image
from PIL import ImageEnhance
from skimage import io
from tqdm import tqdm
import os
import cv2

# Parameters
WINDOW_SIZE = (256, 256) # Patch size

STRIDE = 32 # Stride for testing
IN_CHANNELS = 3 # Number of input channels
BATCH_SIZE = 8

# MODEL = 'UNet'
# MODEL = 'UNetFormer' 
# MODEL = 'UKAN'
MODEL = 'UKANFormer'

MODE = 'Train'
# MODE = 'Test'

DATASET = 'Coral'

# NVIDIA GeForce RTX 4070
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(MODEL + ', ' + MODE + ', ' + DATASET)

# 用于coral数据集中生成train_ids和test_ids
def generate_ids(folder):
    """从指定目录获取所有PNG文件的文件名（不含扩展名）"""
    ids = [os.path.splitext(f)[0] for f in os.listdir(folder) if f.endswith('.png')]
    return ids

if DATASET == 'Coral':
    # Moorea:1
    # Tetiaroa:2
    # Tahiti:3
    # YongXing:4
    # YuZhuo:5
    # HuaGuang:6
    # LangHua:7
    MAIN_FOLDER = ""
    # 生成训练集和测试集 ID
    train_ids = generate_ids(os.path.join(MAIN_FOLDER, 'train', 'images'))
    test_ids = generate_ids(os.path.join(MAIN_FOLDER, 'test', 'images'))
    Stride_Size = 32
    LABELS = ["Others","Reef Biota"]
    N_CLASSES = len(LABELS) # Number of classes
    WEIGHTS = torch.ones(N_CLASSES) # Weights for class balancing
    CACHE = True # Store the dataset in-memory
    if MODE == 'Train':
        DATA_FOLDER = os.path.join(MAIN_FOLDER, 'train', 'images', '{}.png')
        LABEL_FOLDER = os.path.join(MAIN_FOLDER, 'train', 'original', '{}.png')
        ERODED_FOLDER = os.path.join(MAIN_FOLDER, 'train', 'eroded', '{}.png')
    elif MODE == 'Test':
        DATA_FOLDER = os.path.join(MAIN_FOLDER, 'test', 'images', '{}.png')
        LABEL_FOLDER = os.path.join(MAIN_FOLDER, 'test', 'original', '{}.png')
        ERODED_FOLDER = os.path.join(MAIN_FOLDER, 'test', 'eroded', '{}.png')
    # define the coral color palette
    palette = {
        0: (0, 0, 0),  # Others (black)
        1: (255, 255, 255)    # Reef Biota (white)
    }
    invert_palette = {v: k for k, v in palette.items()}

def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i
    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to numeric labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)
    # 遍历所有颜色映射关系
    for c, i in palette.items():
        # 创建颜色掩码（注意：图像颜色可能是 uint8 类型，需转换为元组）
        mask = np.all(arr_3d == np.array(c).astype(arr_3d.dtype), axis=-1)
        arr_2d[mask] = i
    return arr_2d

def save_img(tensor, name):
    tensor = tensor.cpu() .permute((1, 0, 2, 3))
    im = make_grid(tensor, normalize=True, scale_each=True, nrow=8, padding=2).permute((1, 2, 0))
    im = (im.data.numpy() * 255.).astype(np.uint8)
    Image.fromarray(im).save(name + '.jpg')

# Coral数据集用不同的加载方式
class Coral_dataset(torch.utils.data.Dataset):
    def __init__(self, ids, data_files=DATA_FOLDER, label_files=LABEL_FOLDER,
                 cache=False, augmentation=True):
        super(Coral_dataset, self).__init__()

        self.augmentation = augmentation
        self.cache = cache

        # List of files
        self.data_files = [DATA_FOLDER.format(id) for id in ids]
        self.label_files = [LABEL_FOLDER.format(id) for id in ids]

        # Sanity check : raise an error if some files do not exist
        for f in self.data_files + self.label_files:
            if not os.path.isfile(f):
                raise KeyError('{} is not a file !'.format(f))

        # Initialize cache dicts
        self.data_cache_ = {}
        self.label_cache_ = {}

    def __len__(self):
        return len(self.data_files)
    
    def data_augmentation(cls, *arrays, flip=True, mirror=True):
        will_flip, will_mirror = False, False
        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True

        results = []
        for array in arrays:
            if will_flip:
                if len(array.shape) == 2:
                    array = array[::-1, :]
                else:
                    array = array[:, ::-1, :]
            if will_mirror:
                if len(array.shape) == 2:
                    array = array[:, ::-1]
                else:
                    array = array[:, :, ::-1]
            results.append(np.copy(array))

        return tuple(results)

    def __getitem__(self, idx):
        """
        直接加载第 idx 个样本（无需随机裁剪，因为数据已经是 256x256 的小块）
        """
        # 加载数据（若启用缓存则优先从缓存读取）
        if self.cache and idx in self.data_cache_:
            data = self.data_cache_[idx]
        else:
            # 加载数据并归一化到 [0, 1]
            data = io.imread(self.data_files[idx]).astype(np.float32) / 255.0
            # 转换通道顺序 HWC -> CHW
            if data.ndim == 3:
                data = data.transpose((2, 0, 1))  # 转为 [C, H, W]
            elif data.ndim == 2:
                data = np.expand_dims(data, axis=0)  # 单通道 -> [1, H, W]
            if self.cache:
                self.data_cache_[idx] = data

        # 加载标签（若启用缓存则优先从缓存读取）
        if self.cache and idx in self.label_cache_:
            label = self.label_cache_[idx]
        else:
            # 加载标签并转换为类别索引（标签是RGB格式）
            label = np.asarray(convert_from_color(io.imread(self.label_files[idx])), dtype='int64')
            if self.cache:
                self.label_cache_[idx] = label

        # 数据增强
        if self.augmentation:
            data, label = self.data_augmentation(data, label)

        # 转换为 PyTorch 张量
        return torch.from_numpy(data).float(), torch.from_numpy(label).long()

# 下面是损失函数与测度评价部分了
# 这一部分是没有问题的
class CrossEntropy2d_ignore(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d_ignore, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss
    
def loss_calc(pred, label, weights):
    """
    This function returns cross entropy loss for semantic segmentation
    """
    # out shape batch_size x channels x h x w -> batch_size x channels x h x w
    # label shape h x w x 1 x batch_size  -> batch_size x 1 x h x w
    label = Variable(label.long()).cuda()
    criterion = CrossEntropy2d_ignore().cuda()

    return criterion(pred, label, weights)

def CrossEntropy2d(input, target, weight=None, size_average=True):
    """ 2D version of the cross entropy loss """
    dim = input.dim()
    if dim == 2:
        return F.cross_entropy(input, target, weight, size_average)
    elif dim == 4:
        output = input.view(input.size(0), input.size(1), -1)
        output = torch.transpose(output, 1, 2).contiguous()
        output = output.view(-1, output.size(2))
        target = target.view(-1)
        return F.cross_entropy(output, target, weight, size_average)
    else:
        raise ValueError('Expected 2 or 4 dimensions (got {})'.format(dim))

# def sliding_window_predict(net, img, stride, window_size, batch_size):
#     """对单张图像进行滑动窗口预测"""
#     pred = np.zeros(img.shape[:2] + (N_CLASSES,))  # 初始化预测结果
#     total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size
#     # 滑动窗口预测
#     for i, coords in enumerate(tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total, leave=False)):
#         # 提取图像块
#         image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
#         image_patches = np.asarray(image_patches)
#         image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)
#         # 模型预测
#         outs = net(image_patches)
#         outs = outs.data.cpu().numpy()
#         # 将预测结果填充到对应位置
#         for out, (x, y, w, h) in zip(outs, coords):
#             out = out.transpose((1, 2, 0))
#             pred[x:x + w, y:y + h] += out
    
#     # 取最大概率的类别作为最终预测
#     pred = np.argmax(pred, axis=-1)
#     return pred

def sliding_window_predict(net, img, stride, window_size, batch_size):
    pred = np.zeros(img.shape[:2] + (N_CLASSES,), dtype='float32')
    count = np.zeros(img.shape[:2], dtype='float32')

    total = count_sliding_window(img, step=stride, window_size=window_size) // batch_size

    for i, coords in enumerate(tqdm(grouper(batch_size, sliding_window(img, step=stride, window_size=window_size)), total=total, leave=False)):
        image_patches = [np.copy(img[x:x + w, y:y + h]).transpose((2, 0, 1)) for x, y, w, h in coords]
        image_patches = np.asarray(image_patches)
        image_patches = Variable(torch.from_numpy(image_patches).cuda(), volatile=True)

        outs = net(image_patches)
        outs = outs.data.cpu().numpy()

        for out, (x, y, w, h) in zip(outs, coords):
            out = out.transpose((1, 2, 0))
            pred[x:x + w, y:y + h] += out
            count[x:x + w, y:y + h] += 1

    # 避免除以0（某些像素可能没有被覆盖）
    count = np.maximum(count, 1e-8)
    pred /= count[..., None]  # 广播除以每个像素的覆盖次数
    pred = np.argmax(pred, axis=-1)
    return pred


def accuracy(input, target):
    return 100 * float(np.count_nonzero(input == target)) / target.size

def sliding_window(top, step=10, window_size=(20, 20)):
    """ Slide a window_shape window across the image with a stride of step """
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            yield x, y, window_size[0], window_size[1]

def count_sliding_window(top, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image """
    c = 0
    for x in range(0, top.shape[0], step):
        if x + window_size[0] > top.shape[0]:
            x = top.shape[0] - window_size[0]
        for y in range(0, top.shape[1], step):
            if y + window_size[1] > top.shape[1]:
                y = top.shape[1] - window_size[1]
            c += 1
    return c

def grouper(n, iterable):
    """ Browse an iterator by chunk of n elements """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk

def one_hot(label, n_classes, requires_grad=True):
    """Return One Hot Label"""
    one_hot_label = torch.eye(
        n_classes, device='cuda', requires_grad=requires_grad)[label]
    one_hot_label = one_hot_label.transpose(1, 3).transpose(2, 3)

    return one_hot_label
 
def metrics(predictions, gts, label_values=LABELS):
    cm = confusion_matrix(
        gts,
        predictions,
        labels=range(len(label_values)))

    print("Confusion matrix :")
    print(cm)
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("%d pixels processed" % (total))
    print("Total accuracy : %.2f" % (accuracy))

    Acc = np.diag(cm) / cm.sum(axis=1)
    for l_id, score in enumerate(Acc):
        print("%s: %.4f" % (label_values[l_id], score))
    print("---")

    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("%s: %.4f" % (label_values[l_id], score))
    print('mean F1Score: %.4f' % (np.nanmean(F1Score[:])))
    print("---")

    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total * total)
    kappa = (pa - pe) / (1 - pe)
    print("Kappa: %.4f" %(kappa))
    print("---")

    # Compute MIoU coefficient
    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
    print(MIoU)
    MIoU = np.nanmean(MIoU[:])
    print('mean MIoU: %.4f' % (MIoU))
    print("---")
    return MIoU
