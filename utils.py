import os
import json
import random
import sys
import torch
from tqdm import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import SimpleITK as sitk
class MyDataset(Dataset):
    def __init__(self, data,transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]['image']
        label = self.data[index]['label']
        # 应用数据转换
        if self.transform is not None:
            sample = self.transform(sample)
        return sample,label
def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证各平台顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".nii", ".nii.gz"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 排序，保证各平台顺序一致
        images.sort()
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))
        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)
    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    assert len(train_images_path) > 0, "number of training images must greater than 0."
    assert len(val_images_path) > 0, "number of validation images must greater than 0."
    train_data_dict = [{'image': image, 'label': label} for image, label in zip(train_images_path, train_images_label)]
    val_data_dict = [{'image': image, 'label': label} for image, label in zip(val_images_path, val_images_label)]
    return train_data_dict, val_data_dict
def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)  # 累计损失
    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    optimizer.zero_grad()
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]
        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch+1,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)   # 累计预测正确的样本数
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch+1,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def hu2uint8(image, HU_min=-1200.0, HU_max=600.0, HU_nan=-2000.0):
    """
    Convert HU unit into uint8 values. First bound HU values by predfined min
    and max, and then normalize
    image: 3D numpy array of raw HU values from CT series in [z, y, x] order.
    HU_min: float, min HU value.
    HU_max: float, max HU value.
    HU_nan: float, value for nan in the raw CT image.
    """
    image_new = np.array(image)
    image_new[np.isnan(image_new)] = HU_nan

    # normalize to [0, 1]
    image_new = (image_new - HU_min) / (HU_max - HU_min)
    image_new = np.clip(image_new, 0, 1)
    image_new = (image_new * 255).astype('uint8')

    return image_new
def dicom2img(dcm_image, window_center, window_width):
    """
    将一张脑CT切片变换为一幅图像
    """
    assert (isinstance(window_center, int) and isinstance(window_width, int)) or \
           (len(window_center) == len(window_width))

    if isinstance(window_center, int):
        window_center = [window_center]
        window_width = [window_width]
        channel = 1
    else:
        channel = len(window_center)

    img = np.zeros(list(dcm_image.shape) + [channel], dtype="uint8")
    for i, (c, w) in enumerate(zip(window_center, window_width)):
        HU_min, HU_max = c - w // 2, c + w // 2
        img[..., i] = hu2uint8(dcm_image, HU_min=HU_min, HU_max=HU_max)
    return img

def load_dcm(sorted_dcm_list):
    """
    Return img array and [z,y,x]-ordered origin and spacing
    """
    itkimage = sitk.ReadImage(sorted_dcm_list)
    numpyImage = sitk.GetArrayFromImage(itkimage)

    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))

    return numpyImage, numpyOrigin, numpySpacing
def remove_black_edge(img):
    """
    去除脑CT切片图像中的黑边
    img: H, W, C
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    close = cv2.morphologyEx(img[..., 0], cv2.MORPH_OPEN, kernel, iterations=3)
    if close.sum() > 0:
        (hmin, hmax), (wmin, wmax) = map(lambda x: (x.min(), x.max()), np.where(close > 0))
        return img[hmin:hmax, wmin:wmax]
    else:
        return np.zeros_like(img)

def preprocess(dicom_dir):
    """
    对dicom格式的图像进行预处理
    return array数组
    """
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    slice_ids = os.listdir(dicom_dir)
    dcm_list = [os.path.join(dicom_dir, i) for i in slice_ids]
    dcm_image, origin, spacing = load_dcm(dcm_list)
    # brain, subdural, bone
    img = dicom2img(dcm_image, window_center=[40, 80, 40], window_width=[80, 200, 380])
    # remove black boundary
    box = np.where(img > 0)
    y_min, y_max, x_min, x_max = box[1].min(), box[1].max(), box[2].min(), box[2].max()
    img = img[:, y_min:y_max, x_min:x_max]
    return img


if __name__ == '__main__':
    dcm_file = r'E:\medical\chen\1_A'
    img = preprocess(dcm_file)
    print(img.shape)
    plt.imshow(img[10])
    plt.show()
