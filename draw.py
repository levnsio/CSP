import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_utils import Skel_type
import math


# def plot(u):



class SimplepointLayer(nn.Module):
    def __init__(self, half_size, device, num_classes, num_iter):
        super(SimplepointLayer, self).__init__()

        self.unfold = nn.Unfold(kernel_size=2 * half_size + 1, padding=half_size, stride=1)
        self.device = device
        self.num_class = num_classes
        self.num_iter = num_iter

    def build_order_masks(self, img):
        # 创建行和列索引的张量，并直接放在目标设备上
        bs, _, h, w = img.shape
        rows = torch.arange(h, device=self.device).view(-1, 1)
        cols = torch.arange(w, device=self.device).view(1, -1)

        # 计算行列的模2结果
        rows_mod = rows % 2
        cols_mod = cols % 2

        # 一次性生成所有掩码（布尔张量）
        mask00 = (rows_mod == 0) & (cols_mod == 0)  # 00
        mask01 = (rows_mod == 0) & (cols_mod == 1)  # 01
        mask10 = (rows_mod == 1) & (cols_mod == 0)  # 10
        mask11 = (rows_mod == 1) & (cols_mod == 1)  # 11

        # 转换为浮点张量并添加批次/通道维度
        order_mask = torch.stack((
            mask11.float().unsqueeze(0).unsqueeze(0),
            mask01.float().unsqueeze(0).unsqueeze(0),
            mask10.float().unsqueeze(0).unsqueeze(0),
            mask00.float().unsqueeze(0).unsqueeze(0)
        ))

        return order_mask.expand(4, bs, 1, h, w)

    def patchify(self, image):
        B, C, H, W = image.shape
        image_patch = self.unfold(image.float())  # divide the image into patches via sliding window
        # image patch, shape [B,C*9,H*W]
        image_patch = image_patch.permute(0, 2, 1).reshape(B * H * W, C, 3, 3)  # B, H*W, C*9

        return image_patch

    def build_n2(self, pattern):
        """构建批量N2邻域矩阵
        Args:
            pattern: 输入张量，形状为 [B, N, H, W] (此处 B=4, N=500, H=3, W=3)
        Returns:
            n2: 输出的邻域矩阵，形状同输入，dtype=torch.uint8
        """
        # 初始化输出张量（全零）
        n2 = torch.zeros_like(pattern, dtype=torch.uint8)

        # 上边界：检查每个模式的(0,1)位置
        top_mask = pattern[..., 0:1, 1:2] >= 0.5  # 形状: [B, N, 1, 1]
        n2[..., 0:1, :] = top_mask.expand_as(n2[..., 0:1, :])  # 扩展并赋值第一行

        # 下边界：检查每个模式的(2,1)位置
        bottom_mask = pattern[..., 2:3, 1:2] >= 0.5
        n2[..., 2:3, :] = bottom_mask.expand_as(n2[..., 2:3, :])  # 扩展并赋值最后一行

        # 左边界：检查每个模式的(1,0)位置
        left_mask = pattern[..., 1:2, 0:1] >= 0.5
        n2[..., :, 0:1] = left_mask.expand_as(n2[..., :, 0:1])  # 扩展并赋值第一列

        # 右边界：检查每个模式的(1,2)位置
        right_mask = pattern[..., 1:2, 2:3] >= 0.5
        n2[..., :, 2:3] = right_mask.expand_as(n2[..., :, 2:3])  # 扩展并赋值最后一列

        return n2

    def one_process_diff(self, img, alpha, t, k):
        """
        Args:
            img: 输入张量，形状为 (bs, c, h, w) 其中 h=w=3
            alpha, t, k: 标量参数
        Returns:
            处理后的张量，形状为 (bs, c)
        """
        bs, c, h, w = img.shape

        # 步骤1: 构建N2邻域矩阵
        mask = self.build_n2(img)  # 形状 (bs, c, 3, 3)
        # mask = 1

        # 步骤2: 应用mask并重塑张量
        x4 = (mask * img).reshape(bs * c, h * w)  # 形状 (bs*c, 9)

        # 步骤3: 应用排列和复制操作
        perm = [0, 1, 2, 5, 8, 7, 6, 3, 4]
        x4 = x4[:, perm]  # 应用排列
        x4[:, 8] = x4[:, 0]

        # 步骤4: 计算差分结果 (向量化)
        result4 = x4[:, 1:] - x4[:, :-1]  # 形状 (bs*c, 8)

        # 步骤5: 向量化条件覆盖 (避免循环)
        # 条件1: (x4[:, i] < 0.5) & (x4[:, i+1] >= 0.5)
        mask1 = (x4[:, :8] < 0.5) & (x4[:, 1:] >= 0.5)
        # 条件2: (x4[:, i] >= 0.5) & (x4[:, i+1] < 0.5)
        mask2 = (x4[:, :8] >= 0.5) & (x4[:, 1:] < 0.5)

        # 应用条件覆盖
        result4[mask1] = 1
        result4[mask2] = -1

        # 步骤6: 应用sigmoid变换和阈值处理
        result4 = torch.sigmoid((torch.abs(result4) - t) * (alpha ** 2))
        # result4 = torch.threshold(result4, 0.5, 0)  # 小于0.5的值设为0

        # 步骤7: 求和并处理零值 (向量化)
        result4_sum = result4.sum(dim=1)/2.0  # 形状 (bs*c,)
        # result4_sum[result4_sum == 0] = 4  # 将零值替换为4

        # 步骤8: 应用最终变换
        # result = result4_sum * torch.exp(-(result4_sum - 1) ** 2 / (2 * k ** 2))
        result = torch.exp(-(result4_sum - 1) ** 2 / (2 * k ** 2))  # result=1 is simple point

        # 重塑回原始batch和通道维度
        return result.reshape(bs, c)  # 形状 (bs, c)

    def is_edgepoint(self, img):
        # 二值化
        binary_img = (img >= 0.5).float()

        # 创建卷积核
        kernel = torch.tensor([
            [[[1.0, 1.0, 1.0],
              [1.0, 0.0, 1.0],
              [1.0, 1.0, 1.0]]]
        ], device=img.device)  # 形状: (1, 1, 3, 3)

        # # 创建卷积核
        # kernel = torch.tensor([
        #     [[[0.0, 1.0, 0.0],
        #       [1.0, 0.0, 1.0],
        #       [0.0, 1.0, 0.0]]]
        # ], device=img.device)  # 形状: (1, 1, 3, 3)

        sum_neighbors = F.conv2d(binary_img, kernel, padding=1)
        # print(f"img\n{binary_img}")

        # 生成掩码
        # mask_edge = (sum_neighbors > 1).float()
        # mask_edge = torch.sigmoid(10 * (sum_neighbors - 2))
        mask_edge = 1 - F.hardtanh(-(sum_neighbors - 2), min_val=0, max_val=1)  # 1 or fewer neigbors
        # print(f"mask_edge\n{mask_edge}")

        return mask_edge

    def forward(self, img):
        B, C, H, W = img.shape
        u = img.clone()

        for i in range(self.num_iter):

            u_edge = u.view(B*C,1,H,W)
            m_edge = self.is_edgepoint(u_edge)
            # e = m_edge.squeeze().detach().cpu().numpy()
            # plot(e)

            for j in range(4):
                if self.num_class == 1:
                    pick = self.build_order_masks(u)[j]
                    u_unfold = self.patchify(u)
                    a = self.one_process_diff(1.0 - u_unfold, alpha=3, t=0.5, k=0.2).reshape(B, H, W, 1).permute(0, 3,
                                                                                                                 1, 2)
                    # m_edge = self.is_edgepoint(u).reshape(B, 1, H, W)
                    u = (1 - pick * a * m_edge) * u
                    # u = torch.clamp(u, min=0, max=1)
                    # print(f"img:\n {u}")
                    # plot(u.squeeze().detach().numpy())

                else:
                    u1 = u.view(B*C,1,H,W)
                    pick = self.build_order_masks(u1)[j]
                    u_unfold = self.patchify(u1)
                    a = self.one_process_diff(1.0 - u_unfold, alpha=4, t=0.5, k=0.2).reshape(B*C, H, W, 1).permute(0, 3,
                                                                                                                 1, 2)
                    # m_edge = self.is_edgepoint(u1).reshape(B, 1, H, W)
                    u1 = (1 - pick * a * m_edge) * u1
                    # u1 = torch.clamp(u1, min=0, max=1)
                    # u = torch.concat([1 - u1, u1], dim=1)
                    u = u1.view(B,C,H,W)
                    u[:,0,...] = 1 - u[:, 1:C, ...].sum(dim=1)

        return u


u = np.array([
    [6, 2, 0, 2, 2, 0, 3, 0],
    [8, 6, 9, 2, 3, 8, 2, 2],
    [1, 9, 7, 1, 9, 7, 8, 4],
    [2, 3, 9, 7, 8, 9, 3, 0],
    [3, 2, 6, 8, 3, 2, 0, 3],
    [6, 6, 8, 9, 8, 3, 2, 0],
    [3, 7, 6, 1, 9, 7, 9, 1],
    [1, 2, 3, 3, 1, 0, 2, 0]
])
u = u * 0.1
# u[u >= 0.5] = 1
# u[u < 0.5] = 0


data = torch.as_tensor(u, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

sp = Skel_type(1,'cpu',1, 1, 'm')

data = sp(data)

data = data.squeeze().detach().numpy()
# data[data >= 0.5] = 1
# data[data < 0.5] = 0

# 2. 创建自定义颜色映射：0对应黑色，1对应白色
# cmap = LinearSegmentedColormap.from_list('black_to_white', ['black', 'white'])
cmap = 'Blues'
# cmap = 'Reds'

# 3. 绘制热力图
plt.imshow(
    data,
    cmap=cmap,
    interpolation='nearest',
    vmin=0,
    vmax=1
)


# 4. 自定义格式化函数：有小数时保留两位，否则显示整数
def format_value(value):
    if value == int(value):
        return f"{int(value)}"  # 整数不显示小数点
    else:
        return f"{value:.2f}"  # 小数保留两位


# 5. 显示数值标注（使用自定义格式化函数）
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        text = format_value(data[i, j])
        plt.text(j, i, text, ha='center', va='center',
                 color='white' if data[i, j] >= 0.5 else 'black')

# 6. 隐藏坐标轴
plt.axis('off')

# 7. 展示图像
plt.show()

