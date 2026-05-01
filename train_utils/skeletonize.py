import torch
import torch.nn as nn
import torch.nn.functional as F


class Skel_type(nn.Module):
    """
    Args:
        'm': morphology (cvpr2021)
        's': simple point (ours)
        'b': gradient skel (ICCV2023)
    Returns:
        skeleton
    """
    def __init__(self, num_iter, device, num_classes, half_kersize=1, skel_type='m'):
        super(Skel_type, self).__init__()
        self.type = skel_type
        if self.type == 'b':
            self.model = skel_be(probabilistic=True, simple_point_detection='EulerCharacteristic', num_iter=num_iter)
        elif self.type == 'm':
            self.model = MorphLayer(device, num_classes, half_kersize, num_iter=num_iter)
        elif self.type == 's':
            self.model = SimplepointLayer(half_kersize, device, num_classes, num_iter=num_iter)

    def forward(self, img):

        if self.type == 'b':
            skel = self.model(img)
        elif self.type == 'm':
            skel = self.model.skel(img)
        elif self.type == 's':
            skel = self.model(img)

        return skel


class SimplepointLayer(nn.Module):
    def __init__(self, half_size, device, num_classes, num_iter):
        super(SimplepointLayer, self).__init__()

        self.unfold = nn.Unfold(kernel_size=2 * half_size + 1, padding=half_size, stride=1)
        self.device = device
        self.num_class = num_classes
        self.num_iter = num_iter
        self.morph = MorphLayer(device, num_classes, half_size)

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
        top_mask = pattern[..., 0:1, 1:2] > 0.5  # 形状: [B, N, 1, 1]
        n2[..., 0:1, :] = top_mask.expand_as(n2[..., 0:1, :])  # 扩展并赋值第一行

        # 下边界：检查每个模式的(2,1)位置
        bottom_mask = pattern[..., 2:3, 1:2] > 0.5
        n2[..., 2:3, :] = bottom_mask.expand_as(n2[..., 2:3, :])  # 扩展并赋值最后一行

        # 左边界：检查每个模式的(1,0)位置
        left_mask = pattern[..., 1:2, 0:1] > 0.5
        n2[..., :, 0:1] = left_mask.expand_as(n2[..., :, 0:1])  # 扩展并赋值第一列

        # 右边界：检查每个模式的(1,2)位置
        right_mask = pattern[..., 1:2, 2:3] > 0.5
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
        mask1 = (x4[:, :8] <= 0.5) & (x4[:, 1:] > 0.5)
        # 条件2: (x4[:, i] >= 0.5) & (x4[:, i+1] < 0.5)
        mask2 = (x4[:, :8] > 0.5) & (x4[:, 1:] <= 0.5)

        # 应用条件覆盖
        result4[mask1] = 1
        result4[mask2] = -1

        # 步骤6: 应用sigmoid变换和阈值处理
        result4 = torch.sigmoid((torch.abs(result4) - t) * (alpha ** 2))

        # 步骤7: 求和并处理零值 (向量化)
        result4_sum = result4.sum(dim=1)/2.0  # 形状 (bs*c,)

        # 步骤8: 应用最终变换
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

        mask_edge = 1 - F.hardtanh(-(sum_neighbors - 2), min_val=0, max_val=1)

        return mask_edge

    def forward(self, img):
        B, C, H, W = img.shape
        u = img.clone()

        for i in range(self.num_iter):

            u_edge = u.view(B*C,1,H,W)
            m_edge = self.is_edgepoint(u_edge)

            for j in range(4):
                if self.num_class == 1:
                    pick = self.build_order_masks(u)[j]
                    u_unfold = self.patchify(u)
                    a = self.one_process_diff(1.0 - u_unfold, alpha=4, t=0.5, k=0.2).reshape(B, H, W, 1).permute(0, 3,
                                                                                                                 1, 2)
                    u = (1 - pick * a * m_edge) * u

                else:
                    u1 = u.view(B*C,1,H,W)
                    pick = self.build_order_masks(u1)[j]
                    u_unfold = self.patchify(u1)
                    a = self.one_process_diff(1.0 - u_unfold, alpha=4, t=0.5, k=0.2).reshape(B*C, H, W, 1).permute(0, 3,
                                                                                                                 1, 2)
                    u1 = (1 - pick * a * m_edge) * u1
                    u = u1.view(B,C,H,W)
                    u[:,0,...] = 1 - u[:, 1:C, ...].sum(dim=1)

        return u


class MorphLayer(nn.Module):
    def __init__(self, device, in_channel, half_kersize=1, num_iter=15):
        super(MorphLayer, self).__init__()

        self.c = in_channel
        self.num_iter = num_iter
        self.half_kernel = half_kersize
        self.kernel_size = 2 * half_kersize + 1
        self.padding = half_kersize
        self.trans = nn.MaxPool2d(2 * half_kersize + 1, 1, half_kersize)
        self.avgpool = nn.AvgPool2d(2 * half_kersize + 1, 1, half_kersize, count_include_pad=False, divisor_override=1)
        self.alpha = nn.Parameter(torch.FloatTensor([5.0]), requires_grad=True).to(device)
        self.beta = nn.Parameter(torch.FloatTensor([5.0]), requires_grad=True).to(device)

    def connectivity_matrix(self, img, conn_type):

        conn = F.unfold(img, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        conn = conn.view(img.size(0), img.size(1), self.kernel_size * self.kernel_size, img.size(2),
                         img.size(3)).permute(0, 1, 3, 4, 2)
        if conn_type == 4:
            indices = torch.tensor([1, 3, 4, 5, 7], device=img.device)
            conn = torch.index_select(conn, dim=-1, index=indices)

        return conn

    def erode(self, unary, is_soft=False, conn_type=8):
        if is_soft:
            unfolded = self.connectivity_matrix(unary, conn_type=8)

            erode = -torch.logsumexp(-unfolded * (self.beta ** 2), dim=-1) / (self.beta ** 2)

        else:
            erode = -self.trans(-unary)

        return erode

    def dilate(self, unary, is_soft=False, conn_type=8):
        if is_soft:
            unfolded = self.connectivity_matrix(unary, conn_type)

            dilate = torch.logsumexp(unfolded * (self.beta ** 2), dim=-1) / (self.beta ** 2)

        else:
            dilate = self.trans(unary)

        return dilate

    def open(self, unary, is_soft=False, conn_type=8):

        return self.dilate(self.erode(unary, is_soft), is_soft)

    def skel(self, unary, is_soft=False, conn_type=8):

        skel = unary - self.open(unary, is_soft)

        for j in range(self.num_iter*2):
            unary = self.erode(unary, is_soft)
            img1 = self.open(unary, is_soft)
            delta = unary - img1
            skel = skel + delta

        if self.c > 1:
            # 计算第1通道到最后一个通道的和，并取补赋值给第0通道
            skel[:, 0, ...] = 1 - skel[:, 1:self.c, ...].sum(dim=1)

        return skel

class skel_be(torch.nn.Module):
    """
    Class based on PyTorch's Module class to skeletonize two- or three-dimensional input images
    while being fully compatible with PyTorch's autograd automatic differention engine as proposed in [1].

    Attributes:
        propabilistic: a Boolean that indicates whether the input image should be binarized using
                       the reparametrization trick and straight-through estimator.
                       It should always be set to True if non-binary inputs are being provided.
        beta: scale of added logistic noise during the reparametrization trick. If too small, there will not be any learning via
              gradient-based optimization; if too large, the learning is very slow.
        tau: Boltzmann temperature for reparametrization trick.
        simple_point_detection: decides whether simple points should be identified using Boolean characterization of their 26-neighborhood (Boolean) [2]
                                or by checking whether the Euler characteristic changes under their deletion (EulerCharacteristic) [3].
        num_iter: number of iterations that each include one end-point check, eight checks for simple points and eight subsequent deletions.
                  The number of iterations should be tuned to the type of input image.

    [1] Martin J. Menten et al. A skeletonization algorithm for gradient-based optimization.
        Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023.
    [2] Gilles Bertrand. A boolean characterization of three- dimensional simple points.
        Pattern recognition letters, 17(2):115-124, 1996.
    [3] Steven Lobregt et al. Three-dimensional skeletonization:principle and algorithm.
        IEEE Transactions on pattern analysis and machine intelligence, 2(1):75-77, 1980.
    """

    def __init__(self, probabilistic=True, beta=0.1, tau=1.0, simple_point_detection='Boolean', num_iter=5):

        super(skel_be, self).__init__()

        self.probabilistic = probabilistic
        self.tau = tau
        self.beta = beta

        self.num_iter = num_iter
        self.endpoint_check = self._single_neighbor_check
        if simple_point_detection == 'Boolean':
            self.simple_check = self._boolean_simple_check
        elif simple_point_detection == 'EulerCharacteristic':
            self.simple_check = self._euler_characteristic_simple_check
        else:
            raise Exception()

    def forward(self, img):

        img = self._prepare_input(img)

        if self.probabilistic:
            img = self._stochastic_discretization(img)

        for current_iter in range(self.num_iter):

            # At each iteration create a new map of the end-points
            is_endpoint = self.endpoint_check(img)
            # print(f"image: \n{img[:,:,1,:,:]}")
            # print(f"is_endpoint: \n{1 - is_endpoint[:,:,1,1:7,1:7]}")

            # Sub-iterate through eight different subfields
            x_offsets = [0, 1, 0, 1, 0, 1, 0, 1]
            y_offsets = [0, 0, 1, 1, 0, 0, 1, 1]
            z_offsets = [0, 0, 0, 0, 1, 1, 1, 1]

            for x_offset, y_offset, z_offset in zip(x_offsets, y_offsets, z_offsets):
                # At each sub-iteration detect all simple points and delete all simple points that are not end-points
                is_simple = self.simple_check(img[:, :, x_offset:, y_offset:, z_offset:])
                deletion_candidates = is_simple * (1 - is_endpoint[:, :, x_offset::2, y_offset::2, z_offset::2])
                img[:, :, x_offset::2, y_offset::2, z_offset::2] = torch.min(
                    img[:, :, x_offset::2, y_offset::2, z_offset::2].clone(), 1 - deletion_candidates)
                # print(f"img: \n{img[:,:,1,1:7,1:7]}")

        img = self._prepare_output(img)

        return img

    def _prepare_input(self, img):
        """
        Function to check that the input image is compatible with the subsequent calculations.
        Only two- and three-dimensional images with values between 0 and 1 are supported.
        If the input image is two-dimensional then it is converted into a three-dimensional one for further processing.
        """

        if img.dim() == 5:
            self.expanded_dims = False
        elif img.dim() == 4:
            self.expanded_dims = True
            img = img.unsqueeze(2)
        else:
            raise Exception(
                "Only two-or three-dimensional images (tensor dimensionality of 4 or 5) are supported as input.")

        if img.shape[2] == 2 or img.shape[3] == 2 or img.shape[4] == 2 or img.shape[3] == 1 or img.shape[4] == 1:
            raise Exception()

        if img.min() < 0.0 or img.max() > 1.0:
            raise Exception("Image values must lie between 0 and 1.")

        img = F.pad(img, (1, 1, 1, 1, 1, 1), value=0)

        return img

    def _stochastic_discretization(self, img):
        """
        Function to binarize the image so that it can be processed by our skeletonization method.
        In order to remain compatible with backpropagation we utilize the reparameterization trick and a straight-through estimator.
        """

        alpha = (img + 1e-8) / (1.0 - img + 1e-8)

        uniform_noise = torch.rand_like(img)
        uniform_noise = torch.empty_like(img).uniform_(1e-8, 1 - 1e-8)
        logistic_noise = (torch.log(uniform_noise) - torch.log(1 - uniform_noise))

        img = torch.sigmoid((torch.log(alpha) + logistic_noise * self.beta) / self.tau)
        img = (img.detach() > 0.5).float() - img.detach() + img

        return img

    def _single_neighbor_check(self, img):
        """
        Function that characterizes points as endpoints if they have a single neighbor or no neighbor at all.
        """

        img = F.pad(img, (1, 1, 1, 1, 1, 1))

        # Check that number of ones in twentysix-neighborhood is exactly 0 or 1
        K = torch.tensor([[[1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0]],
                          [[1.0, 1.0, 1.0],
                           [1.0, 0.0, 1.0],
                           [1.0, 1.0, 1.0]],
                          [[1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0],
                           [1.0, 1.0, 1.0]]], device=img.device).view(1, 1, 3, 3, 3)

        num_twentysix_neighbors = F.conv3d(img, K)
        condition1 = F.hardtanh(-(num_twentysix_neighbors - 2), min_val=0, max_val=1)  # 1 or fewer neigbors

        return condition1

    def _boolean_simple_check(self, img):
        """
        Function that identifies simple points using Boolean conditions introduced by Bertrand et al. [1].
        Each Boolean conditions can be assessed via convolutions with a limited number of pre-defined kernels.
        It total, four conditions are checked. If any one is fulfilled, the point is deemed simple.

        [1] Gilles Bertrand. A boolean characterization of three- dimensional simple points.
            Pattern recognition letters, 17(2):115-124, 1996.
        """

        img = F.pad(img, (1, 1, 1, 1, 1, 1), value=0)

        # Condition 1: number of zeros in the six-neighborhood is exactly 1
        K_N6 = torch.tensor([[[0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0]],
                             [[0.0, 1.0, 0.0],
                              [1.0, 0.0, 1.0],
                              [0.0, 1.0, 0.0]],
                             [[0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0]]], device=img.device).view(1, 1, 3, 3, 3)

        num_six_neighbors = F.conv3d(1 - img, K_N6, stride=2)

        subcondition1a = F.hardtanh(num_six_neighbors, min_val=0, max_val=1)  # 1 or more neighbors
        subcondition1b = F.hardtanh(-(num_six_neighbors - 2), min_val=0, max_val=1)  # 1 or fewer neighbors

        condition1 = subcondition1a * subcondition1b

        # Condition 2: number of ones in twentysix-neighborhood is exactly 1
        K_N26 = torch.tensor([[[1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0]],
                              [[1.0, 1.0, 1.0],
                               [1.0, 0.0, 1.0],
                               [1.0, 1.0, 1.0]],
                              [[1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0],
                               [1.0, 1.0, 1.0]]], device=img.device).view(1, 1, 3, 3, 3)

        num_twentysix_neighbors = F.conv3d(img, K_N26, stride=2)

        subcondition2a = F.hardtanh(num_twentysix_neighbors, min_val=0, max_val=1)  # 1 or more neighbors
        subcondition2b = F.hardtanh(-(num_twentysix_neighbors - 2), min_val=0, max_val=1)  # 1 or fewer neigbors

        condition2 = subcondition2a * subcondition2b

        # Condition 3: Number of ones in eighteen-neigborhood exactly 1...
        K_N18 = torch.tensor([[[0.0, 1.0, 0.0],
                               [1.0, 1.0, 1.0],
                               [0.0, 1.0, 0.0]],
                              [[1.0, 1.0, 1.0],
                               [1.0, 0.0, 1.0],
                               [1.0, 1.0, 1.0]],
                              [[0.0, 1.0, 0.0],
                               [1.0, 1.0, 1.0],
                               [0.0, 1.0, 0.0]]], device=img.device).view(1, 1, 3, 3, 3)

        num_eighteen_neighbors = F.conv3d(img, K_N18, stride=2)

        subcondition3a = F.hardtanh(num_eighteen_neighbors, min_val=0, max_val=1)  # 1 or more neighbors
        subcondition3b = F.hardtanh(-(num_eighteen_neighbors - 2), min_val=0, max_val=1)  # 1 or fewer neigbors

        # ... and cell configration B26 does not exist
        K_B26 = torch.tensor([[[1.0, -1.0, 0.0],
                               [-1.0, -1.0, 0.0],
                               [0.0, 0.0, 0.0]],
                              [[-1.0, -1.0, 0.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]]], device=img.device).view(1, 1, 3, 3, 3)

        B26_1_present = F.relu(F.conv3d(2.0 * img - 1.0, K_B26, stride=2) - 6)
        B26_2_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[2]), stride=2) - 6)
        B26_3_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[3]), stride=2) - 6)
        B26_4_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[4]), stride=2) - 6)
        B26_5_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[2, 3]), stride=2) - 6)
        B26_6_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[2, 4]), stride=2) - 6)
        B26_7_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[3, 4]), stride=2) - 6)
        B26_8_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[2, 3, 4]), stride=2) - 6)
        num_B26_cells = B26_1_present + B26_2_present + B26_3_present + B26_4_present + B26_5_present + B26_6_present + B26_7_present + B26_8_present

        subcondition3c = F.hardtanh(-(num_B26_cells - 1), min_val=0, max_val=1)

        condition3 = subcondition3a * subcondition3b * subcondition3c

        # Condition 4: cell configuration A6 does not exist...
        K_A6 = torch.tensor([[[0.0, 1.0, 0.0],
                              [1.0, -1.0, 1.0],
                              [0.0, 1.0, 0.0]],
                             [[0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0]],
                             [[0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0]]], device=img.device).view(1, 1, 3, 3, 3)

        A6_1_present = F.relu(F.conv3d(2.0 * img - 1.0, K_A6, stride=2) - 4)
        A6_2_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_A6, dims=[2, 3]), stride=2) - 4)
        A6_3_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_A6, dims=[2, 4]), stride=2) - 4)
        A6_4_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_A6, dims=[2]), stride=2) - 4)
        A6_5_present = F.relu(
            F.conv3d(2.0 * img - 1.0, torch.rot90(torch.flip(K_A6, dims=[2]), dims=[2, 3]), stride=2) - 4)
        A6_6_present = F.relu(
            F.conv3d(2.0 * img - 1.0, torch.rot90(torch.flip(K_A6, dims=[2]), dims=[2, 4]), stride=2) - 4)
        num_A6_cells = A6_1_present + A6_2_present + A6_3_present + A6_4_present + A6_5_present + A6_6_present

        subcondition4a = F.hardtanh(-(num_A6_cells - 1), min_val=0, max_val=1)

        # ... and cell configuration B26 does not exist...
        K_B26 = torch.tensor([[[1.0, -1.0, 0.0],
                               [-1.0, -1.0, 0.0],
                               [0.0, 0.0, 0.0]],
                              [[-1.0, -1.0, 0.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]]], device=img.device).view(1, 1, 3, 3, 3)

        B26_1_present = F.relu(F.conv3d(2.0 * img - 1.0, K_B26, stride=2) - 6)
        B26_2_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[2]), stride=2) - 6)
        B26_3_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[3]), stride=2) - 6)
        B26_4_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[4]), stride=2) - 6)
        B26_5_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[2, 3]), stride=2) - 6)
        B26_6_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[2, 4]), stride=2) - 6)
        B26_7_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[3, 4]), stride=2) - 6)
        B26_8_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_B26, dims=[2, 3, 4]), stride=2) - 6)
        num_B26_cells = B26_1_present + B26_2_present + B26_3_present + B26_4_present + B26_5_present + B26_6_present + B26_7_present + B26_8_present

        subcondition4b = F.hardtanh(-(num_B26_cells - 1), min_val=0, max_val=1)

        # ... and cell configuration B18 does not exist...
        K_B18 = torch.tensor([[[0.0, 1.0, 0.0],
                               [-1.0, -1.0, -1.0],
                               [0.0, 0.0, 0.0]],
                              [[-1.0, -1.0, -1.0],
                               [-1.0, 0.0, -1.0],
                               [0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]]], device=img.device).view(1, 1, 3, 3, 3)

        B18_1_present = F.relu(F.conv3d(2.0 * img - 1.0, K_B18, stride=2) - 8)
        B18_2_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_B18, dims=[2, 4]), stride=2) - 8)
        B18_3_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_B18, dims=[2, 4], k=2), stride=2) - 8)
        B18_4_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_B18, dims=[2, 4], k=3), stride=2) - 8)
        B18_5_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_B18, dims=[3, 4]), stride=2) - 8)
        B18_6_present = F.relu(
            F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_B18, dims=[3, 4]), dims=[2, 4]), stride=2) - 8)
        B18_7_present = F.relu(
            F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_B18, dims=[3, 4]), dims=[2, 4], k=2), stride=2) - 8)
        B18_8_present = F.relu(
            F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_B18, dims=[3, 4]), dims=[2, 4], k=3), stride=2) - 8)
        B18_9_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_B18, dims=[3, 4], k=2), stride=2) - 8)
        B18_10_present = F.relu(
            F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_B18, dims=[3, 4], k=2), dims=[2, 4]), stride=2) - 8)
        B18_11_present = F.relu(
            F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_B18, dims=[3, 4], k=2), dims=[2, 4], k=2),
                     stride=2) - 8)
        B18_12_present = F.relu(
            F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_B18, dims=[3, 4], k=2), dims=[2, 4], k=3),
                     stride=2) - 8)
        num_B18_cells = B18_1_present + B18_2_present + B18_3_present + B18_4_present + B18_5_present + B18_6_present + B18_7_present + B18_8_present + B18_9_present + B18_10_present + B18_11_present + B18_12_present

        subcondition4c = F.hardtanh(-(num_B18_cells - 1), min_val=0, max_val=1)

        # ... and the number of zeros in the six-neighborhood minus the number of A18 cell configurations plus the number of A26 cell configurations is exactly one
        K_N6 = torch.tensor([[[0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0]],
                             [[0.0, 1.0, 0.0],
                              [1.0, 0.0, 1.0],
                              [0.0, 1.0, 0.0]],
                             [[0.0, 0.0, 0.0],
                              [0.0, 1.0, 0.0],
                              [0.0, 0.0, 0.0]]], device=img.device).view(1, 1, 3, 3, 3)

        num_six_neighbors = F.conv3d(1 - img, K_N6, stride=2)

        K_A18 = torch.tensor([[[0.0, -1.0, 0.0],
                               [0.0, -1.0, 0.0],
                               [0.0, 0.0, 0.0]],
                              [[0.0, -1.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]]], device=img.device).view(1, 1, 3, 3, 3)

        A18_1_present = F.relu(F.conv3d(2.0 * img - 1.0, K_A18, stride=2) - 2)
        A18_2_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_A18, dims=[2, 4]), stride=2) - 2)
        A18_3_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_A18, dims=[2, 4], k=2), stride=2) - 2)
        A18_4_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_A18, dims=[2, 4], k=3), stride=2) - 2)
        A18_5_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_A18, dims=[3, 4]), stride=2) - 2)
        A18_6_present = F.relu(
            F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_A18, dims=[3, 4]), dims=[2, 4]), stride=2) - 2)
        A18_7_present = F.relu(
            F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_A18, dims=[3, 4]), dims=[2, 4], k=2), stride=2) - 2)
        A18_8_present = F.relu(
            F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_A18, dims=[3, 4]), dims=[2, 4], k=3), stride=2) - 2)
        A18_9_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.rot90(K_A18, dims=[3, 4], k=2), stride=2) - 2)
        A18_10_present = F.relu(
            F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_A18, dims=[3, 4], k=2), dims=[2, 4]), stride=2) - 2)
        A18_11_present = F.relu(
            F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_A18, dims=[3, 4], k=2), dims=[2, 4], k=2),
                     stride=2) - 2)
        A18_12_present = F.relu(
            F.conv3d(2.0 * img - 1.0, torch.rot90(torch.rot90(K_A18, dims=[3, 4], k=2), dims=[2, 4], k=3),
                     stride=2) - 2)
        num_A18_cells = A18_1_present + A18_2_present + A18_3_present + A18_4_present + A18_5_present + A18_6_present + A18_7_present + A18_8_present + A18_9_present + A18_10_present + A18_11_present + A18_12_present

        K_A26 = torch.tensor([[[-1.0, -1.0, 0.0],
                               [-1.0, -1.0, 0.0],
                               [0.0, 0.0, 0.0]],
                              [[-1.0, -1.0, 0.0],
                               [-1.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]],
                              [[0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0]]], device=img.device).view(1, 1, 3, 3, 3)

        A26_1_present = F.relu(F.conv3d(2.0 * img - 1.0, K_A26, stride=2) - 6)
        A26_2_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_A26, dims=[2]), stride=2) - 6)
        A26_3_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_A26, dims=[3]), stride=2) - 6)
        A26_4_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_A26, dims=[4]), stride=2) - 6)
        A26_5_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_A26, dims=[2, 3]), stride=2) - 6)
        A26_6_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_A26, dims=[2, 4]), stride=2) - 6)
        A26_7_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_A26, dims=[3, 4]), stride=2) - 6)
        A26_8_present = F.relu(F.conv3d(2.0 * img - 1.0, torch.flip(K_A26, dims=[2, 3, 4]), stride=2) - 6)
        num_A26_cells = A26_1_present + A26_2_present + A26_3_present + A26_4_present + A26_5_present + A26_6_present + A26_7_present + A26_8_present

        subcondition4d = F.hardtanh(num_six_neighbors - num_A18_cells + num_A26_cells, min_val=0,
                                    max_val=1)  # 1 or more configurations
        subcondition4e = F.hardtanh(-(num_six_neighbors - num_A18_cells + num_A26_cells - 2), min_val=0,
                                    max_val=1)  # 1 or fewer configurations

        condition4 = subcondition4a * subcondition4b * subcondition4c * subcondition4d * subcondition4e

        # If any of the four conditions is fulfilled the point is simple
        combined = torch.cat([condition1, condition2, condition3, condition4], dim=1)
        is_simple = torch.amax(combined, dim=1, keepdim=True)

        return is_simple

    # Specifically designed to be used with the eight-subfield iterative scheme from above.
    def _euler_characteristic_simple_check(self, img):
        """
        Function that identifies simple points by assessing whether the Euler characteristic changes when deleting it [1].
        In order to calculate the Euler characteristic, the amount of vertices, edges, faces and octants are counted using convolutions with pre-defined kernels.
        The function is meant to be used in combination with the subfield-based iterative scheme employed in the forward function.

        [1] Steven Lobregt et al. Three-dimensional skeletonization:principle and algorithm.
            IEEE Transactions on pattern analysis and machine intelligence, 2(1):75-77, 1980.
        """

        img = F.pad(img, (1, 1, 1, 1, 1, 1), value=0)

        # Create masked version of the image where the center of 26-neighborhoods is changed to zero
        mask = torch.ones_like(img)
        mask[:, :, 1::2, 1::2, 1::2] = 0
        masked_img = img.clone() * mask

        # m = (masked_img != 0) & (masked_img != 1)
        # indices = torch.where(m)
        # print("满足条件的位置索引:", indices)
        # values = masked_img[m]
        # print("满足条件的值:", values)

        # Count vertices
        vertices = F.relu(-(2.0 * img - 1.0))
        num_vertices = F.avg_pool3d(vertices, (3, 3, 3), stride=2) * 27

        masked_vertices = F.relu(-(2.0 * masked_img - 1.0))
        num_masked_vertices = F.avg_pool3d(masked_vertices, (3, 3, 3), stride=2) * 27

        # Count edges
        K_ud_edge = torch.tensor([0.5, 0.5], device=img.device).view(1, 1, 2, 1, 1)
        K_ns_edge = torch.tensor([0.5, 0.5], device=img.device).view(1, 1, 1, 2, 1)
        K_we_edge = torch.tensor([0.5, 0.5], device=img.device).view(1, 1, 1, 1, 2)

        ud_edges = F.relu(F.conv3d(-(2.0 * img - 1.0), K_ud_edge))
        num_ud_edges = F.avg_pool3d(ud_edges, (2, 3, 3), stride=2) * 18
        ns_edges = F.relu(F.conv3d(-(2.0 * img - 1.0), K_ns_edge))
        num_ns_edges = F.avg_pool3d(ns_edges, (3, 2, 3), stride=2) * 18
        we_edges = F.relu(F.conv3d(-(2.0 * img - 1.0), K_we_edge))
        num_we_edges = F.avg_pool3d(we_edges, (3, 3, 2), stride=2) * 18
        num_edges = num_ud_edges + num_ns_edges + num_we_edges

        masked_ud_edges = F.relu(F.conv3d(-(2.0 * masked_img - 1.0), K_ud_edge))
        num_masked_ud_edges = F.avg_pool3d(masked_ud_edges, (2, 3, 3), stride=2) * 18
        masked_ns_edges = F.relu(F.conv3d(-(2.0 * masked_img - 1.0), K_ns_edge))
        num_masked_ns_edges = F.avg_pool3d(masked_ns_edges, (3, 2, 3), stride=2) * 18
        masked_we_edges = F.relu(F.conv3d(-(2.0 * masked_img - 1.0), K_we_edge))
        num_masked_we_edges = F.avg_pool3d(masked_we_edges, (3, 3, 2), stride=2) * 18
        num_masked_edges = num_masked_ud_edges + num_masked_ns_edges + num_masked_we_edges

        # Count faces
        K_ud_face = torch.tensor([[0.25, 0.25], [0.25, 0.25]], device=img.device).view(1, 1, 1, 2, 2)
        K_ns_face = torch.tensor([[0.25, 0.25], [0.25, 0.25]], device=img.device).view(1, 1, 2, 1, 2)
        K_we_face = torch.tensor([[0.25, 0.25], [0.25, 0.25]], device=img.device).view(1, 1, 2, 2, 1)

        ud_faces = F.relu(F.conv3d(-(2.0 * img - 1.0), K_ud_face) - 0.5) * 2
        num_ud_faces = F.avg_pool3d(ud_faces, (3, 2, 2), stride=2) * 12
        ns_faces = F.relu(F.conv3d(-(2.0 * img - 1.0), K_ns_face) - 0.5) * 2
        num_ns_faces = F.avg_pool3d(ns_faces, (2, 3, 2), stride=2) * 12
        we_faces = F.relu(F.conv3d(-(2.0 * img - 1.0), K_we_face) - 0.5) * 2
        num_we_faces = F.avg_pool3d(we_faces, (2, 2, 3), stride=2) * 12
        num_faces = num_ud_faces + num_ns_faces + num_we_faces

        masked_ud_faces = F.relu(F.conv3d(-(2.0 * masked_img - 1.0), K_ud_face) - 0.5) * 2
        num_masked_ud_faces = F.avg_pool3d(masked_ud_faces, (3, 2, 2), stride=2) * 12
        masked_ns_faces = F.relu(F.conv3d(-(2.0 * masked_img - 1.0), K_ns_face) - 0.5) * 2
        num_masked_ns_faces = F.avg_pool3d(masked_ns_faces, (2, 3, 2), stride=2) * 12
        masked_we_faces = F.relu(F.conv3d(-(2.0 * masked_img - 1.0), K_we_face) - 0.5) * 2
        num_masked_we_faces = F.avg_pool3d(masked_we_faces, (2, 2, 3), stride=2) * 12
        num_masked_faces = num_masked_ud_faces + num_masked_ns_faces + num_masked_we_faces

        # Count octants
        K_octants = torch.tensor([[[0.125, 0.125], [0.125, 0.125]], [[0.125, 0.125], [0.125, 0.125]]],
                                 device=img.device).view(1, 1, 2, 2, 2)

        octants = F.relu(F.conv3d(-(2.0 * img - 1.0), K_octants) - 0.75) * 4
        num_octants = F.avg_pool3d(octants, (2, 2, 2), stride=2) * 8

        masked_octants = F.relu(F.conv3d(-(2.0 * masked_img - 1.0), K_octants) - 0.75) * 4
        num_masked_octants = F.avg_pool3d(masked_octants, (2, 2, 2), stride=2) * 8

        # Combined number of vertices, edges, faces and octants to calculate the euler characteristic
        euler_characteristic = num_vertices - num_edges + num_faces - num_octants
        masked_euler_characteristic = num_masked_vertices - num_masked_edges + num_masked_faces - num_masked_octants

        # If the Euler characteristic is unchanged after switching a point from 1 to 0 this indicates that the point is simple
        euler_change = F.hardtanh(torch.abs(masked_euler_characteristic - euler_characteristic), min_val=0, max_val=1)
        is_simple = 1 - euler_change
        is_simple = (is_simple.detach() > 0.5).float() - is_simple.detach() + is_simple

        return is_simple

    def _prepare_output(self, img):
        """
        Function that removes the padding and dimensions added by _prepare_input function.
        """

        img = img[:, :, 1:-1, 1:-1, 1:-1]

        if self.expanded_dims:
            img = torch.squeeze(img, dim=2)

        return img


if __name__ == '__main__':
    device = 'cpu'
    num_classes = 1
    u = torch.ones((1, num_classes, 5, 5))
    # u[0, 0, 2, 2] = 0
    # u[0, 0, 0, 1] = 0.
    # u[0, 0, 0, 2] = 1
    # u[0, 0, 1, 0] = 0
    # u[0, 0, 1, 1] = 1
    # u[0, 0, 1, 2] = 0
    # u[0, 0, 0, 0:5] = 0
    # u[0, 0, 5, 0:5] = 0
    # u[0, 0, 0:5, 5] = 0
    # u[0, 0, 0:5, 0] = 0

    sp = Skel_type(3, device, num_classes, 1, 's')

    print(u)
    # sk = skel_be(probabilistic=False, simple_point_detection='EulerCharacteristic', num_iter=5)
    # sp = SimplepointLayer(1, device, 1, 5)
    # e1 = sk.endpoint_check(u)
    # e2 = sp.is_edgepoint(u)
    # print(e1)
    # print(e2)
    r1 = sp(u)
    # r2 = sp(u)
    # r2 = torch.clamp(r2, 0, 1)
    # print(r1)
    print(torch.round(r1))
