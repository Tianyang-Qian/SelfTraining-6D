import torch.utils.data as data
from PIL import Image
import os
import cv2
import os.path
import torch
import numpy as np
import torchvision.transforms as transforms
import argparse
import time
import random
import _pickle as cPickle
from lib.transformations import quaternion_from_euler, euler_matrix, random_quaternion, quaternion_matrix
import numpy.ma as ma
import copy
import scipy.misc
import scipy.io as scio

import imgaug.augmenters as iaa

'''
就是取一个矩形
'''
def get_bbox(bbox):
    """ Compute square image crop window. """
    y1, x1, y2, x2 = bbox
    
    img_width = 1024
    img_length = 1280

    # 窗口大小
    window_size = (max(y2-y1, x2-x1) // 40 + 1) * 40
    window_size = min(window_size, 640)

    # 取中心
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
    # row 极值 和 column列极值
    rmin = center[0] - int(window_size / 2)
    rmax = center[0] + int(window_size / 2)
    cmin = center[1] - int(window_size / 2)
    cmax = center[1] + int(window_size / 2)
    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt
    return rmin, rmax, cmin, cmax


'''
对图像处理
计算范数
白点、随机处理
裁剪、采样等等
'''

class PoseDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root, noise_trans):
        if mode == 'train':
            self.path = 'dataset/zigzag/dataset_config/st_real_train_list.txt'
        elif mode == 'test':
            self.path = 'dataset/zigzag/dataset_config/real_test_list.txt'
        elif mode == 'validation':
            self.path = 'dataset/zigzag/dataset_config/real_vali_list.txt'

        self.mode = mode
        self.num_pt = num_pt
        self.root = root
        self.add_noise = add_noise
        self.noise_trans = noise_trans

        self.list = []
        # list存 real_test_list的内容
        input_file = open(self.path)
        while 1:
            input_line = input_file.readline()
            if not input_line:
                break
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            self.list.append(input_line)
        input_file.close()

        self.length = len(self.list)

        self.cam_cx = 379.32687
        self.cam_cy = 509.43720
        self.cam_fx = 1083.09705
        self.cam_fy = 1083.09705
        
        self.cad_model = np.load('dataset/zigzag/dataset_config/cad.npy')

        self.diameter = []
        obj_center = (np.amin(self.cad_model, axis=0) + np.amax(self.cad_model, axis=0)) / 2.0
        obj_cld = self.cad_model - obj_center
        # 最大的范数平方 求多个行 向量的范数;
        obj_diameter = np.amax(np.linalg.norm(obj_cld, axis=1)) * 2
        self.diameter.append(obj_diameter)

        self.xmap = np.array([[i for i in range(1280)] for j in range(1024)])
        self.ymap = np.array([[j for i in range(1280)] for j in range(1024)])
        
        self.img_size = 192
        self.norm_scale = 1000.0
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.noise_img_loc = 0.0
        self.noise_img_scale = 7.0
        self.minimum_num_pt = 50
        self.norm = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.485, 0.485],
                                                            std=[0.229, 0.229, 0.229])])
        # self.symmetry_obj_idx = [0]
        self.symmetry_obj_idx = []
        self.num_pt_mesh = 1000
        print(len(self.list))






    # 噪音添加
    # 方形裁剪
    #

    def __getitem__(self, index):
        img = cv2.imread('{0}/{1}_color.png'.format(self.root, self.list[index]))[:, :, :3]# 只要0 1 2 三个通道  不需要透明度
        img = img[:, :, ::-1]# 倒序
        depth = np.array(cv2.imread('{0}/{1}_depth.png'.format(self.root, self.list[index]), -1))
        mask = np.array(cv2.imread('{0}/{1}_mask.png'.format(self.root, self.list[index]), -1))

        # # random dropout the pixel in mask
        # 遮盖处理   具体是黑白 有点忘了
        # 就是图上多了白点  mad写这么多
        if self.mode == 'train':
            drop_mask = np.ones(mask.shape, dtype=np.uint8) * 255  #全白
            # 将0%到5%的像素用原图大小10%到15%的黑色方块覆盖
            # 反正就是图片上多了一些黑色点
            aug = iaa.CoarseDropout((0.0, 0.05), size_percent=(0.1, 0.15))
            drop_mask = 255 - aug(images=drop_mask)  # 黑图里白点

            mask = drop_mask.astype(np.uint16) + mask.astype(np.uint16)
            mask[np.where(mask>255)] = 255
            mask = mask.astype(np.uint8)


        # 随机变化图片
        if self.add_noise:
            img = self.trancolor(Image.fromarray(np.uint8(img)))
            img = np.array(img)

        # 有些training set 数据是用.pkl 文件存储的。用module cPickle读取非常方便。
        with open('{0}/{1}_label.pkl'.format(self.root, self.list[index]), 'rb') as f:
            label = cPickle.load(f)# 字典
        
        # random select one object
        # train随机选择一张图片 test第一
        if self.mode == 'test':
            idx = 0
        else:        
            idx = random.randint(0, len(label['instance_ids']) - 1)

        # 正方形裁剪
        inst_id = label['instance_ids'][idx]+1# 所有元素＋1   这是个id还是  矩阵？
        rmin, rmax, cmin, cmax = get_bbox(label['bboxes'][idx])



        # sample points
        # 采样？？ 这部分没看懂
        # np.equal实现把label image每个像素的RGB值与某个class的RGB值进行比对，变成RGB bool值。
        mask = np.equal(mask, inst_id)
        # depth图中 大于0 的   和  mask  做 逻辑和
        mask = np.logical_and(mask, depth>0)

        target_r = label['rotations'][idx]
        target_t = label['translations'][idx]


        # choose是矩阵（2行）的第一组      [0]表示行   [1]表示列
        # https://blog.csdn.net/xiezhen_zheng/article/details/81326106?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522167697416516800213044030%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=167697416516800213044030&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~baidu_landing_v2~default-1-81326106-null-null.142^v73^insert_down3,201^v4^add_ask,239^v2^insert_chatgpt&utm_term=a.nonzero&spm=1018.2226.3001.4187
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

        # 这里不知道要干嘛  反正是扩充了choose到num_pt
        # 让前 num列为1 其他为0
        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        elif len(choose) > 0:
            # 后面填充   （前面的值）
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')
        else:
            choose = np.zeros(self.num_pt).astype(np.int32)
                
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]# 行变列
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]#相当于随机的取点
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        pt2 = depth_masked / self.norm_scale
        pt0 = (xmap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (ymap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)# 列拼接

        # cad model points
        model_points = self.cad_model

        # 矩阵相点积   旋转
        target = np.dot(model_points, target_r.T)
        target = np.add(target, target_t)



        # resize cropped image to standard size and adjust 'choose' accordingly
        img = img[rmin:rmax, cmin:cmax, :]
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        crop_w = rmax - rmin
        ratio = self.img_size / crop_w
        col_idx = choose % crop_w
        row_idx = choose // crop_w
        choose = (np.floor(row_idx * ratio) * self.img_size + np.floor(col_idx * ratio)).astype(np.int64)
        choose = np.array([choose])

        # data augmentation
        if self.mode == 'train':
            # point shift
            # 加上噪声
            add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])
            target_t = target_t + add_t

            # point jitter
            add_t = add_t + np.clip(0.001*np.random.randn(cloud.shape[0], 3), -0.005, 0.005)
            cloud = np.add(cloud, add_t)

        img = self.norm(img)
        cloud = cloud.astype(np.float32)

        # position target
        gt_t = target_t
        target_t = target_t - cloud
        target_t = target_t / np.linalg.norm(target_t, axis=1)[:, None]

        target_r = np.dot(model_points, target_r.T)

        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               img, \
               torch.from_numpy(target_t.astype(np.float32)), \
               torch.from_numpy(target_r.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([0]), \
               torch.from_numpy(gt_t.astype(np.float32))
    
    def __len__(self):
        return self.length
    
    def get_sym_list(self):
        return self.symmetry_obj_idx

    def get_num_points_mesh(self):
        return self.num_pt_mesh
    
    def get_diameter(self):
        return self.diameter


if __name__ == '__main__':
    dataset = PoseDataset('train', num_pt=1024, add_noise=True, root='./data_0201', noise_trans=0.01, refine=False)
    data = dataset.__getitem__(0)
    print(data[0].shape)
    print(data[1].shape)
    print(data[2].shape)
    print(data[3].shape)
    print(data[4].shape)
