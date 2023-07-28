# import open3d as o3d
import os
import glob
import cv2
import time
import argparse
from PIL import Image
import numpy as np
import copy
import _pickle as cPickle
import numpy.ma as ma
import scipy.io as scio
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from lib.transformations import quaternion_matrix
from lib.network import PoseNet
from lib.ransac_voting.ransac_voting_gpu import ransac_voting_layer
from lib.utils import load_obj, uniform_sample, transform_coordinates_3d

from lib.knn.__init__ import KNearestNeighbor
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from renderer.rendering import *
from perceptual_loss.alexnet_perception import *


def get_bbox(bbox, h, w):
    """ Compute square image crop window. """
    y1, x1, y2, x2 = bbox
    h = 1024
    w = 1280
    window_size = (max(y2 - y1, x2 - x1) // 40 + 1) * 40
    window_size = min(window_size, 840)
    center = [(y1 + y2) // 2, (x1 + x2) // 2]
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
    if rmax > h:
        delt = rmax - h
        rmax = h
        rmin -= delt
    if cmax > w:
        delt = cmax - w
        cmax = w
        cmin -= delt
    return rmin, rmax, cmin, cmax


# posenet img\depth\current mask? 采样吗\相机参数、ZIGZAG顶点和面信息
"""
对图像中的一个物体进行某种姿势估计。该函数首先使用列表理解和range()函数创建了两个名为xmap和ymap的2D数组。
然后，它从一个物体掩码中提取一个边界框，并使用它来裁剪图像和深度图。然后，裁剪后的图像被调整为标准尺寸，并被归一化。
裁剪后的深度图被用来创建一个点云，然后与正常化的图像一起被用作神经网络的输入。神经网络输出旋转和平移矢量，然后用来计算物体的预测姿势。
如果姿势估计失败，该函数返回none。
输出看最后
"""
def label_individual_pose(estimator, scene_img, scene_depth, obj_mask, intrinsics, cad_model_pcs, cad_model_faces,
                          visualize=True):
    norm = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.485, 0.485],
                                                    std=[0.229, 0.229, 0.229])])
    h, w = obj_mask.shape

    xmap = np.array([[i for i in range(w)] for j in range(h)])  # 2维数组[[1 2 3], [1 2 3]]类似的
    ymap = np.array([[j for i in range(w)] for j in range(h)])

    bbox = obj_mask.flatten().nonzero()[0]
    if not len(bbox) > 0:  # 全黑返回000
        return None, None, None
    ys = bbox // w
    xs = bbox - ys * w
    bbox = [np.min(ys), np.min(xs), np.max(ys), np.max(xs)]
    rmin, rmax, cmin, cmax = get_bbox(bbox, h, w)

    choose = obj_mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
    model_pc = uniform_sample(cad_model_pcs, cad_model_faces, 1000)

    predict_sRT = None
    num_points = 1000# 采样点云数
    depth_scale = 1000.0
    if len(choose) > num_points / 10:
        if len(choose) > num_points:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:num_points] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, num_points - len(choose)), 'wrap')

        depth_masked = scene_depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        xmap_masked = xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        ymap_masked = ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis]
        pt2 = depth_masked / depth_scale
        pt0 = (xmap_masked - intrinsics[0, 2]) * pt2 / intrinsics[0, 0]
        pt1 = (ymap_masked - intrinsics[1, 2]) * pt2 / intrinsics[1, 1]
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)

        # resize cropped image to standard size and adjust 'choose' accordingly
        img_size = 192
        img_masked = copy.deepcopy(scene_img[rmin:rmax, cmin:cmax, :])
        img_masked = cv2.resize(img_masked, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        crop_w = rmax - rmin
        ratio = img_size / crop_w
        col_idx = choose % crop_w
        row_idx = choose // crop_w
        choose = (np.floor(row_idx * ratio) * img_size + np.floor(col_idx * ratio)).astype(np.int64)
        choose = np.array([choose])

        img_masked = norm(img_masked)
        cloud = cloud.astype(np.float32)

        cloud = torch.from_numpy(cloud.astype(np.float32))
        choose = torch.LongTensor(choose.astype(np.int32))
        index = torch.LongTensor([0])

        cloud = Variable(cloud).cuda()
        choose = Variable(choose).cuda()
        img_masked = Variable(img_masked).cuda()
        index = Variable(index).cuda()

        cloud = cloud.view(1, num_points, 3)
        img_masked = img_masked.view(1, 3, img_masked.size()[1], img_masked.size()[2])

        pred_r, pred_t, pred_c = estimator(img_masked, cloud, choose, index)
        try:
            pred_t, pred_mask = ransac_voting_layer(cloud, pred_t)
        except RuntimeError:
            print('RANSAC voting fails')
            return predict_sRT, None, np.array(model_pc.points)[:, :3]

        my_t = pred_t.cpu().data.numpy()
        how_min, which_min = torch.min(pred_c, 1)
        my_r = pred_r[0][which_min[0]].view(-1).cpu().data.numpy()
        points = cloud.view(num_points, 3)

        predict_sRT = quaternion_matrix(my_r)
        predict_sRT[:3, 3] = my_t

    if predict_sRT is None:
        return predict_sRT, None, model_pc[:, :3]
    else:
        return predict_sRT, points.squeeze().detach().cpu().numpy(), model_pc[:, :3]
        # predict_sRT是一个4x4的numpy数组，姿态变换矩阵
        # points是一个numpy数组，包含点云中作为神经网络输入的点的三维坐标。点的形状是（num_points, 3），其中num_points是点云中的点的数量。
        # model_pc[:, :3]是一个numpy数组，包含CAD模型中的点的三维坐标。model_pc[:, :3]的形状是（num_points_in_cad_model, 3），
        # 其中num_points_in_cad_model是CAD模型中的点的数量。(我的理解是  model的初始姿态的 三为坐标)

        # pts_in_camera: (1000, 3)

        # pts_in_obj: (1000, 3)
        # init_pose: (4, 4)

        """将相机中的点和物体中的点作为输入，然后使用knn算法来计算这些点之间的距离。knn是一个参数，它代表着k值，即最近邻居的数量"""

# 对一个object进行初始的一个变换（ init pose ），即上个函数的return sRT
def measure_pseudo_pose_in_3d(pts_in_camera, pts_in_obj, init_pose, knn):

    """
    测量两个不同坐标系中的两组三维点之间的距离。该函数首先使用一个同质转换矩阵将输入的三维点(model初始)从一个坐标系转换到另一个坐标系（姿态变换）。
              然后，该函数使用k-nearest neighbors（k-NN）算法来寻找一个张量中的每个点在另一个张量中的最近邻居。
    然后计算每个点与其最近的邻居之间的距离并取其平均值。最后，这两个平均距离的总和作为该函数的输出被返回。
    """

    transformed_obj_pts = transform_coordinates_3d(pts_in_obj.T, init_pose)# 返回的是变换后的 坐标   转置了
    transformed_camera_pts = pts_in_camera.T# 相机中的物体 坐标

    target_pts = torch.from_numpy(transformed_camera_pts).cuda()
    source_pts = torch.from_numpy(transformed_obj_pts).cuda()

    index1 = knn.apply(target_pts.unsqueeze(0), source_pts.unsqueeze(0))  # unsqueeze(0)  加一个维度  在前方（batch=1）   1，3,num
    index2 = knn.apply(source_pts.unsqueeze(0), target_pts.unsqueeze(0))

    #pts1 = torch.index_select(target_pts, 1, index1.view(-1) -1 )# -1的作用？？？   knn_cuda是否多余
    pts1 = torch.index_select(target_pts, 1, index1.view(-1))# 1是维度 表示按index1.view(-1) - 1的第二维（k)索引     index1.view(-1) - 1是索引数  第几个
    #view(-1)将index1展平得到0 ，1 ，2。。。                  得到了K个  坐标

    cd1 = torch.mean(torch.norm((source_pts.float() - pts1.float()), dim=0)).cpu().item()

    pts2 = torch.index_select(source_pts, 1, index2.view(-1))
    cd2 = torch.mean(torch.norm((target_pts.float() - pts2.float()), dim=0)).cpu().item()
    return cd1 + cd2


def measuring_mask_in_2d(pseudo_mask, observed_mask):
    positive_pixels = pseudo_mask[observed_mask > 0]
    negative_pixels = observed_mask[pseudo_mask == 0]

    positive_error = sum(abs(1 - positive_pixels)) / (len(positive_pixels) + 1)
    negative_error = sum(negative_pixels) / (len(negative_pixels) + 1)

    return positive_error + negative_error


def measuring_rgb_in_2d(pseudo_rgb, observed_rgb, observed_mask):
    return measure_perceptual_similarity(pseudo_rgb, observed_rgb, observed_mask)


def measuring_pseudo_pose_in_2d(obj_id, inst_id, pose, observed_img, observed_mask, ren, ren_model, render_height,
                                render_width, intrinsics):
    horizontal_indicies = np.where(np.any(observed_mask, axis=0))[0]
    vertical_indicies = np.where(np.any(observed_mask, axis=1))[0]
    x1, x2 = horizontal_indicies[[0, -1]]
    y1, y2 = vertical_indicies[[0, -1]]
    x2 += 1
    y2 += 1

    img_height = observed_img.shape[0]
    img_width = observed_img.shape[1]

    rgb_img, _, mask_img, _ = croped_rendering(obj_id, pose, intrinsics, img_height, img_width, int(float(x1 + x2) / 2),
                                               int(float(y1 + y2) / 2), render_height, render_width, ren, ren_model)

    rgb_img = copy.deepcopy(rgb_img[y1:y2, x1:x2, :])
    observed_img = copy.deepcopy(observed_img[y1:y2, x1:x2, :])
    mask_img = copy.deepcopy(mask_img[y1:y2, x1:x2])
    observed_mask = copy.deepcopy(observed_mask[y1:y2, x1:x2])

    mask_error = measuring_mask_in_2d(mask_img, observed_mask)
    perceptual_error = measuring_rgb_in_2d(rgb_img, observed_img, observed_mask)

    return mask_error, perceptual_error


"""
输入：
迭代数，renderer/robi_models（渲染器路径）, obj_id = zigzag , intrinsics=intrinsics相机相关 不用管

此处有网络
"""
def label_poses_with_teacher(iter_idx=1, renderer_model_dir='renderer/robi_models', obj_id='01', render_h=480,
                             render_w=480, intrinsics=None):
    knn = KNearestNeighbor(1)  # 1

    # 定义渲染器 和 路径
    render_height = render_h
    render_width = render_w
    ren = DIBRenderer(render_height, render_width, mode="VertexColorBatch")  # 渲染器
    obj_paths = [os.path.join(renderer_model_dir, "{}/textured.obj".format(obj_id))]
    texture_paths = [os.path.join(renderer_model_dir, "{}/texture_map.png".format(obj_id))]
    ren_model = load_objs(obj_paths, texture_paths, height=render_height, width=render_width)  # 反会带渲染信息的字典  的列表

    # 加载cad
    cad_model_name = obj_id
    data_root_dir = './data'
    cad_model_path = data_root_dir + '/cad_models/' + cad_model_name + '.obj'  # data/cad models/zigzag.obj
    cad_model_pcs, cad_model_faces = load_obj(cad_model_path)  # 返回顶点和面信息

    # load the trained network for pose labelling加载模型
    estimator = PoseNet(num_points=1000, num_obj=1, num_rot=60)
    estimator.cuda()

    estimator.load_state_dict(torch.load('./real_models/' + obj_id + '/last/pose_model.pth'))# real_models/zigzag/best（原来的路径  改过）


    estimator.eval()# 进行验证？？？

    # read the data for current object
    # current_obj_data_dir = os.path.join(data_root_dir, obj_id, 'training_data')#data/zigzag/training_data  !!!缺少
    current_obj_data_dir = os.path.join(data_root_dir, obj_id, 'RealData/data')  # 自己改的 真实数据集


    #  排列好路径并存储
    # color image
    img_pathes = glob.glob(current_obj_data_dir + '/*_color.png')
    img_pathes = sorted(img_pathes, key=lambda a: a.split('/')[-1].split('.')[0])

    # depth image
    depth_pathes = glob.glob(current_obj_data_dir + '/*_depth.png')
    depth_pathes = sorted(depth_pathes, key=lambda a: a.split('/')[-1].split('.')[0])

    # mask image
    mask_pathes = glob.glob(current_obj_data_dir + '/*_mask.png')
    mask_pathes = sorted(mask_pathes, key=lambda a: a.split('/')[-1].split('.')[0])

    # label
    '''
    使用真实数据集的标签是为了方便地获取物体的实例编号，而不是为了依赖于真实数据集的姿态标签。
    如果您没有真实数据集的标签，您也可以使用其他方法来获取物体的实例编号，例如使用  聚类算法或分割网络   。
    这样，您就不需要使用真实标签，而只需要使用无标签的图像和深度数据，以及仿真数据训练好的姿态估计器。
    
    问题：仍然需要物体编号，需要真实label或者网络算法来获取  会很复杂！！！  作者这里没有给出
    实际上还是别人的网络为基础，  而别人的网络是用来识别不同的物体的  而我们的场景是多个相同物体 可以用但毕竟不是最合适的
    '''
    label_pathes = glob.glob(current_obj_data_dir + '/*_label.pkl')
    label_pathes = sorted(label_pathes, key=lambda a: a.split('/')[-1].split('.')[0])  # 。。。label

    # data name 图片名称
    data_names = []
    for i in range(len(img_pathes)):
        data_names.append(img_pathes[i].split('/')[-1].split('.')[0][:-6])  # 0001_view_71_LEFT（0到倒数第7个）

    # trverse all scenes for labelling
    scene_poses = []
    scene_residuals = []  # 3d error
    scene_mask_errors = []
    scene_rgb_errors = []
    scene_2d_errors = []
    scene_valid_iid = []

    flatten_residuals = []
    flatten_mask_errors = []
    flatten_rgb_errors = []
    flatten_2d_errors = []

    total_inst_num = 0
    for i in range(len(img_pathes)):# 图片数量
        # for i in range(10):
        print(i, img_pathes[i])
        current_scene_poses = []
        current_scene_residuals = []
        current_scene_mask_errors = []
        current_scene_rgb_errors = []
        current_scene_2d_errors = []
        current_scene_valid_iid = []

        # name check 检查 不然就报错
        assert (data_names[i] == img_pathes[i].split('/')[-1].split('.')[0][:-6])
        assert (data_names[i] == depth_pathes[i].split('/')[-1].split('.')[0][:-6])
        assert (data_names[i] == mask_pathes[i].split('/')[-1].split('.')[0][:-5])
        assert (data_names[i] == label_pathes[i].split('/')[-1].split('.')[0][:-6])

        img = cv2.imread(img_pathes[i])[:, :, :3]  # 3通道 不需要alpha
        img = img[:, :, ::-1]  # RGB  《--  BGR灰色变彩色  cv2读取图片顺序是反的！！！
        depth = cv2.imread(depth_pathes[i], -1)  # -1   读取原图
        mask = cv2.imread(mask_pathes[i], -1)

        # 读取label
        with open(label_pathes[i], 'rb') as f:
            label = cPickle.load(f)  # 列表每个元素都是 一个字典  因为是label存了信息

        # traverse all instances in each scene for labelling
        for iid in range(len(label['instance_ids'])):# 实例个数  图片里显示的  但这里不是已经用到了 label吗
            total_inst_num += 1
            inst_id = label['instance_ids'][iid] + 1# 实例编号 最小是1  但不一定是 1 2 3 4只是为了获取编号 但是真实数据集应该怎么获取编号呢

            # sample points
            observed_mask = np.equal(mask, inst_id)  #image每个像素的RGB值与inst_id进行比对，bool值。
            current_mask = np.equal(mask, inst_id)
            current_mask = np.logical_and(current_mask, depth > 0)# 有深度信息 且 mask又是某个编号的才显示true

            # posenet img\depth\current mask? 采样吗\相机参数、ZIGZAG顶点和面信息 标记单独的姿态
            pose_from_teacher, pts_in_camera, pts_in_obj = label_individual_pose(estimator, img, depth, current_mask,
                                                                                 intrinsics, cad_model_pcs,
                                                                                 cad_model_faces)#此处有网络

            if pose_from_teacher is not None:
                dis_3d = measure_pseudo_pose_in_3d(pts_in_camera, pts_in_obj, pose_from_teacher, knn)
                current_scene_valid_iid.append(label['instance_ids'][iid])# 实例id

                current_scene_poses.append(pose_from_teacher)
                current_scene_residuals.append(dis_3d)
                flatten_residuals.append(dis_3d)

                mask_error, rgb_error = measuring_pseudo_pose_in_2d(obj_id, total_inst_num, pose_from_teacher, img,
                                                                    observed_mask, ren, ren_model, render_height,
                                                                    render_width, intrinsics)

                current_scene_mask_errors.append(mask_error)
                flatten_mask_errors.append(mask_error)

                current_scene_rgb_errors.append(rgb_error)
                flatten_rgb_errors.append(rgb_error)

                current_scene_2d_errors.append(mask_error * rgb_error)
                flatten_2d_errors.append(mask_error * rgb_error)

        scene_poses.append(current_scene_poses)
        scene_residuals.append(current_scene_residuals)
        scene_mask_errors.append(current_scene_mask_errors)
        scene_rgb_errors.append(current_scene_rgb_errors)
        scene_2d_errors.append(current_scene_2d_errors)
        scene_valid_iid.append(current_scene_valid_iid)

    # compute statistics information for all residuals
    mean_residual = np.mean(np.array(flatten_residuals))
    var_residual = np.std(np.array(flatten_residuals))# 标准差
    threshold_3d_error = mean_residual + var_residual

    mean_2d_error = np.mean(np.array(flatten_2d_errors))
    var_2d_error = np.std(np.array(flatten_2d_errors))
    threshold_2d_error = mean_2d_error + var_2d_error

    # filtering out the pose whose residual is larger than the threshold
    scene_good_iid = []
    scene_good_poses = []
    good_scene_names = []

    valid_inst_num = 0
    for i in range(len(scene_poses)):
        current_scene_good_iid = []# 每次循环都重置
        current_scene_good_poses = []
        assert (len(scene_residuals[i]) == len(scene_poses[i]))
        assert (len(scene_residuals[i]) == len(scene_valid_iid[i]))

        for j in range(len(scene_residuals[i])):
            if scene_2d_errors[i][j] < threshold_2d_error and scene_residuals[i][j] < threshold_3d_error:
                current_scene_good_iid.append(scene_valid_iid[i][j])
                current_scene_good_poses.append(scene_poses[i][j])
                valid_inst_num += 1

        if len(current_scene_good_iid) > 1:# 有元素  就加入图片
            good_scene_names.append(data_names[i])# 添加图片名称
            scene_good_iid.append(current_scene_good_iid)# current_scene_good_iid是一维的  所以  此变量是2维
            scene_good_poses.append(current_scene_good_poses)

    # also need to update the label.pkl and the mask.png at the same time
    teacher_label_dir = os.path.join(data_root_dir, obj_id, 'teacher_label_iter_' + str(iter_idx).zfill(2))
    if not os.path.exists(teacher_label_dir):
        os.makedirs(teacher_label_dir)

    for i in range(len(good_scene_names)):
        # update the label.pkl
        current_label = {}
        current_label['instance_ids'] = scene_good_iid[i]

        translations = np.zeros((len(scene_good_poses[i]), 3))
        rotations = np.zeros((len(scene_good_poses[i]), 3, 3))

        for j in range(len(scene_good_poses[i])):
            translations[j, :] = scene_good_poses[i][j][:3, 3]
            rotations[j, :, :] = scene_good_poses[i][j][:3, :3]
        current_label['translations'] = translations.astype(np.float32)
        current_label['rotations'] = rotations.astype(np.float32)

        # update the mask.png
        current_label['bboxes'] = []
        original_mask = cv2.imread(os.path.join(current_obj_data_dir, good_scene_names[i] + '_mask.png'), -1)
        updated_mask = np.ones(original_mask.shape[:2], dtype=np.uint8) * 255# 变成黑白
        for j in range(len(scene_good_iid[i])):
            inst_id = scene_good_iid[i][j] + 1
            current_mask = np.equal(original_mask, inst_id)
            updated_mask[current_mask] = inst_id# 将 updated_mask 数组中与 current_mask 数组中为真的元素对应的位置赋值为 inst_id 的值

            bbox = current_mask.flatten().nonzero()[0]
            ys = bbox // original_mask.shape[1]
            xs = bbox - ys * original_mask.shape[1]
            bbox = [np.min(ys), np.min(xs), np.max(ys), np.max(xs)]
            current_label['bboxes'].append(bbox)

        img = cv2.imread(os.path.join(current_obj_data_dir, good_scene_names[i] + '_color.png'), -1)
        depth = cv2.imread(os.path.join(current_obj_data_dir, good_scene_names[i] + '_depth.png'), -1)
        cv2.imwrite(os.path.join(teacher_label_dir, good_scene_names[i] + '_color.png'), img)
        cv2.imwrite(os.path.join(teacher_label_dir, good_scene_names[i] + '_depth.png'), depth)

        cv2.imwrite(os.path.join(teacher_label_dir, good_scene_names[i] + '_mask.png'), updated_mask)
        with open(os.path.join(teacher_label_dir, good_scene_names[i] + '_label.pkl'), 'wb') as f:
            cPickle.dump(current_label, f)
    del knn


if __name__ == '__main__':
    pass
