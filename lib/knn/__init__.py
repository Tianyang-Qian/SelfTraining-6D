import unittest
import gc
import operator as op
import functools
import torch
from torch.autograd import Variable, Function
import knn_cuda
# import torchknn
from knn_cuda import KNN


class KNearestNeighbor(Function):
    """ Compute k nearest neighbors for each query point.
    """

    def __init__(self, k):
        self.k = k

    def forward(self, ref, query):
        """
        Args:
            ref: b x dim x n_ref（要求）
            query: b x dim x n_query
            寻找一个张量中的每个点在另一个张量中的最近邻居
            根据源文件 将CUDAKNN改称 IND输出格式
        """
        ref = ref.contiguous().float().cuda()
        query = query.contiguous().float().cuda()
        #inds = torch.empty(query.shape[0], self.k, query.shape[2]).long().cuda()# [batch,k,query num]  k=1  就是所  每个点都找一个最近的点（索引）

        #
        # knn_cuda.knn(ref, query, inds)


        # cuda版本使用方法
        knn = KNN(k=1, transpose_mode=False)
        dist, indx = knn(ref, query)# indx[bs x k x nq]

#       batch ,k ,numquery  #不需要返回坐标  所以dim变成了k（1，索引）
        return indx



class TestKNearestNeighbor(unittest.TestCase):
    def test_forward(self):
        knn = KNearestNeighbor(20)
        while (1):
            D, N, M = 128, 100, 1000
            ref = Variable(torch.rand(2, D, N))
            query = Variable(torch.rand(2, D, M))

            inds = knn.forward(ref, query)
            for obj in gc.get_objects():
                if torch.is_tensor(obj):
                    print(functools.reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0, type(obj), obj.size())
            # ref = ref.cpu()
            # query = query.cpu()
            print(inds)


if __name__ == '__main__':
    unittest.main()
