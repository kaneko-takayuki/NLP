import numpy

from chainer import function
from chainer import cuda
from chainer.utils import type_check
from econvertor import word2vec as w2v


class SimilaritySquaredError(function.Function):

    """Mean squared error (a.k.a. Euclidean loss) function."""

    def check_type_forward(self, in_types):
        type_check.expect(in_types.size() == 2)
        type_check.expect(
            in_types[0].dtype == numpy.float32,
            in_types[1].dtype == numpy.float32,
            in_types[0].shape == in_types[1].shape
        )

    def forward_cpu(self, inputs):
        x0, x1, t = inputs

        # コサイン類似度
        cos_sim = numpy.array(
            [numpy.dot(v0, v1) / (numpy.linalg.norm(v0) * numpy.linalg.norm(v1)) for (v0, v1) in zip(x0, x1)])

        # コサイン類似度と、教師類似度の差を計算し、2乗したものを誤差とする
        return numpy.array([(v - t) ** 2 for v in cos_sim])

    def forward_gpu(self, inputs):
        x0, x1, t = inputs

        # コサイン類似度
        cos_sim = numpy.array(
            [numpy.dot(v0, v1) / (numpy.linalg.norm(v0) * numpy.linalg.norm(v1)) for (v0, v1) in zip(x0, x1)])

        # コサイン類似度と、教師類似度の差を計算し、2乗したものを誤差とする
        return cuda.ndarray([(v - t) ** 2 for v in cos_sim])

        x0, x1 = inputs
        self.diff = x0 - x1
        diff = self.diff.ravel()
        return diff.dot(diff) / diff.dtype.type(diff.size),

    def backward(self, inputs, gy):
        coeff = gy[0] * gy[0].dtype.type(2. / self.diff.size)
        gx0 = coeff * self.diffvapulanlp

        return gx0, -gx0


def similarity_squared_error(x0, x1, t):
    return SimilaritySquaredError()(x0, x1)
