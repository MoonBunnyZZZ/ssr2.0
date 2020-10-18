import numpy as np
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline


def transform_matrix(index):
    dst_cx, dst_cy = (54, 54)
    src_cx, src_cy = (54, 54)
    t1 = np.array([[1, 0, -dst_cx], [0, 1, -dst_cy], [0, 0, 1]])
    t2 = np.array([[1, 0, src_cx], [0, 1, src_cy], [0, 0, 1]])

    zx, zy = np.random.uniform(0.8, 1.2, 2)
    zoom_matrix = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])

    # shear = np.random.uniform(-0.2, 0.2)
    # shear_martrix = np.array([[1, -np.sin(shear), 0], [0, np.cos(shear), 0], [0, 0, 1]])

    tx, ty = np.random.uniform(-0.2, 0.2, 2)
    shift_matrix = np.array([[1, 0, tx * 108], [0, 1, ty * 108], [0, 0, 1]])

    theta = np.pi / 180 * np.random.uniform(-20, 20)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    m = np.matmul(zoom_matrix, shift_matrix)
    m = np.matmul(rotation_matrix, m)

    # translate input coordinates to center (src_cx, src_cy) and combine the transforms
    m = (np.matmul(t2, np.matmul(m, t1)))
    return m[0:2, 0:3]  # remove the last row; it's not used by affine transform


def gen_transforms(batch_size, single_transform_fn):
    out = np.zeros([batch_size, 2, 3])
    for i in range(batch_size):
        out[i, :, :] = single_transform_fn(i)
    return out.astype(np.float32)


class TrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, db_dir):
        super(TrainPipe, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.MXNetReader(path=[db_dir + "train.rec"], index_path=[db_dir + "train.idx"],
                                     random_shuffle=False, shard_id=device_id, num_shards=num_gpus)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        # self.blur = ops.GaussianBlur(device="cpu", sigma=5.0, window_size=5)
        # self.reshape = ops.Reshape(device="cpu", layout="HWC")
        self.normalize = ops.CropMirrorNormalize(device="gpu",
                                                 output_dtype=types.FLOAT, output_layout=types.NCHW,
                                                 mean=[123.0, 116.0, 103.0], std=[100.0, 100.0, 100.0]
                                                 )
        self.transform_source = ops.ExternalSource()
        self.warp_gpu = ops.WarpAffine(device="gpu", interp_type=types.INTERP_LINEAR)

    def define_graph(self):
        jpegs, labels = self.input(name='Reader')
        outputs = self.decode(jpegs)

        # outputs = self.blur(outputs)
        # outputs = self.reshape(outputs)
        self.transform = self.transform_source()
        outputs = self.warp_gpu(outputs, self.transform)
        outputs = self.normalize(outputs)

        return outputs, labels

    # Generate the transforms for the batch and feed them to the ExternalSource
    def iter_setup(self):
        self.feed_input(self.transform, gen_transforms(self.batch_size, transform_matrix))


class ValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, num_gpus, db_dir):
        super(ValPipe, self).__init__(batch_size, num_threads, device_id)
        self.input = ops.MXNetReader(path=[db_dir + "val.rec"], index_path=[db_dir + "val.idx"],
                                     random_shuffle=False, shard_id=device_id, num_shards=num_gpus)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.normalize = ops.CropMirrorNormalize(device="gpu",
                                                 output_dtype=types.FLOAT, output_layout=types.NCHW,
                                                 mean=[123.0, 116.0, 103.0], std=[100.0, 100.0, 100.0]
                                                 )

    def define_graph(self):
        jpegs, labels = self.input(name='Reader')
        images = self.decode(jpegs)

        outputs = self.normalize(images)

        return outputs, labels
