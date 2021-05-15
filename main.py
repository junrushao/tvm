import tvm
import numpy as np
import tvm.testing

rocblas_matmul = tvm._ffi.get_global_func("tvm.contrib.rocblas.matmul")

shapes = [
    (1024, 2048, 1024),
    (1024, 2048, 4096),
    (2048, 1024, 1024),
    (2048, 1024, 30528),
    (2048, 1024, 4096),
    (2048, 30528, 1024),
    (2048, 4096, 1024),
    (30528, 2048, 1024),
    (4096, 2048, 1024),
]

for M, K, N in shapes:
    a_np = np.zeros((M, K), dtype="float32")
    b_np = np.zeros((N, K), dtype="float32")
    c_np = np.zeros((M, N), dtype="float32")

    a_np[:] = np.random.uniform(size=a_np.shape)
    b_np[:] = np.random.uniform(size=b_np.shape)
    c_np[:] = np.random.uniform(size=c_np.shape)
    c_np_expected = a_np @ b_np.T

    a = tvm.nd.array(a_np, ctx=tvm.rocm(0))
    b = tvm.nd.array(b_np, ctx=tvm.rocm(0))
    c = tvm.nd.array(c_np, ctx=tvm.rocm(0))

    rocblas_matmul(a, b, c, False, True)

    c_np = c.asnumpy()
    tvm.testing.assert_allclose(c_np, c_np_expected, rtol=1e-4)
