import numpy as np


def convolve_window(x: np.ndarray, w: np.ndarray, b: float) -> float:
    """
    Local convolution
    """
    assert x.shape == w.shape

    C, H, W = w.shape

    a = 0.0
    for i in range(C):
        for j in range(H):
            for k in range(W):
                a += x[i][j][k] * w[i][j][k]

    return a + b


def convolve(x: np.ndarray, w: np.ndarray, b: float, stride: int) -> np.ndarray:
    """
    Convolution of one sample with one kernel and the given stride
    """

    # compute output size
    H_out = (x.shape[1] - w.shape[1]) // stride + 1
    W_out = (x.shape[2] - w.shape[2]) // stride + 1

    # get kernel height and width
    H = w.shape[1]
    W = w.shape[2]

    output = np.zeros(shape=(H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            output[i][j] = convolve_window(
                x[
                    :,
                    i * stride : i * stride + H,
                    j * stride : j * stride + W,
                ],
                w,
                b,
            )

    return output


def conv2d(
    x: np.ndarray, w: np.ndarray, b: np.ndarray, stride: int, padding: int
) -> np.ndarray:
    """
    Bidimensional convolution from C_in input channels to C_out output channels.
    The convolution has kernel size (kH x kW) and arbitrary stride and padding.

    Shapes:
      - W: (C_out, C_in, kH, kW)
      - b: (C_out,)
    """

    N = x.shape[0]
    print(f"number of samples: {N}")
    print(f"sample shape: {x[0].shape}")

    # padding
    padded_x = []
    for i in range(N):
        padded_x.append(
            np.pad(x[i], pad_width=((0, 0), (padding, padding), (padding, padding)))
        )
    x = np.array(padded_x)

    # get new height and width after padding
    C, H, W = x.shape[1], x.shape[2], x.shape[3]
    print(f"samples shape after padding: {(C, H, W)}")

    # compute output size
    H_out = (x.shape[2] - w.shape[2]) // stride + 1
    W_out = (x.shape[3] - w.shape[3]) // stride + 1

    C_out = w.shape[0]
    output = np.zeros(shape=(N, C_out, H_out, W_out))
    print(f"output shape: {output.shape}")
    for i in range(N):  # iterate of samples
        for j in range(C_out):  # iterate over kernels
            # convolve one sample with one kernel
            output[i][j] = convolve(x[i], w[j], b[j], stride)

    return output


def max_pool_sample(x: np.ndarray, pool_size: int, stride: int) -> np.ndarray:

    # compute output size
    H, W = x.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    output = np.zeros(shape=(H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            output[i][j] = np.max(
                x[i * stride : i * stride + H, j * stride : j * stride + W]
            )

    return output


def max_pool(x: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
    """
    Max-pooling with arbitrary pool size and stride.
    """

    N, C, H, W = x.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    output = np.zeros(shape=(N, C, H_out, W_out))
    for i in range(N):
        for j in range(C):
            output[i][j] = max_pool_sample(x[i][j], pool_size, stride)

    return output


def avg_pool_sample(x: np.ndarray, pool_size: int, stride: int) -> np.ndarray:

    # compute output size
    H, W = x.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    output = np.zeros(shape=(H_out, W_out))
    for i in range(H_out):
        for j in range(W_out):
            output[i][j] = np.average(
                x[i * stride : i * stride + H, j * stride : j * stride + W]
            )

    return output


def avg_pool(x: np.ndarray, pool_size: int = 2, stride: int = 2) -> np.ndarray:
    """
    Average-pooling with arbitrary pool size and stride.
    """

    N, C, H, W = x.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    output = np.zeros(shape=(N, C, H_out, W_out))
    for i in range(N):
        for j in range(C):
            output[i][j] = avg_pool_sample(x[i][j], pool_size, stride)

    return output


if __name__ == "__main__":
    np.random.seed(42)

    # samples settings
    N = 10
    H = 8
    W = 8
    C = 3

    # kernel settings
    C_out = 2
    K = 3

    # stride and padding
    S = 1
    P = 1

    x = np.random.uniform(0, 1, size=(N, C, H, W))
    w = np.random.uniform(0, 1, size=(C_out, C, K, K))
    b = np.random.uniform(0, 1, size=(C_out))

    out = conv2d(x, w, b, S, P)
    print(f"shape after convolution: {out.shape}")
    out = max_pool(out)
    print(f"shape after max pooling: {out.shape}")
    out = avg_pool(out)
    print(f"shape after avg pooling: {out.shape}")
    out = np.maximum(0, out)
    print(f"shape after relu: {out.shape}")
