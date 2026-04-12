import numpy as np


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
    channels = x.shape[0]
    height = x.shape[1]
    width = x.shape[2]

    m = w.shape[0]

    out_height = ((height - m + 2 * padding) // stride) + 1
    out_width = ((width - m + 2 * padding) // stride) + 1
    channels_out = w.shape[0]
    res = np.zeros(shape=(out_height, out_width))

    # add padding
    x = np.pad(x, ((padding, padding), (padding, padding)))

    # get new height and width after padding
    height, width = x.shape

    # convolve
    for c in range(channels_out):
        w_c = w[c]
        for i in range(out_height):
            for j in range(out_width):
                i_start = i * stride
                j_start = j * stride
                window = x[i_start : i_start + m, j_start : j_start + m]
                res[i, j] += np.sum(window * w_c) + b

    return res


if __name__ == "__main__":
    np.random.seed(42)
    x = np.random.uniform(0, 1, size=(5, 5))
    w = np.random.uniform(0, 1, size=(2, 3, 3))
    b = np.array(0.0)

    c = conv2d(x, w, b, stride=2, padding=2)
    print(c)
