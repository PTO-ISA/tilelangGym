import os

import numpy as np

np.random.seed(19)


def main():
    src = np.random.randint(-8, 9, size=[64, 128]).astype(np.float16)
    golden = src.astype(np.float32)
    case = "SIMTF2fTest.case1_half_64_128"
    os.makedirs(case, exist_ok=True)
    src.tofile(os.path.join(case, "input1.bin"))
    golden.tofile(os.path.join(case, "golden.bin"))


if __name__ == "__main__":
    main()
