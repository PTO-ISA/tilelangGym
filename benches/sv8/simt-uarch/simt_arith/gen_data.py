#!/usr/bin/env python3
import os

import numpy as np

np.random.seed(19)

HEIGHT = 64
WIDTH = 128


def write_case(name, src0, src1, golden):
    path = f"SIMTArithTest.{name}"
    os.makedirs(path, exist_ok=True)
    src0.astype(np.float32).tofile(os.path.join(path, "input1.bin"))
    src1.astype(np.float32).tofile(os.path.join(path, "input2.bin"))
    golden.astype(np.float32).tofile(os.path.join(path, "golden.bin"))


def main():
    src0 = np.random.randint(1, 10, size=(HEIGHT, WIDTH)).astype(np.float32)
    src1 = np.random.randint(1, 10, size=(HEIGHT, WIDTH)).astype(np.float32)
    write_case("case_fmul", src0, src1, src0 * src1)
    write_case("case_fdiv", src0, src1, src0 / src1)
    write_case("case_fmnmax", src0, src1, np.maximum(src0, src1))
    flat = src0.reshape(-1)
    reduced = flat.reshape(-1, 32).max(axis=1)
    golden = np.repeat(reduced, 32).reshape(src0.shape)
    write_case("case_redux", src0, src1, golden)


if __name__ == "__main__":
    main()
