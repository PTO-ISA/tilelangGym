"""SV9 stable top-K. The three arms match benches/sv9/sv9_topk_e2e.py."""

import tilelang.ascend.language as T

NEG = -3.402823e38
INT_MAX = 2147483647


def make_kernel(E: int, K: int, threads: int, arm: str):
    if arm == "gap32":
        if threads != 32 or E % threads != 0:
            raise ValueError("gap32 is one warp and E must divide the warp")
        reps = E // threads

        @T.prim_func
        def main(
            A: T.Tensor((E,), "float32"),
            IdxOut: T.Tensor((K,), "int32"),
        ):
            with T.Kernel(1):
                s = T.alloc_shared((E,), "float32")
                idx_ub = T.alloc_shared((K,), "int32")
                T.copy(A, s)
                with T.SimtVF(threads=threads):
                    # Lane t holds s[t], s[t+32], s[t+64], ...
                    scores = T.alloc_fragment((reps,), "float32")
                    tx = T.get_thread_binding()
                    mloc = T.alloc_var("float32")
                    amax = T.alloc_var("float32")
                    best_l = T.alloc_var("int32")
                    best = T.alloc_var("int32")
                    for step in T.unroll(reps):
                        scores[step] = s[tx + step * threads]
                    for _rnd in T.serial(K):
                        mloc = scores[0]
                        for step in T.unroll(reps):
                            mloc = T.max(mloc, scores[step])
                        amax = T.warp_reduce_max(mloc)
                        best_l = INT_MAX
                        for step in T.unroll(reps):
                            if scores[step] == amax:
                                best_l = T.min(best_l, tx + step * threads)
                        best = T.warp_reduce_min(best_l)
                        if tx == 0:
                            idx_ub[_rnd] = best
                        for step in T.unroll(reps):
                            if tx + step * threads == best:
                                scores[step] = NEG
                T.copy(idx_ub, IdxOut)

        return main

    if arm == "remat_scores":

        @T.prim_func
        def main(
            A: T.Tensor((E,), "float32"),
            IdxOut: T.Tensor((K,), "int32"),
        ):
            with T.Kernel(1):
                s = T.alloc_shared((E,), "float32")
                idx_ub = T.alloc_shared((K,), "int32")
                T.copy(A, s)
                with T.SimtVF(threads=threads):
                    scores = T.alloc_fragment((E,), "float32")
                    idxs = T.alloc_fragment((E,), "int32")
                    idx_cand = T.alloc_fragment((E,), "int32")
                    amax = T.alloc_fragment((1,), "float32")
                    best = T.alloc_fragment((1,), "int32")
                    for i in T.Parallel(E):
                        idxs[i] = i
                    for k in T.serial(K):
                        for i in T.Parallel(E):
                            scores[i] = s[i]
                        T.reduce_max(scores, amax)
                        for i in T.Parallel(E):
                            if scores[i] == amax[0]:
                                idx_cand[i] = idxs[i]
                            else:
                                idx_cand[i] = INT_MAX
                        T.reduce_min(idx_cand, best)
                        idx_ub[k] = best[0]
                        for i in T.Parallel(E):
                            if idxs[i] == best[0]:
                                s[i] = NEG
                T.copy(idx_ub, IdxOut)

        return main

    if arm == "remat_idx":

        @T.prim_func
        def main(
            A: T.Tensor((E,), "float32"),
            IdxOut: T.Tensor((K,), "int32"),
        ):
            with T.Kernel(1):
                s = T.alloc_shared((E,), "float32")
                idx_ub = T.alloc_shared((K,), "int32")
                T.copy(A, s)
                with T.SimtVF(threads=threads):
                    scores = T.alloc_fragment((E,), "float32")
                    idx_cand = T.alloc_fragment((E,), "int32")
                    amax = T.alloc_fragment((1,), "float32")
                    best = T.alloc_fragment((1,), "int32")
                    for i in T.Parallel(E):
                        scores[i] = s[i]
                    for k in T.serial(K):
                        T.reduce_max(scores, amax)
                        for i in T.Parallel(E):
                            idx_i = i
                            if scores[i] == amax[0]:
                                idx_cand[i] = idx_i
                            else:
                                idx_cand[i] = INT_MAX
                        T.reduce_min(idx_cand, best)
                        idx_ub[k] = best[0]
                        for i in T.Parallel(E):
                            if i == best[0]:
                                scores[i] = NEG
                T.copy(idx_ub, IdxOut)

        return main

    if arm != "keep":
        raise ValueError(arm)

    @T.prim_func
    def main(
        A: T.Tensor((E,), "float32"),
        IdxOut: T.Tensor((K,), "int32"),
    ):
        with T.Kernel(1):
            s = T.alloc_shared((E,), "float32")
            idx_ub = T.alloc_shared((K,), "int32")
            T.copy(A, s)
            with T.SimtVF(threads=threads):
                scores = T.alloc_fragment((E,), "float32")
                idxs = T.alloc_fragment((E,), "int32")
                idx_cand = T.alloc_fragment((E,), "int32")
                amax = T.alloc_fragment((1,), "float32")
                best = T.alloc_fragment((1,), "int32")
                for i in T.Parallel(E):
                    scores[i] = s[i]
                    idxs[i] = i
                for k in T.serial(K):
                    T.reduce_max(scores, amax)
                    for i in T.Parallel(E):
                        if scores[i] == amax[0]:
                            idx_cand[i] = idxs[i]
                        else:
                            idx_cand[i] = INT_MAX
                    T.reduce_min(idx_cand, best)
                    idx_ub[k] = best[0]
                    for i in T.Parallel(E):
                        if idxs[i] == best[0]:
                            scores[i] = NEG
            T.copy(idx_ub, IdxOut)

    return main
