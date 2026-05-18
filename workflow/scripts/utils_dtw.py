import os
import math
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit


def dtw_features_to_templates(signals, templates, template_names=None, window_frac=0.10, n_jobs=None, norm='path'):
    """
    Compute DTW distances from each signal to each template.
    """
    if n_jobs is None:
        n_jobs = max(1, (os.cpu_count() or 2) - 1)
    N = len(signals)
    M = len(templates)
    if template_names is None:
        template_names = [f"tmpl_{i}" for i in range(M)]
    if M == 0:
        return pd.DataFrame(index=np.arange(N))

    sigs = [np.asarray(s) for s in signals]
    tmpls = [np.asarray(t) for t in templates]

    # build all pairs (i_template, i_signal)
    pairs = [(ti, si) for ti in range(M) for si in range(N)]

    # worker: compute distance for a given (ti, si)
    def _worker_pair(ti, si):
        d = _dtw_dist_pair(tmpls[ti], sigs[si], window_frac=window_frac)
        return ti, si, d

    results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_worker_pair)(ti, si) for (ti, si) in pairs)
    # build matrix
    mat = np.empty((N, M), dtype=float)
    for ti, si, d in results:
        mat[si, ti] = d

    cols = [f"dtw_{name}" for name in template_names]
    df_dtw = pd.DataFrame(mat, columns=cols)
    return df_dtw

def _dtw_dist_pair(t, s, window_frac=0.1):
    # return normalized distance only (use dtw_distance_numba)
    cost, path_len = dtw_distance_numba(t, s, window_frac=window_frac)
    if path_len <= 0:
        return float("inf")
    return cost / float(path_len)

def get_window_dtw(a, b, frac=0.10, min_w=8, max_w=500):
    len_a = len(a)
    len_b = len(b)
    n = max(len_a, len_b)
    w = int(math.ceil(frac * n))
    w = max(w, abs(len_a - len_b))
    w = max(min_w, min(max_w, w))
    return w

def dtw_distance_numba(a, b, window_frac=0.10):
    a = _ensure_1d_float(a)
    b = _ensure_1d_float(b)
    w = get_window_dtw(a, b, window_frac)
    cost, path_len = _dtw_numba_optimized(a, b, w)
    return float(cost), int(path_len)



def _ensure_1d_float(a):
    a = np.asarray(a, dtype=float)
    if a.ndim > 1:
        a = a.ravel()
    # trim trailing zeros/NaNs earlier in pipeline if relevant
    return a


@njit(fastmath=True)
def _dtw_numba_optimized(a, b, window):
    n = a.shape[0]
    m = b.shape[0]
    if n == 0 or m == 0:
        return 1e18, 0

    w = max(window, abs(n - m))
    INF = 1e18

    # only keep two rows for cost and path length
    prev_cost = np.full(m + 1, INF, dtype=np.float64)
    curr_cost = np.full(m + 1, INF, dtype=np.float64)
    prev_len = np.zeros(m + 1, dtype=np.int32)
    curr_len = np.zeros(m + 1, dtype=np.int32)

    prev_cost[0] = 0.0  # D[0,0] = 0

    for i in range(1, n + 1):
        curr_cost[0] = INF  # D[i,0] = inf
        j_start = max(1, i - w)
        j_end = min(m, i + w)

        ai = a[i - 1]
        for j in range(j_start, j_end + 1):
            cost = abs(ai - b[j - 1])

            # check three neighbors: (i-1,j), (i,j-1), (i-1,j-1)
            c1, l1 = prev_cost[j], prev_len[j]       # from top
            c2, l2 = curr_cost[j - 1], curr_len[j - 1]  # from left
            c3, l3 = prev_cost[j - 1], prev_len[j - 1]  # from diag

            # find min cost
            if c1 <= c2 and c1 <= c3:
                min_cost, min_len = c1, l1
            elif c2 <= c3:
                min_cost, min_len = c2, l2
            else:
                min_cost, min_len = c3, l3

            curr_cost[j] = cost + min_cost
            curr_len[j] = min_len + 1

        # swap rows
        prev_cost, curr_cost = curr_cost, prev_cost
        prev_len, curr_len = curr_len, prev_len

    total_cost = prev_cost[m]
    path_len = prev_len[m]
    return total_cost, path_len

def dtw_path(ref, sig, window_frac=0.1):
    a = np.asarray(ref, dtype=float)
    b = np.asarray(sig, dtype=float)
    n = a.shape[0]
    m = b.shape[0]
    if n == 0 or m == 0:
        return []

    # window size in samples
    w = int(max(1, round(max(n, m) * window_frac)))
    w = max(w, abs(n - m))  # ensure covers length difference

    INF = 1e18
    # cost matrix (n+1 x m+1) to simplify boundary conditions
    D = np.full((n + 1, m + 1), INF, dtype=float)
    D[0, 0] = 0.0

    # we also keep a backpointer (same shape, store 0=diag,1=up,2=left)
    bp = np.full((n + 1, m + 1), -1, dtype=np.int8)

    for i in range(1, n + 1):
        j_start = max(1, i - w)
        j_end = min(m, i + w)
        ai = a[i - 1]
        for j in range(j_start, j_end + 1):
            cost = abs(ai - b[j - 1])
            # neighbors: diag (i-1,j-1), up (i-1,j), left (i,j-1)
            c_diag = D[i - 1, j - 1]
            c_up   = D[i - 1, j]
            c_left = D[i, j - 1]

            # find minimum neighbor
            if c_diag <= c_up and c_diag <= c_left:
                best = c_diag
                bp[i, j] = 0  # diag
            elif c_up <= c_left:
                best = c_up
                bp[i, j] = 1  # up
            else:
                best = c_left
                bp[i, j] = 2  # left

            D[i, j] = cost + best

    # backtrack from (n, m)
    i, j = n, m
    path = []
    while i > 0 or j > 0:
        if i <= 0:
            j -= 1
            path.append((0, j))
            continue
        if j <= 0:
            i -= 1
            path.append((i, 0))
            continue
        ptr = bp[i, j]
        if ptr == 0:        # diag
            i -= 1
            j -= 1
            path.append((i, j))
        elif ptr == 1:      # up (came from i-1, j)
            i -= 1
            path.append((i, j - 1))  # careful: map to 0-based j index
            # Note: slight index trick: we append current mapping in terms of (i_ref, j_sig)
            # To keep consistency we prefer adding (i-1, j-1) when diag, else use (i-1, j-1) variants
        elif ptr == 2:      # left (came from i, j-1)
            j -= 1
            path.append((i - 1, j))
        else:
            # fallback if something odd (shouldn't happen)
            i -= 1
            j -= 1
            path.append((i, j))

    path.reverse()
    # Ensure path contains pairs of valid indices (0..n-1, 0..m-1)
    # compress consecutive duplicates per ref index if desired (not necessary)
    return path


@njit(fastmath=True, cache=True)
def dtw_path_numba(ref, sig, window_frac):
    # Ensure inputs are 1D float64 arrays
    a = ref.astype(np.float64)
    b = sig.astype(np.float64)
    n = a.shape[0]
    m = b.shape[0]
    if n == 0 or m == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)

    # window size (samples)
    w = int(max(1, round(max(n, m) * window_frac)))
    if w < abs(n - m):
        w = abs(n - m)

    INF = 1e18

    # allocate DP matrix D of shape (n+1, m+1)
    D = np.full((n + 1, m + 1), INF, dtype=np.float64)
    D[0, 0] = 0.0

    # fill DP within window
    for i in range(1, n + 1):
        j_start = 1
        if i - w > 1:
            j_start = i - w
        j_end = m
        if i + w < m:
            j_end = i + w
        ai = a[i - 1]
        for j in range(j_start, j_end + 1):
            cost = abs(ai - b[j - 1])
            # neighbors
            c_diag = D[i - 1, j - 1]
            c_up = D[i - 1, j]
            c_left = D[i, j - 1]
            # choose best
            best = c_diag
            if c_up < best:
                best = c_up
            if c_left < best:
                best = c_left
            D[i, j] = cost + best

    # backtrack path: build list in reverse then flip
    # worst-case path length <= n + m, so allocate that
    maxlen = n + m
    path_i = np.empty(maxlen, dtype=np.int32)
    path_j = np.empty(maxlen, dtype=np.int32)
    pos = 0
    i = n
    j = m

    while i > 0 and j > 0:
        # append mapping of current pair (i-1, j-1)
        path_i[pos] = i - 1
        path_j[pos] = j - 1
        pos += 1

        # neighbor costs (use INF if out of range)
        c_diag = D[i - 1, j - 1]
        c_up = D[i - 1, j]    # move up => i-1, j
        c_left = D[i, j - 1]  # move left => i, j-1

        # choose predecessor with minimal cost
        if c_diag <= c_up and c_diag <= c_left:
            i -= 1
            j -= 1
        elif c_up <= c_left:
            i -= 1
        else:
            j -= 1

    # finish remaining i or j
    while i > 0:
        path_i[pos] = i - 1
        path_j[pos] = 0
        pos += 1
        i -= 1
    while j > 0:
        path_i[pos] = 0
        path_j[pos] = j - 1
        pos += 1
        j -= 1

    # reverse to get forward path order
    if pos == 0:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    out_i = np.empty(pos, dtype=np.int32)
    out_j = np.empty(pos, dtype=np.int32)
    for k in range(pos):
        out_i[k] = path_i[pos - 1 - k]
        out_j[k] = path_j[pos - 1 - k]

    return out_i, out_j
