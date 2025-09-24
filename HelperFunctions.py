import numpy as np
import re


def is_int_token(tok):
    return re.fullmatch(r"[+-]?\d+", tok) is not None

def weighted_center_and_cov(X, w):

    w = np.asarray(w, float)
    Wsum = w.sum()
    if Wsum <= 0:
        raise ValueError("Sum of weights must be positive.")
    c = (w[:, None] * X).sum(axis=0) / Wsum
    Xc = X - c
    C = (Xc.T * w) @ Xc / Wsum
    return c, C


def findMin(coord,frame):
    min  = 100
    for fcoord in frame:
        distance  = np.linalg.norm(coord - fcoord)
        if distance < min:min=distance
    return min


def parse_frame_positions(fp):
    time = None
    positions = []
    for raw in fp:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("t") and "=" in line:
            if positions:
                yield (time, np.array(positions, dtype=float))
                positions = []
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
            time = float(nums[-1]) if nums else None
            continue

        if re.match(r"^[A-Za-z]", line):
            continue

        toks = line.split()
        if len(toks) < 3:
            continue
        use_id_shift = is_int_token(toks[0]) and len(toks) >= 4
        try:
            if use_id_shift:
                x = float(toks[1]);
                y = float(toks[2]);
                z = float(toks[3])
            else:
                x = float(toks[0]);
                y = float(toks[1]);
                z = float(toks[2])
        except ValueError:
            floats = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
            if len(floats) < 3:
                continue
            start = 1 if use_id_shift else 0
            x, y, z = map(float, floats[start:start + 3])

        positions.append((x * 0.8518, y * 0.8518, z * 0.8518))
    if positions:
        yield (time, np.array(positions, dtype=float))

def gaussian_smooth(y, sigma):
    if sigma is None or sigma <= 0:
        return y
    radius = max(1, int(4 * sigma))
    xk = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (xk / sigma) ** 2)
    k /= k.sum()
    return np.convolve(y, k, mode="same")


def findMin(coord,frame):
    min  = 100
    for fcoord in frame:
        distance  = np.linalg.norm(coord - fcoord)
        if distance < min:min=distance
    return min


def frame_pca1(coords, prev_pc1=None):

    c = coords.mean(axis=0)
    X = coords - c


    C = np.cov(X.T, bias=False)
    evals, evecs = np.linalg.eigh(C)

    idx = np.argmax(evals)
    pc1 = evecs[:, idx]
    pc1 = pc1 / np.linalg.norm(pc1)

    if prev_pc1 is not None:
        if np.dot(pc1, prev_pc1) < 0.0:
            pc1 = -pc1

    s = X @ pc1


    lam = evals[[np.argmax(evals), *sorted([i for i in range(3) if i != idx], key=lambda i: -evals[i])]]

    total = evals.sum()
    ratio = evals / total if total > 0 else np.zeros_like(evals)

    return c, pc1, evals, ratio, s



def compute_z_density(dat_path,
    bins=100,
    average_frames=True,
    normalize=True,
):

    times = []
    per_frame_hist = []
    per_frame_pc1 = []
    per_frame_centroid = []
    per_frame_evals = []
    per_frame_ratio = []

    per_frame_box = []
    edges_ref = None
    prev_pc1 = None

    with open(dat_path, "r") as f:
        for t, coords in parse_frame_positions(f):

            centroid, pc1, evals, vratio, s = frame_pca1(coords, prev_pc1)
            prev_pc1 = pc1.copy()

            if isinstance(bins, int):
                smin, smax = float(s.min()), float(s.max())
                pad = 1e-9 * max(1.0, abs(smax - smin))
                s_edges = np.linspace(smin - pad, smax + pad, bins + 1)
            else:
                s_edges = np.asarray(bins, dtype=float)


            counts, s_edges = np.histogram(s, bins=s_edges)
            bw = np.diff(s_edges)


            density = counts / bw if normalize else counts

            per_frame_hist.append(density)
            per_frame_pc1.append(pc1)
            per_frame_centroid.append(centroid)
            per_frame_evals.append(evals)
            per_frame_ratio.append(vratio)
            per_frame_box.append(None)
            times.append(t)

            if edges_ref is None:
                edges_ref = s_edges
            else:
                if not np.allclose(edges_ref, s_edges):
                    counts_ref, _ = np.histogram(s, bins=edges_ref)
                    bw_ref = np.diff(edges_ref)
                    density_ref = counts_ref / bw_ref if normalize else counts_ref
                    per_frame_hist[-1] = density_ref

    if edges_ref is None:
        raise ValueError("No particle data found in the provided .dat file.")

    s_edges = edges_ref
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])

    if average_frames:
        density = np.mean(np.stack(per_frame_hist, axis=0), axis=0)
    else:
        density = per_frame_hist

    return {
        "z_centers": s_centers,
        "density": density,
        "edges": s_edges,
        "times": times,
        "pc1": per_frame_pc1,
        "centroid": per_frame_centroid,
        "explained_var": per_frame_evals,
        "explained_ratio": per_frame_ratio,
    }


def find_global_extrema(pca_out, return_indices=True):
    if isinstance(pca_out, dict):
        coords_list = pca_out.get("coords_pca", [])
    else:
        coords_list = pca_out

    if not isinstance(coords_list, (list, tuple)) or len(coords_list) == 0:
        raise ValueError("pca_out must be a non-empty list of (N,3) arrays or a dict with 'coords_pca'.")

    max_vals = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    min_vals = np.array([ np.inf,  np.inf,  np.inf], dtype=float)
    max_locs = [(None, None), (None, None), (None, None)]
    min_locs = [(None, None), (None, None), (None, None)]

    for f_idx, arr in enumerate(coords_list):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("Each frame array must have shape (N, 3).")
        if arr.size == 0:
            continue

        for d in range(3):
            col = arr[:, d]
            finite = np.isfinite(col)
            if not np.any(finite):
                continue

            local_max = np.max(col[finite])
            local_min = np.min(col[finite])

            if local_max > max_vals[d]:
                max_vals[d] = local_max
                p_idx_max = int(np.nanargmax(col))
                max_locs[d] = (f_idx, p_idx_max)

            if local_min < min_vals[d]:
                min_vals[d] = local_min
                p_idx_min = int(np.nanargmin(col))
                min_locs[d] = (f_idx, p_idx_min)
    max_vals[~np.isfinite(max_vals)] = np.nan
    min_vals[~np.isfinite(min_vals)] = np.nan

    out = {
        "max_xyz": max_vals,
        "min_xyz": min_vals,
    }
    if return_indices:
        out["max_locs"] = max_locs
        out["min_locs"] = min_locs
    return out





