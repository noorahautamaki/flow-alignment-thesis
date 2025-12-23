# scripts/run_alignment.py
#!/usr/bin/env python3
import argparse
import os
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import cdist

from src.norm_functions import extract_landmarks, register_channel


exclude = {"FSC-A", "FSC-H", "FSC-W", "SSC-A", "SSC-H", "SSC-W"}
no_quantile = {"CD45", "CD34"}

fixed_ref_landmarks = {
    "CD33": np.array([0.239, 0.846], dtype=float)
}

emd_worsen_tol = 0.01
chamfer_drop_ratio = 0.50
q_low, q_high = 0.10, 0.90

marker_params = {
    "CD45":   {"max_lms": 1, "density_thr": 0.03,  "distance_thr": 0.13},
    "CD10":   {"max_lms": 3, "density_thr": 0.01,  "distance_thr": 0.01},
    "CD117":  {"max_lms": 2, "density_thr": 0.01,  "distance_thr": 0.01},
    "CD11b":  {"max_lms": 2, "density_thr": 0.2,   "distance_thr": 0.2},
    "CD123":  {"max_lms": 3, "density_thr": 0.1,   "distance_thr": 0.01},
    "CD13":   {"max_lms": 2, "density_thr": 0.005, "distance_thr": 0.25},
    "CD14":   {"max_lms": 2, "density_thr": 0.01,  "distance_thr": 0.4},
    "CD16":   {"max_lms": 1, "density_thr": 0.001, "distance_thr": 0.3},
    "CD33":   {"max_lms": 2, "density_thr": 0.05,  "distance_thr": 0.2},
    "CD34":   {"max_lms": 2, "density_thr": 0.001, "distance_thr": 0.5},
    "CD64":   {"max_lms": 3, "density_thr": 0.05,  "distance_thr": 0.1},
    "CD71":   {"max_lms": 2, "density_thr": 0.1,   "distance_thr": 0.1},
    "HLA-DR": {"max_lms": 2, "density_thr": 0.01,  "distance_thr": 0.3},
    "CD105":  {"max_lms": 2, "density_thr": 0.01,  "distance_thr": 0.01},
    "CD235a": {"max_lms": 2, "density_thr": 0.05,  "distance_thr": 0.3},
    "CD36":   {"max_lms": 2, "density_thr": 0.1,   "distance_thr": 0.01},
}


def to_1d(x):
    arr = np.asarray(x).ravel()
    return arr[np.isfinite(arr)]


def mask_boundaries_1d(x, tol=1e-9):
    x = np.asarray(x, dtype=float).ravel()
    if x.size == 0:
        return x

    finite = np.isfinite(x)
    if not np.any(finite):
        return x

    mn = np.min(x[finite])
    mx = np.max(x[finite])

    boundary = np.zeros_like(x, dtype=bool)
    boundary |= finite & np.isclose(x, mn, atol=tol)
    boundary |= finite & np.isclose(x, mx, atol=tol)

    y = x.copy()
    y[boundary] = np.nan
    return y


def chamfer_1d(a, b, n_samples=3000):
    a, b = to_1d(a), to_1d(b)
    if len(a) == 0 or len(b) == 0:
        return np.nan

    if len(a) > n_samples:
        a = np.random.choice(a, n_samples, replace=False)
    if len(b) > n_samples:
        b = np.random.choice(b, n_samples, replace=False)

    A, B = a.reshape(-1, 1), b.reshape(-1, 1)
    D = cdist(A, B)
    return float(D.min(axis=1).mean() + D.min(axis=0).mean())


def quantile_align(raw_vals, ref_vals, ql=q_low, qh=q_high):
    r1, r2 = np.quantile(raw_vals, [ql, qh])
    t1, t2 = np.quantile(ref_vals, [ql, qh])
    a = (t2 - t1) / (r2 - r1 + 1e-12)
    b = t1 - a * r1
    return a * raw_vals + b, (a, b)


def landmark_align(raw_vals, ref_vals, params):
    raw_for_lms = mask_boundaries_1d(raw_vals)
    ref_for_lms = mask_boundaries_1d(ref_vals)

    raw2d = raw_for_lms.reshape(-1, 1)
    ref2d = ref_for_lms.reshape(-1, 1)

    raw_lms = extract_landmarks(
        [raw2d], [0], [params["max_lms"]],
        params["density_thr"], params["distance_thr"]
    )["filter"][0][0]

    ref_lms = extract_landmarks(
        [ref2d], [0], [params["max_lms"]],
        params["density_thr"], params["distance_thr"]
    )["filter"][0][0]

    raw_lms = to_1d(raw_lms)
    ref_lms = to_1d(ref_lms)

    k = min(len(raw_lms), len(ref_lms))
    if k == 0:
        return None, raw_lms, ref_lms, 0

    pairs = np.zeros(2 * k, dtype=float)
    pairs[0::2], pairs[1::2] = raw_lms[:k], ref_lms[:k]

    aligned = register_channel(raw_vals, pairs, ref_channel=ref_vals, add_tail_anchors=True)
    return np.asarray(aligned, dtype=float).ravel(), raw_lms, ref_lms, k


def decide_and_pick(marker, raw_vals, ref_vals, params):
    emd_before = wasserstein_distance(ref_vals, raw_vals)
    ch_before = chamfer_1d(ref_vals, raw_vals)

    if marker in fixed_ref_landmarks:
        raw_for_lms = mask_boundaries_1d(raw_vals)

        raw_lms = extract_landmarks(
            [raw_for_lms.reshape(-1, 1)], [0], [params["max_lms"]],
            params["density_thr"], params["distance_thr"]
        )["filter"][0][0]
        raw_lms = to_1d(raw_lms)
        ref_lms = to_1d(fixed_ref_landmarks[marker])

        if len(raw_lms) == 0 or len(ref_lms) == 0:
            met = {
                "emd_before": emd_before, "emd_after": emd_before,
                "ch_before": ch_before, "ch_after": ch_before,
                "k": 0, "raw_lms": [], "ref_lms": []
            }
            return "none", f"no valid raw or fixed ref landmarks for {marker}", raw_vals, met

        k = min(len(raw_lms), len(ref_lms))
        pairs = np.zeros(2 * k, dtype=float)
        pairs[0::2], pairs[1::2] = raw_lms[:k], ref_lms[:k]

        aligned = register_channel(raw_vals, pairs, ref_channel=ref_vals, add_tail_anchors=True)
        aligned = np.asarray(aligned, dtype=float).ravel()

        met = {
            "emd_before": emd_before,
            "emd_after": wasserstein_distance(ref_vals, aligned),
            "ch_before": ch_before,
            "ch_after": chamfer_1d(ref_vals, aligned),
            "k": k,
            "raw_lms": raw_lms,
            "ref_lms": ref_lms,
        }
        return "landmark", f"fixed reference landmarks for {marker}", aligned, met

    if marker in {"CD10", "CD117", "CD123", "HLA-DR"}:
        qt, _ = quantile_align(raw_vals, ref_vals)
        met = {
            "emd_before": emd_before,
            "emd_after": wasserstein_distance(ref_vals, qt),
            "ch_before": ch_before,
            "ch_after": chamfer_1d(ref_vals, qt),
            "k": np.nan,
            "raw_lms": [],
            "ref_lms": [],
        }
        return "quantile", f"forced quantile alignment for {marker}", qt, met

    aligned_lm, raw_lms, ref_lms, k = landmark_align(raw_vals, ref_vals, params)

    if aligned_lm is None or k == 0:
        if marker in no_quantile:
            met = {
                "emd_before": emd_before, "emd_after": emd_before,
                "ch_before": ch_before, "ch_after": ch_before,
                "k": k, "raw_lms": raw_lms, "ref_lms": ref_lms
            }
            return "none", f"no valid landmarks for {marker} and quantile is disabled", raw_vals, met

        qt, _ = quantile_align(raw_vals, ref_vals)
        met = {
            "emd_before": emd_before,
            "emd_after": wasserstein_distance(ref_vals, qt),
            "ch_before": ch_before,
            "ch_after": chamfer_1d(ref_vals, qt),
            "k": k,
            "raw_lms": raw_lms,
            "ref_lms": ref_lms,
        }
        return "quantile", "no valid landmarks; fallback to quantile", qt, met

    emd_lm = wasserstein_distance(ref_vals, aligned_lm)
    ch_lm = chamfer_1d(ref_vals, aligned_lm)

    improves_emd = emd_lm < emd_before
    tolerable = (
        (emd_lm - emd_before) <= emd_worsen_tol
        and np.isfinite(ch_before)
        and ch_before > 0
        and ch_lm <= chamfer_drop_ratio * ch_before
    )

    if marker in no_quantile:
        reason = "landmark chosen: emd improved" if improves_emd else "landmark kept; quantile disabled"
        met = {
            "emd_before": emd_before, "emd_after": emd_lm,
            "ch_before": ch_before, "ch_after": ch_lm,
            "k": k, "raw_lms": raw_lms, "ref_lms": ref_lms
        }
        return "landmark", reason, aligned_lm, met

    if improves_emd or tolerable:
        reason = "landmark chosen: emd improved" if improves_emd else "landmark chosen: small emd increase but chamfer improved"
        met = {
            "emd_before": emd_before, "emd_after": emd_lm,
            "ch_before": ch_before, "ch_after": ch_lm,
            "k": k, "raw_lms": raw_lms, "ref_lms": ref_lms
        }
        return "landmark", reason, aligned_lm, met

    qt, _ = quantile_align(raw_vals, ref_vals)
    emd_qt = wasserstein_distance(ref_vals, qt)
    ch_qt = chamfer_1d(ref_vals, qt)

    if emd_qt < emd_lm:
        met = {
            "emd_before": emd_before, "emd_after": emd_qt,
            "ch_before": ch_before, "ch_after": ch_qt,
            "k": k, "raw_lms": raw_lms, "ref_lms": ref_lms
        }
        return "quantile", "quantile chosen: lower emd", qt, met

    met = {
        "emd_before": emd_before, "emd_after": emd_lm,
        "ch_before": ch_before, "ch_after": ch_lm,
        "k": k, "raw_lms": raw_lms, "ref_lms": ref_lms
    }
    return "landmark", "landmark kept: quantile did not improve emd", aligned_lm, met


def parse_int_list(s):
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def extract_ids_from_filename(path):
    donor_id, tube_id = None, None
    base = os.path.basename(path)
    try:
        donor_id = int(base.split("Femoral head ")[1].split(" tube")[0])
        tube_id = int(base.split(" tube ")[1].split(".h5ad")[0])
    except Exception:
        pass
    return donor_id, tube_id


def align_file(raw_path, ref_adata, out_path, report_rows):
    if not os.path.exists(raw_path):
        print(f"missing file: {raw_path}")
        return

    ad_raw = sc.read_h5ad(raw_path)
    X = ad_raw.X.copy()
    markers = [m for m in ad_raw.var_names if (m not in exclude) and (m in ref_adata.var_names)]
    donor_id, tube_id = extract_ids_from_filename(raw_path)

    print(f"\naligning: donor {donor_id} | tube {tube_id} | {os.path.basename(raw_path)}")

    for marker in markers:
        params = marker_params.get(marker, {"max_lms": 2, "density_thr": 0.05, "distance_thr": 0.05})
        raw_vals = np.asarray(ad_raw[:, marker].X).ravel()
        ref_vals = np.asarray(ref_adata[:, marker].X).ravel()

        raw_vals = raw_vals[np.isfinite(raw_vals)]
        ref_vals = ref_vals[np.isfinite(ref_vals)]
        if raw_vals.size == 0 or ref_vals.size == 0:
            continue

        method, reason, aligned_vals, met = decide_and_pick(marker, raw_vals, ref_vals, params)

        col = ad_raw.var_names.get_loc(marker)
        if hasattr(X, "toarray"):
            X = X.toarray()

        if aligned_vals.shape[0] != X.shape[0]:
            print(f"[{marker:8s}] skipped: aligned length mismatch ({aligned_vals.shape[0]} vs {X.shape[0]})")
            continue

        X[:, col] = aligned_vals

        print(
            f"[{marker:8s}] {method:8s} "
            f"emd {met['emd_before']:.4f}->{met['emd_after']:.4f} "
            f"chamfer {met['ch_before']:.4f}->{met['ch_after']:.4f} "
            f"({reason})"
        )

        report_rows.append({
            "donor": donor_id,
            "tube": tube_id,
            "marker": marker,
            "method": method,
            "reason": reason,
            "k": met.get("k", np.nan),
            "raw_landmarks": ";".join(f"{x:.4f}" for x in (met.get("raw_lms") or [])),
            "ref_landmarks": ";".join(f"{x:.4f}" for x in (met.get("ref_lms") or [])),
            "emd_before": met["emd_before"],
            "emd_after": met["emd_after"],
            "chamfer_before": met["ch_before"],
            "chamfer_after": met["ch_after"],
        })

    ad_raw.X = X
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ad_raw.write(out_path)
    print(f"saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", required=True)
    p.add_argument("--ref-path", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tubes", default="1-7")
    p.add_argument("--donors", default="1-51")
    args = p.parse_args()

    tubes = parse_int_list(args.tubes)
    donors = parse_int_list(args.donors)

    ref_adata = sc.read_h5ad(args.ref_path)
    report = []

    for tube in tubes:
        tube_dir = os.path.join(args.out_dir, f"tube_{tube}")
        os.makedirs(tube_dir, exist_ok=True)
        print(f"\nprocessing tube {tube} -> {tube_dir}")

        for donor in donors:
            raw_path = os.path.join(args.raw_dir, f"Femoral head {donor} tube {tube}.h5ad")
            out_path = os.path.join(tube_dir, f"Femoral head {donor} tube {tube}_aligned.h5ad")
            align_file(raw_path, ref_adata, out_path, report)

    df = pd.DataFrame(report)
    csv_path = os.path.join(args.out_dir, "alignment_report.csv")
    df.to_csv(csv_path, index=False)
    print(f"\ndone. report: {csv_path}")


if __name__ == "__main__":
    main()
