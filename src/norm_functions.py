# src/norm_functions.py
import numpy as np
import warnings
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks


def _interp_monotone_with_extrap(x, xp, fp):
    x = np.asarray(x, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)

    y = np.interp(x, xp, fp)

    left = x < xp[0]
    if np.any(left):
        slope = (fp[1] - fp[0]) / (xp[1] - xp[0])
        y[left] = fp[0] + slope * (x[left] - xp[0])

    right = x > xp[-1]
    if np.any(right):
        slope = (fp[-1] - fp[-2]) / (xp[-1] - xp[-2])
        y[right] = fp[-1] + slope * (x[right] - xp[-1])

    return y


def _add_quantile_anchors(data_channel, ref_channel, m, b, q_low=0.02, q_high=0.98):
    try:
        m_lo, m_hi = np.quantile(data_channel[~np.isnan(data_channel)], [q_low, q_high])
        b_lo, b_hi = np.quantile(ref_channel[~np.isnan(ref_channel)], [q_low, q_high])
    except Exception:
        return m, b

    m_aug = np.concatenate([m, [m_lo, m_hi]])
    b_aug = np.concatenate([b, [b_lo, b_hi]])

    order = np.argsort(m_aug)
    m_aug, b_aug = m_aug[order], b_aug[order]
    mu, idx = np.unique(m_aug, return_index=True)
    return mu, b_aug[idx]


def _clip_slopes_and_reconstruct(m, b, min_slope, max_slope):
    m = np.asarray(m, dtype=float)
    b = np.asarray(b, dtype=float)

    slopes = np.diff(b) / np.diff(m)
    slopes = np.clip(slopes, min_slope, max_slope)

    b_adj = np.empty_like(b)
    b_adj[0] = b[0]
    for i in range(len(slopes)):
        b_adj[i + 1] = b_adj[i] + slopes[i] * (m[i + 1] - m[i])
    return b_adj


def _interp_y(kde_x, kde_y, x):
    return np.interp(x, kde_x, kde_y, left=np.nan, right=np.nan)


def find_candidate_landmarks(data, channel_index):
    if channel_index >= data.shape[1]:
        return np.array([])

    d = data[:, channel_index]
    d = d[~np.isnan(d)]
    if d.size < 2:
        return np.array([])

    try:
        kde = gaussian_kde(d, bw_method="scott")
        mn, mx = np.min(d), np.max(d)
        if np.isclose(mn, mx):
            return np.array([mn], dtype=float)

        x = np.linspace(mn, mx, 512)
        y = kde(x)
        peaks, _ = find_peaks(y)
        if peaks.size == 0:
            return np.array([])
        return np.sort(x[peaks].astype(float))
    except Exception:
        return np.array([])


def score_landmarks(candidates, data_channel, peak_density_thr=0.05, peak_distance_thr=0.05):
    scores = np.zeros_like(candidates, dtype=float)
    if candidates.size == 0:
        return scores

    data_clean = data_channel[~np.isnan(data_channel)]
    if data_clean.size < 2:
        return scores

    try:
        kde = gaussian_kde(data_clean, bw_method="scott")
        mn, mx = np.min(data_clean), np.max(data_clean)
        if np.isclose(mn, mx):
            if candidates.size > 0 and np.isclose(candidates[0], mn):
                scores[0] = 1.0
            return scores

        x = np.linspace(mn, mx, 512)
        y = kde(x)

        heights = _interp_y(x, y, candidates)
        if np.all(np.isnan(heights)):
            return scores

        hmax = np.nanmax(heights)
        if hmax == 0 or np.isnan(hmax):
            return scores

        cutoff = peak_density_thr * hmax
        data_range = mx - mn
        min_dist = data_range * peak_distance_thr if data_range > 0 else 0.0

        idx_eval = np.array([np.argmin(np.abs(x - c)) if np.isfinite(c) else -1 for c in candidates])

        step = (mx - mn) / (len(x) - 1)
        bw = int(max(1, (peak_distance_thr * (mx - mn)) / step))
        bw = min(bw, len(x) - 1)

        tmp = np.zeros_like(candidates, dtype=float)

        for i in range(len(candidates)):
            if not np.isfinite(candidates[i]) or idx_eval[i] == -1:
                continue

            ei = idx_eval[i]
            hi = heights[i]
            if np.isnan(hi):
                hi = y[ei]

            if np.isnan(hi) or hi < cutoff:
                continue

            start = max(0, ei - bw // 2)
            end = min(len(x), ei + bw // 2 + 1)
            window = y[start:end]

            w = hi - window
            w[w < 0] *= 3
            sharp = np.sum(w)
            tmp[i] = max(0.0, sharp * hi)

        order = np.argsort(candidates)
        last = -1

        for idx in order:
            if not np.isfinite(candidates[idx]):
                continue
            scores[idx] = tmp[idx]
            if scores[idx] == 0:
                continue

            if last == -1:
                last = idx
                continue

            dist = candidates[idx] - candidates[last]
            close = (min_dist == 0 and np.isclose(dist, 0)) or (min_dist > 0 and dist < min_dist)
            if close:
                if scores[idx] > scores[last]:
                    scores[last] = 0
                    last = idx
                else:
                    scores[idx] = 0
            else:
                last = idx

        return scores
    except Exception:
        return np.zeros_like(candidates, dtype=float)


def filter_landmarks(candidates, scores, max_lms):
    if candidates.size == 0 or scores.size == 0 or candidates.size != scores.size:
        return np.array([]), np.array([])

    keep = np.where(scores > 0)[0]
    if keep.size == 0:
        return np.array([]), np.array([])

    cand = candidates[keep]
    sc = scores[keep]

    order = np.argsort(sc)[::-1]
    top = keep[order[: min(max_lms, len(order))]]

    cand_top = candidates[top]
    sc_top = scores[top]

    srt = np.argsort(cand_top)
    return cand_top[srt], sc_top[srt]


def extract_landmarks(flowset, channel_indices, max_lms_per_channel, peak_density_thr, peak_distance_thr):
    num_samples = len(flowset)
    num_channels = len(channel_indices)

    if isinstance(max_lms_per_channel, int):
        max_lms_list = [max_lms_per_channel] * num_channels
    elif len(max_lms_per_channel) == num_channels:
        max_lms_list = list(max_lms_per_channel)
    else:
        raise ValueError("max_lms_per_channel must be int or match channel_indices length")

    original = [[] for _ in range(num_channels)]
    score = [[] for _ in range(num_channels)]
    filt = [[] for _ in range(num_channels)]

    for c_idx, p in enumerate(channel_indices):
        max_lms = max_lms_list[c_idx]
        if max_lms == 0:
            original[c_idx] = [np.array([]) for _ in range(num_samples)]
            score[c_idx] = [np.array([]) for _ in range(num_samples)]
            filt[c_idx] = [np.array([]) for _ in range(num_samples)]
            continue

        for i in range(num_samples):
            sample = flowset[i]
            if p >= sample.shape[1]:
                warnings.warn(f"channel index {p} out of bounds for sample {i}")
                original[c_idx].append(np.array([]))
                score[c_idx].append(np.array([]))
                filt[c_idx].append(np.array([]))
                continue

            channel_data = sample[:, p]
            cand = find_candidate_landmarks(sample, p)
            original[c_idx].append(cand)

            sc = score_landmarks(cand, channel_data, peak_density_thr, peak_distance_thr)
            score[c_idx].append(sc)

            f_lms, _ = filter_landmarks(cand, sc, max_lms)
            filt[c_idx].append(f_lms)

    return {"original": original, "score": score, "filter": filt}


def register_channel(data_channel, matched_lms_pairs, ref_channel=None, add_tail_anchors=True, slope_clip=None):
    matched_lms_pairs = np.asarray(matched_lms_pairs, dtype=float)
    if matched_lms_pairs.size == 0:
        return np.asarray(data_channel, dtype=float).copy()
    if matched_lms_pairs.size % 2 != 0:
        warnings.warn("matched_lms_pairs size is not even")
        return np.asarray(data_channel, dtype=float).copy()

    m = matched_lms_pairs[0::2]
    b = matched_lms_pairs[1::2]

    ok = np.isfinite(m) & np.isfinite(b)
    m, b = m[ok], b[ok]
    if m.size == 0:
        return np.asarray(data_channel, dtype=float).copy()

    if m.size == 1:
        return np.asarray(data_channel, dtype=float) + (b[0] - m[0])

    order = np.argsort(m)
    m, b = m[order], b[order]
    mu, idx = np.unique(m, return_index=True)
    bu = b[idx]

    if add_tail_anchors and ref_channel is not None:
        mu, bu = _add_quantile_anchors(
            np.asarray(data_channel, dtype=float),
            np.asarray(ref_channel, dtype=float),
            mu, bu, q_low=0.02, q_high=0.98
        )

    if slope_clip is not None and len(mu) >= 2:
        min_slope, max_slope = slope_clip
        bu = _clip_slopes_and_reconstruct(mu, bu, min_slope, max_slope)

    aligned = _interp_monotone_with_extrap(np.asarray(data_channel, dtype=float), mu, bu)
    aligned = np.clip(aligned, 0, None)
    return aligned


__all__ = ["extract_landmarks", "register_channel"]
