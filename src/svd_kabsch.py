# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 19:07:34 2025

@author: lmcam
"""

import sys
import os
import numpy as np
import math
import ezc3d  # pip install ezc3d numpy
import matplotlib.pyplot as plt
try:
    from joint_angles import compute_angles_per_frame as _compute_angles_pf
    HAVE_JA = True
except Exception:
    HAVE_JA = False


# ----- CONFIG -----
STATIC_C3D = r"C:/Users/lmcam/Documents/Grad project/c3d/subject 02/S_Cal02.c3d"
# override via CLI: python script.py "C:\path\walk.c3d"
WALK_C3D = r"C:/Users/lmcam/Documents/Grad project/c3d/subject 02/Walk_R04.c3d"
STATIC_FRAME = 0
# True -> Umeyama (uniform scale). Usually False for marker sets.
USE_SCALING = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Pelvis markers
PELVIS_MARKERS = ["R_ASIS", "L_ASIS", "R_PSIS", "L_PSIS"]

# Left- and Right-side segment marker sets
SEGMENTS = {
    "pelvis": {
        "parent": None,
        "markers": PELVIS_MARKERS,
    },
    "l_thigh": {
        "parent": "pelvis",
        "markers": ["L_Thigh_PS", "L_Thigh_AS", "L_Thigh_PI", "L_Thigh_AI"],
    },
    "l_shank": {
        "parent": "l_thigh",
        "markers": ["L_Shank_AS", "L_Shank_PS", "L_Shank_AI", "L_Shank_PI"],
    },
    "l_foot": {
        "parent": "l_shank",
        "markers": ["L_Calc", "L_Ank_Lat", "L_Midfoot_Sup", "L_Midfoot_Lat"],
    },
    "r_thigh": {
        "parent": "pelvis",
        "markers": ["R_Thigh_PS", "R_Thigh_AS", "R_Thigh_PI", "R_Thigh_AI"],
    },
    "r_shank": {
        "parent": "r_thigh",
        "markers": ["R_Shank_AS", "R_Shank_PS", "R_Shank_AI", "R_Shank_PI"],
    },
    "r_foot": {
        "parent": "r_shank",
        "markers": ["R_Calc", "R_Ank_Lat", "R_Midfoot_Sup", "R_Midfoot_Lat"],
    },
}

# Optional QC-only knee markers (NOT used in fits)
KNEE_MARKER_L = "L_Knee_Lat"
KNEE_MARKER_R = "R_Knee_Lat"

# ----- UTILITIES -----


def get_labels(c3d_obj):
    # Returns list of point labels
    return list(c3d_obj["parameters"]["POINT"]["LABELS"]["value"])


def get_points(c3d_obj):
    # Returns array of shape (num_frames, num_points, 3)
    pts = c3d_obj["data"]["points"]  # (4, Npoints, Nframes)
    xyz = np.transpose(pts[:3, :, :], (2, 1, 0))
    return xyz


def select_corresponding(static_labels, walk_labels, static_xyz, walk_xyz_frame, wanted_labels, static_frame_idx):
    # Returns P (static), Q (walking frame) for the subset present in both and not NaN
    static_idx = []
    walk_idx = []
    for w in wanted_labels:
        if w in static_labels and w in walk_labels:
            static_idx.append(static_labels.index(w))
            walk_idx.append(walk_labels.index(w))
    if len(static_idx) == 0:
        return None, None
    P = static_xyz[static_frame_idx, static_idx, :]  # (M,3)
    Q = walk_xyz_frame[walk_idx, :]                  # (M,3)
    # Drop rows with NaNs
    mask = (~np.isnan(P).any(axis=1)) & (~np.isnan(Q).any(axis=1))
    P = P[mask]
    Q = Q[mask]
    if P.shape[0] < 3:
        return None, None
    return P, Q


def kabsch(P, Q):
    # P, Q: (N,3). Returns R, t
    Pc = P - P.mean(axis=0, keepdims=True)
    Qc = Q - Q.mean(axis=0, keepdims=True)
    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = Q.mean(axis=0) - R @ P.mean(axis=0)
    return R, t


def _kabsch_R_t_mean_error_mm(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Kabsch alignment and mean point-to-point distance (mm) after mapping P → Q."""
    R, t = kabsch(P, Q)
    pred = (R @ P.T).T + t
    err_mm = float(np.linalg.norm(pred - Q, axis=1).mean())
    return R, t, err_mm


def umeyama(P, Q):
    # Uniform-scale variant. Returns s, R, t
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    cov = (Qc.T @ Pc) / P.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    Sfix = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        Sfix[-1, -1] = -1
    R = U @ Sfix @ Vt
    varP = (Pc ** 2).sum() / P.shape[0]
    s = np.sum(D * np.diag(Sfix)) / varP
    t = Q.mean(axis=0) - s * R @ P.mean(axis=0)
    return s, R, t


def to_h(R, t, s=1.0):
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def invert_h(T):
    R = T[:3, :3]
    t = T[:3, 3]
    # Handle potential uniform scale
    detR = np.linalg.det(R)
    s = np.cbrt(detR) if detR > 0 else 1.0
    if not np.isclose(s, 1.0):
        Rn = R / s
        Rinv = Rn.T
        tinv = -Rinv @ (t / s)
    else:
        Rinv = R.T
        tinv = -Rinv @ t
    Ti = np.eye(4)
    Ti[:3, :3] = Rinv
    Ti[:3, 3] = tinv
    return Ti


def fit_segment_global(P_static, Q_walk, use_scaling):
    if P_static is None or Q_walk is None:
        return None, np.nan
    if use_scaling:
        s, R, t = umeyama(P_static, Q_walk)
        T = to_h(R, t, s)
        pred = (T[:3, :3] @ P_static.T).T + T[:3, 3]
        err_mm = float(np.linalg.norm(pred - Q_walk, axis=1).mean())
    else:
        R, t, err_mm = _kabsch_R_t_mean_error_mm(P_static, Q_walk)
        T = to_h(R, t, 1.0)
    return T, err_mm


def save_npz(path, **arrays):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.savez(path, **arrays)
    print(f"Saved: {os.path.abspath(path)}")


def subject_output_dir(static_c3d_path: str) -> str:
    """Subject-specific output folder under scripts/static calib."""
    static_base = os.path.splitext(os.path.basename(static_c3d_path))[0]
    subject_name = os.path.basename(os.path.dirname(static_c3d_path))
    out_dir = os.path.join(SCRIPT_DIR, f"{subject_name} - {static_base}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

# ---- TEMPLATE LOADING (TCS→ACS static) ----


def load_template(npz_path):
    tpl = np.load(npz_path, allow_pickle=True)
    labels = tpl["marker_labels"].tolist()
    C_local = tpl["C_local"]
    # Gracefully handle templates without static TCS→ACS offsets (use identity/zero)
    R_TA = tpl["R_TA"] if ("R_TA" in tpl.files) else np.eye(3, dtype=float)
    t_TA = tpl["t_TA"] if ("t_TA" in tpl.files) else np.zeros(3, dtype=float)
    return {
        "labels": labels,
        "C_local": C_local,   # (N,3) cluster in TCS coords (static)
        "R_TA": R_TA,         # (3,3) TCS→ACS (identity if missing)
        "t_TA": t_TA,         # (3,)  TCS→ACS (zero if missing)
    }


def _try_load_first(paths):
    """Return loaded template from the first existing path in list; else None."""
    for p in paths:
        if isinstance(p, str) and os.path.exists(p):
            return load_template(p)
    return None


def _template_scene_index_pairs(tpl_labels, scene_labels, scene_xyz_row: np.ndarray) -> tuple[list[int], list[int]] | tuple[None, None]:
    """Map each template label to a scene row index; require ≥3 pairs and finite scene coordinates."""
    pairs: list[tuple[int, int]] = []
    for i, name in enumerate(tpl_labels):
        if name not in scene_labels:
            continue
        j = scene_labels.index(name)
        if np.isnan(scene_xyz_row[j]).any():
            continue
        pairs.append((i, j))
    if len(pairs) < 3:
        return None, None
    i_idx = [a for a, _ in pairs]
    j_idx = [b for _, b in pairs]
    return i_idx, j_idx


def subset_CQ_from_labels(tpl_labels, C_local, walk_labels, walk_xyz_frame):
    # choose common markers (order-consistent) and return C_sub (TCS) and Q_sub (world)
    i_idx, j_idx = _template_scene_index_pairs(
        tpl_labels, walk_labels, walk_xyz_frame)
    if i_idx is None:
        return None, None
    C_sub = C_local[i_idx, :]
    Q_sub = walk_xyz_frame[j_idx, :]
    return C_sub, Q_sub


# --- Static TCS<->world helpers (for knee QC) ---
def fit_segment_on_static(static_labels, static_xyz, tpl, seg):
    """Compute T_Tw (TCS->world) for 'seg' on the static frame by fitting its template."""
    tpl_seg = tpl.get(seg)
    if tpl_seg is None:
        return None
    C_local = tpl_seg["C_local"]
    tpl_labels = tpl_seg["labels"]
    i_idx, j_idx = _template_scene_index_pairs(
        tpl_labels, static_labels, static_xyz[STATIC_FRAME])
    if i_idx is None:
        return None
    C_sub = C_local[i_idx, :]
    Q_sub = static_xyz[STATIC_FRAME, j_idx, :]
    R, t, _ = _kabsch_R_t_mean_error_mm(C_sub, Q_sub)
    return to_h(R, t, 1.0)


def world_to_TCS(T_Tw, p_world):
    R = T_Tw[:3, :3]
    t = T_Tw[:3, 3]
    return R.T @ (p_world - t)


def T_apply_point(T, p_local):
    return (T[:3, :3] @ p_local) + T[:3, 3]

# ==== QC HELPERS ====


def rot_ortho_err(R: np.ndarray) -> float:
    """Frobenius norm of (R^T R - I). Should be ~1e-12 to 1e-6; flag >1e-3."""
    return float(np.linalg.norm(R.T @ R - np.eye(3), ord='fro'))


def rot_det(R: np.ndarray) -> float:
    """Determinant; should be very close to +1.0."""
    return float(np.linalg.det(R))


def relative_rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Angle (deg) between orientations Ra->Rb (SO(3) geodesic)."""
    R = Ra.T @ Rb
    tr = max(-1.0, min(1.0, (np.trace(R) - 1.0) * 0.5))
    return float(np.degrees(math.acos(tr)))


def count_markers_used(C_sub: np.ndarray) -> int:
    return 0 if C_sub is None else int(C_sub.shape[0])


def robust_threshold(series: np.ndarray, k: float = 5.0):
    """Median + k*MAD for outlier flagging."""
    x = series[np.isfinite(series)]
    if x.size == 0:
        return np.nan, np.nan
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-12
    return med, med + k * mad

# Simple timeseries plot helper


def _plot_timeseries(y, title, ylabel, outpng):
    x = np.arange(len(y))
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpng, dpi=160)
    plt.close()


def _plot_angle_triplet(arr, title, labels, outpng):
    arr = np.array(arr, float)
    x = np.arange(arr.shape[0])
    plt.figure()
    for i in range(3):
        plt.plot(x, arr[:, i], label=labels[i])
    plt.title(title)
    plt.xlabel("Frame")
    plt.ylabel("deg")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpng, dpi=160)
    plt.close()


def _cos(a, b):
    a = a / (np.linalg.norm(a)+1e-12)
    b = b / (np.linalg.norm(b)+1e-12)
    return float(np.dot(a, b))


def pelvis_axis_sanity(acs_R_lists, acs_O_lists):
    y_forward_cos, z_up_cos = [], []
    F = len(acs_R_lists["pelvis"])
    forward = None
    for f in range(F):
        R = acs_R_lists["pelvis"][f]
        O = acs_O_lists["pelvis"][f]
        if R is None or O is None:
            y_forward_cos.append(np.nan)
            z_up_cos.append(np.nan)
            continue
        z = R[:, 2]
        y = R[:, 1]
        # estimate forward from pelvis origin motion
        if f < F-1 and (acs_O_lists["pelvis"][f+1] is not None):
            v = acs_O_lists["pelvis"][f+1] - O
            if np.linalg.norm(v) > 1e-6:
                forward = v
        y_forward_cos.append(
            _cos(y, forward) if forward is not None else np.nan)
        z_up_cos.append(_cos(z, np.array([0, 0, 1.0])))
    return np.array(y_forward_cos, float), np.array(z_up_cos, float)

# ----- REFACTORED HELPERS -----


def load_c3ds(static_path: str, walk_path: str):
    static_c3d = ezc3d.c3d(static_path)
    walk_c3d = ezc3d.c3d(walk_path)
    static_labels = get_labels(static_c3d)
    walk_labels = get_labels(walk_c3d)
    static_xyz = get_points(static_c3d)
    walk_xyz = get_points(walk_c3d)
    return static_labels, walk_labels, static_xyz, walk_xyz


def load_templates_dict(static_c3d_path: str):
    out_dir = subject_output_dir(static_c3d_path)
    static_base = os.path.splitext(os.path.basename(static_c3d_path))[0]
    PELVIS_TPL = os.path.join(
        out_dir, f"{static_base}_pelvis_tcs_template.npz")
    THIGH_TPL = os.path.join(out_dir, f"{static_base}_femur_tcs_template.npz")
    SHANK_TPL = os.path.join(out_dir, f"{static_base}_tibia_tcs_template.npz")
    FOOT_TPL = os.path.join(out_dir, f"{static_base}_foot_tcs_template.npz")
    # Attempt to locate right-side templates near the left-side ones
    r_thigh_tpl = _try_load_first([
        THIGH_TPL.replace("femur_tcs_template", "femur_R_tcs_template"),
        THIGH_TPL.replace("femur_tcs_template", "femur_right_tcs_template"),
        THIGH_TPL.replace("S_Cal02_femur", "S_Cal02_R_femur"),
    ])
    r_shank_tpl = _try_load_first([
        SHANK_TPL.replace("tibia_tcs_template", "tibia_R_tcs_template"),
        SHANK_TPL.replace("tibia_tcs_template", "tibia_right_tcs_template"),
        SHANK_TPL.replace("S_Cal02_tibia", "S_Cal02_R_tibia"),
    ])
    r_foot_tpl = _try_load_first([
        FOOT_TPL.replace("foot_tcs_template", "foot_R_tcs_template"),
        FOOT_TPL.replace("foot_tcs_template", "foot_right_tcs_template"),
        FOOT_TPL.replace("S_Cal02_foot", "S_Cal02_R_foot"),
    ])
    return {
        "pelvis": load_template(PELVIS_TPL),
        "l_thigh": load_template(THIGH_TPL),
        "l_shank": load_template(SHANK_TPL),
        "l_foot": load_template(FOOT_TPL),
        "r_thigh": r_thigh_tpl,
        "r_shank": r_shank_tpl,
        "r_foot": r_foot_tpl,
    }


def validate_inputs(static_labels, walk_labels):
    for m in PELVIS_MARKERS:
        assert m in static_labels, f"Static missing pelvis marker: {m}"
        assert m in walk_labels, f"Walking missing pelvis marker: {m}"


def init_storage():
    global_T = {seg: [] for seg in SEGMENTS.keys()}
    relative_T = {seg: [] for seg in SEGMENTS.keys()}
    fit_err_mm = {seg: [] for seg in SEGMENTS.keys()}
    knee_qc_mm_left = {"from_thigh": [], "from_shank": []}
    knee_qc_mm_right = {"from_thigh": [], "from_shank": []}
    acs_R_lists = {seg: [] for seg in SEGMENTS.keys()}
    acs_O_lists = {seg: [] for seg in SEGMENTS.keys()}
    angles = {"hipL": [], "kneeL": [], "ankleL": [],
              "hipR": [], "kneeR": [], "ankleR": []}
    return global_T, relative_T, fit_err_mm, knee_qc_mm_left, knee_qc_mm_right, acs_R_lists, acs_O_lists, angles


def _append_frame_angles(
    angles: dict,
    R_A_world: dict,
    *,
    side: str,
    thigh_key: str,
    shank_key: str,
    foot_key: str,
) -> None:
    """One frame: hip/knee/ankle triplets from joint_angles or NaN placeholders."""
    suffix = "L" if side == "left" else "R"
    nan3 = [np.nan, np.nan, np.nan]
    keys = (f"hip{suffix}", f"knee{suffix}", f"ankle{suffix}")
    need = ("pelvis", thigh_key, shank_key, foot_key)
    if HAVE_JA and all(R_A_world.get(k) is not None for k in need):
        ja = _compute_angles_pf(
            pelvis_R=R_A_world["pelvis"],
            femur_R=R_A_world[thigh_key],
            tibia_R=R_A_world[shank_key],
            ankle_R=R_A_world[foot_key],
            side=side,
        )
        angles[keys[0]].append(list(ja["hip"]))
        angles[keys[1]].append(list(ja["knee"]))
        angles[keys[2]].append(list(ja["ankle"]))
    else:
        angles[keys[0]].append(nan3)
        angles[keys[1]].append(nan3)
        angles[keys[2]].append(nan3)


def _knee_tcs_from_static(
    static_labels: list,
    static_xyz: np.ndarray,
    tpl: dict,
    knee_marker: str,
    have_knee: bool,
    thigh_seg: str,
    shank_seg: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Express knee marker in thigh/shank TCS on the static frame (for QC)."""
    if not have_knee:
        return None, None
    T_thigh = fit_segment_on_static(static_labels, static_xyz, tpl, thigh_seg)
    T_shank = fit_segment_on_static(static_labels, static_xyz, tpl, shank_seg)
    if T_thigh is None or T_shank is None:
        return None, None
    k_world = static_xyz[STATIC_FRAME, static_labels.index(knee_marker)]
    return world_to_TCS(T_thigh, k_world), world_to_TCS(T_shank, k_world)


def fit_frame_with_templates(tpl, walk_labels, walk_frame_pts):
    frame_globals = {}
    frame_errors = {}
    markers_used = {}
    for seg, cfg in SEGMENTS.items():
        tpl_seg = tpl.get(seg)
        if tpl_seg is None:
            frame_globals[seg] = None
            frame_errors[seg] = np.nan
            markers_used[seg] = 0
            continue
        C_sub, Q_sub = subset_CQ_from_labels(
            tpl_seg["labels"], tpl_seg["C_local"], walk_labels, walk_frame_pts)
        if C_sub is None or Q_sub is None:
            frame_globals[seg] = None
            frame_errors[seg] = np.nan
            markers_used[seg] = 0
            continue
        R_Tw, t_Tw, err_mm = _kabsch_R_t_mean_error_mm(C_sub, Q_sub)
        frame_globals[seg] = to_h(R_Tw, t_Tw, 1.0)
        frame_errors[seg] = err_mm
        markers_used[seg] = int(C_sub.shape[0])
    return frame_globals, frame_errors, markers_used


def compute_acs_world(tpl, frame_globals):
    R_T_world_all = {}
    t_T_world_all = {}
    for seg in SEGMENTS.keys():
        T_Tw = frame_globals.get(seg)
        if T_Tw is None:
            R_T_world_all[seg] = None
            t_T_world_all[seg] = None
        else:
            R_T_world_all[seg] = T_Tw[:3, :3]
            t_T_world_all[seg] = T_Tw[:3, 3]
    R_A_world = {}
    O_A_world = {}
    for seg in SEGMENTS.keys():
        if R_T_world_all[seg] is None:
            R_A_world[seg] = None
            O_A_world[seg] = None
            continue
        R_T_world = R_T_world_all[seg]
        t_T_world = t_T_world_all[seg]
        R_TA = tpl[seg]["R_TA"]
        t_TA = tpl[seg]["t_TA"]
        R_A_world[seg] = R_T_world @ R_TA
        O_A_world[seg] = t_T_world + R_T_world @ t_TA
    return R_A_world, O_A_world


def compute_relatives(frame_globals):
    frame_relatives = {}
    for seg, cfg in SEGMENTS.items():
        parent = cfg["parent"]
        Tg = frame_globals.get(seg)
        if Tg is None:
            frame_relatives[seg] = None
            continue
        if parent is None:
            frame_relatives[seg] = Tg
        else:
            Tp = frame_globals.get(parent)
            frame_relatives[seg] = None if Tp is None else (invert_h(Tp) @ Tg)
    return frame_relatives


def knee_qc_frame(have_knee, knee_marker, walk_labels, walk_frame_pts,
                  frame_globals, frame_errors,
                  thigh_seg_name, shank_seg_name,
                  k_thigh_TCS, k_shank_TCS, knee_qc_mm):
    if not have_knee:
        return
    if (knee_marker not in walk_labels):
        knee_qc_mm["from_thigh"].append(np.nan)
        knee_qc_mm["from_shank"].append(np.nan)
        return
    k_meas = walk_frame_pts[walk_labels.index(knee_marker)]
    # From thigh
    T_thigh = frame_globals.get(thigh_seg_name)
    if (T_thigh is None) or np.isnan(frame_errors.get(thigh_seg_name, np.nan)) or (k_thigh_TCS is None):
        knee_qc_mm["from_thigh"].append(np.nan)
    else:
        k_pred_thigh = T_apply_point(T_thigh, k_thigh_TCS)
        knee_qc_mm["from_thigh"].append(
            float(np.linalg.norm(k_pred_thigh - k_meas)))
    # From shank
    T_shank = frame_globals.get(shank_seg_name)
    if (T_shank is None) or np.isnan(frame_errors.get(shank_seg_name, np.nan)) or (k_shank_TCS is None):
        knee_qc_mm["from_shank"].append(np.nan)
    else:
        k_pred_shank = T_apply_point(T_shank, k_shank_TCS)
        knee_qc_mm["from_shank"].append(
            float(np.linalg.norm(k_pred_shank - k_meas)))


def _stack_per_frame(rows: list, num_frames: int, item_shape: tuple[int, ...]) -> np.ndarray:
    """Stack frame-wise values into (num_frames, *item_shape); missing entries are NaN."""
    arr = np.full((num_frames,) + item_shape, np.nan, dtype=float)
    for i, item in enumerate(rows):
        if item is not None:
            arr[i] = item
    return arr


def stack_all(num_frames, global_T, relative_T, fit_err_mm, acs_R_lists, acs_O_lists):
    global_T_arr = {seg: _stack_per_frame(
        global_T[seg], num_frames, (4, 4)) for seg in SEGMENTS.keys()}
    relative_T_arr = {seg: _stack_per_frame(
        relative_T[seg], num_frames, (4, 4)) for seg in SEGMENTS.keys()}
    fit_err_arr = {seg: np.array(
        fit_err_mm[seg], dtype=float) for seg in SEGMENTS.keys()}
    acs_R_arr = {seg: _stack_per_frame(
        acs_R_lists[seg], num_frames, (3, 3)) for seg in SEGMENTS.keys()}
    acs_O_arr = {seg: _stack_per_frame(
        acs_O_lists[seg], num_frames, (3,)) for seg in SEGMENTS.keys()}
    return global_T_arr, relative_T_arr, fit_err_arr, acs_R_arr, acs_O_arr


def save_results(WALK_C3D, output_dir, num_frames, global_T_arr, relative_T_arr, fit_err_arr, acs_R_arr, acs_O_arr, knee_qc_L, knee_qc_R, angles):
    out_base = os.path.splitext(os.path.basename(WALK_C3D))[0]
    out_npz = os.path.join(
        output_dir, f"{out_base}_bilateral_chain_results.npz")
    save_npz(
        out_npz,
        pelvis_global=global_T_arr["pelvis"],
        l_thigh_global=global_T_arr["l_thigh"],
        l_shank_global=global_T_arr["l_shank"],
        l_foot_global=global_T_arr["l_foot"],
        r_thigh_global=global_T_arr.get("r_thigh"),
        r_shank_global=global_T_arr.get("r_shank"),
        r_foot_global=global_T_arr.get("r_foot"),
        pelvis_in_world=relative_T_arr["pelvis"],
        l_thigh_in_pelvis=relative_T_arr["l_thigh"],
        l_shank_in_thigh=relative_T_arr["l_shank"],
        l_foot_in_shank=relative_T_arr["l_foot"],
        r_thigh_in_pelvis=relative_T_arr.get("r_thigh"),
        r_shank_in_thigh=relative_T_arr.get("r_shank"),
        r_foot_in_shank=relative_T_arr.get("r_foot"),
        pelvis_acs_R=acs_R_arr["pelvis"],
        pelvis_acs_O=acs_O_arr["pelvis"],
        l_thigh_acs_R=acs_R_arr["l_thigh"],
        l_thigh_acs_O=acs_O_arr["l_thigh"],
        l_shank_acs_R=acs_R_arr["l_shank"],
        l_shank_acs_O=acs_O_arr["l_shank"],
        l_foot_acs_R=acs_R_arr["l_foot"],
        l_foot_acs_O=acs_O_arr["l_foot"],
        r_thigh_acs_R=acs_R_arr.get("r_thigh"),
        r_thigh_acs_O=acs_O_arr.get("r_thigh"),
        r_shank_acs_R=acs_R_arr.get("r_shank"),
        r_shank_acs_O=acs_O_arr.get("r_shank"),
        r_foot_acs_R=acs_R_arr.get("r_foot"),
        r_foot_acs_O=acs_O_arr.get("r_foot"),
        pelvis_err_mm=fit_err_arr["pelvis"],
        l_thigh_err_mm=fit_err_arr["l_thigh"],
        l_shank_err_mm=fit_err_arr["l_shank"],
        l_foot_err_mm=fit_err_arr["l_foot"],
        r_thigh_err_mm=fit_err_arr.get("r_thigh"),
        r_shank_err_mm=fit_err_arr.get("r_shank"),
        r_foot_err_mm=fit_err_arr.get("r_foot"),
        knee_qc_L_from_thigh=np.array(knee_qc_L["from_thigh"], dtype=float),
        knee_qc_L_from_shank=np.array(knee_qc_L["from_shank"], dtype=float),
        knee_qc_R_from_thigh=np.array(knee_qc_R["from_thigh"], dtype=float),
        knee_qc_R_from_shank=np.array(knee_qc_R["from_shank"], dtype=float),
        static_frame=np.array([STATIC_FRAME]),
        use_scaling=np.array([int(USE_SCALING)]),
        hipL=np.array(angles["hipL"], dtype=float),
        kneeL=np.array(angles["kneeL"], dtype=float),
        ankleL=np.array(angles["ankleL"], dtype=float),
        hipR=np.array(angles["hipR"], dtype=float),
        kneeR=np.array(angles["kneeR"], dtype=float),
        ankleR=np.array(angles["ankleR"], dtype=float),
    )


def run_pipeline():
    static_labels, walk_labels, static_xyz, walk_xyz = load_c3ds(
        STATIC_C3D, WALK_C3D)
    output_dir = subject_output_dir(STATIC_C3D)
    out_base = os.path.splitext(os.path.basename(WALK_C3D))[0]
    tpl = load_templates_dict(STATIC_C3D)
    validate_inputs(static_labels, walk_labels)
    num_frames = walk_xyz.shape[0]
    global_T, relative_T, fit_err_mm, knee_qc_L, knee_qc_R, acs_R_lists, acs_O_lists, angles = init_storage()
    # QC storage
    qc = {
        "markers_used": {seg: [] for seg in SEGMENTS},
        "fit_mean_mm": {seg: [] for seg in SEGMENTS},
        "rot_ortho":   {seg: [] for seg in SEGMENTS},
        "rot_det":     {seg: [] for seg in SEGMENTS},
    }
    # Precompute knee + static TCS transforms (Left and Right)
    have_knee_L = (KNEE_MARKER_L in static_labels) and (
        KNEE_MARKER_L in walk_labels)
    have_knee_R = (KNEE_MARKER_R in static_labels) and (
        KNEE_MARKER_R in walk_labels)
    kL_thigh_TCS, kL_shank_TCS = _knee_tcs_from_static(
        static_labels, static_xyz, tpl, KNEE_MARKER_L, have_knee_L, "l_thigh", "l_shank"
    )
    kR_thigh_TCS, kR_shank_TCS = _knee_tcs_from_static(
        static_labels, static_xyz, tpl, KNEE_MARKER_R, have_knee_R, "r_thigh", "r_shank"
    )
    for f in range(num_frames):
        walk_frame_pts = walk_xyz[f]
        frame_globals, frame_errors, markers_used = fit_frame_with_templates(
            tpl, walk_labels, walk_frame_pts)
        R_A_world, O_A_world = compute_acs_world(tpl, frame_globals)
        # Record markers used and fit errors for each segment
        for seg in SEGMENTS:
            # We can infer markers used from frame_errors presence + subset sizes.
            # Now we have exact counts returned by fitter
            qc["markers_used"][seg].append(markers_used.get(seg, 0))
            qc["fit_mean_mm"][seg].append(frame_errors[seg])

            Ri = R_A_world.get(seg)
            if Ri is None or np.isnan(frame_errors[seg]):
                qc["rot_ortho"][seg].append(np.nan)
                qc["rot_det"][seg].append(np.nan)
            else:
                qc["rot_ortho"][seg].append(rot_ortho_err(Ri))
                qc["rot_det"][seg].append(rot_det(Ri))
        _append_frame_angles(
            angles, R_A_world, side="left", thigh_key="l_thigh", shank_key="l_shank", foot_key="l_foot"
        )
        _append_frame_angles(
            angles, R_A_world, side="right", thigh_key="r_thigh", shank_key="r_shank", foot_key="r_foot"
        )
        frame_relatives = compute_relatives(frame_globals)
        knee_qc_frame(have_knee_L, KNEE_MARKER_L, walk_labels, walk_frame_pts,
                      frame_globals, frame_errors,
                      "l_thigh", "l_shank",
                      kL_thigh_TCS, kL_shank_TCS, knee_qc_L)
        knee_qc_frame(have_knee_R, KNEE_MARKER_R, walk_labels, walk_frame_pts,
                      frame_globals, frame_errors,
                      "r_thigh", "r_shank",
                      kR_thigh_TCS, kR_shank_TCS, knee_qc_R)
        for seg in SEGMENTS.keys():
            global_T[seg].append(frame_globals[seg])
            relative_T[seg].append(frame_relatives[seg])
            fit_err_mm[seg].append(frame_errors[seg])
            acs_R_lists[seg].append(R_A_world.get(seg))
            acs_O_lists[seg].append(O_A_world.get(seg))
    global_T_arr, relative_T_arr, fit_err_arr, acs_R_arr, acs_O_arr = stack_all(
        num_frames, global_T, relative_T, fit_err_mm, acs_R_lists, acs_O_lists)
    print(f"Frames: {num_frames}")
    for seg in SEGMENTS.keys():
        mean_err = np.nanmean(fit_err_arr[seg])
        valid = np.sum(~np.isnan(fit_err_arr[seg]))
        print(f"{seg}: mean fit error = {mean_err:.2f} mm over {
              valid}/{num_frames} frames")
    if (len(knee_qc_L["from_thigh"]) > 0) or (len(knee_qc_L["from_shank"]) > 0):
        print(f"Knee QC Left (not used in fits):")
        print(f"  from_thigh mean err: {
              np.nanmean(knee_qc_L['from_thigh']):.2f} mm")
        print(f"  from_shank mean err: {
              np.nanmean(knee_qc_L['from_shank']):.2f} mm")
    if (len(knee_qc_R["from_thigh"]) > 0) or (len(knee_qc_R["from_shank"]) > 0):
        print(f"Knee QC Right (not used in fits):")
        print(f"  from_thigh mean err: {
              np.nanmean(knee_qc_R['from_thigh']):.2f} mm")
        print(f"  from_shank mean err: {
              np.nanmean(knee_qc_R['from_shank']):.2f} mm")
    # Orientation jump per segment (deg) across frames
    qc["rot_jump_deg"] = {seg: [] for seg in SEGMENTS}
    for seg in SEGMENTS:
        Rseq = [r for r in acs_R_lists[seg]]
        jumps = []
        for i in range(len(Rseq) - 1):
            Ra, Rb = Rseq[i], Rseq[i+1]
            if Ra is None or Rb is None:
                jumps.append(np.nan)
            else:
                jumps.append(relative_rot_angle_deg(Ra, Rb))
        qc["rot_jump_deg"][seg] = jumps
    # Save compact QC NPZ
    out_base = os.path.splitext(os.path.basename(WALK_C3D))[0]
    save_npz(
        os.path.join(output_dir, f"{out_base}_QC.npz"),
        markers_used_pelvis=np.array(qc["markers_used"]["pelvis"], float),
        markers_used_thigh=np.array(qc["markers_used"]["l_thigh"], float),
        markers_used_shank=np.array(qc["markers_used"]["l_shank"], float),
        markers_used_foot=np.array(qc["markers_used"]["l_foot"], float),
        markers_used_r_thigh=np.array(qc["markers_used"].get(
            "r_thigh", []), float) if "r_thigh" in qc["markers_used"] else np.array([], float),
        markers_used_r_shank=np.array(qc["markers_used"].get(
            "r_shank", []), float) if "r_shank" in qc["markers_used"] else np.array([], float),
        markers_used_r_foot=np.array(qc["markers_used"].get(
            "r_foot", []), float) if "r_foot" in qc["markers_used"] else np.array([], float),

        fit_mm_pelvis=np.array(qc["fit_mean_mm"]["pelvis"], float),
        fit_mm_thigh=np.array(qc["fit_mean_mm"]["l_thigh"], float),
        fit_mm_shank=np.array(qc["fit_mean_mm"]["l_shank"], float),
        fit_mm_foot=np.array(qc["fit_mean_mm"]["l_foot"], float),
        fit_mm_r_thigh=np.array(qc["fit_mean_mm"].get(
            "r_thigh", []), float) if "r_thigh" in qc["fit_mean_mm"] else np.array([], float),
        fit_mm_r_shank=np.array(qc["fit_mean_mm"].get(
            "r_shank", []), float) if "r_shank" in qc["fit_mean_mm"] else np.array([], float),
        fit_mm_r_foot=np.array(qc["fit_mean_mm"].get(
            "r_foot", []), float) if "r_foot" in qc["fit_mean_mm"] else np.array([], float),

        rot_ortho_pelvis=np.array(qc["rot_ortho"]["pelvis"], float),
        rot_ortho_thigh=np.array(qc["rot_ortho"]["l_thigh"], float),
        rot_ortho_shank=np.array(qc["rot_ortho"]["l_shank"], float),
        rot_ortho_foot=np.array(qc["rot_ortho"]["l_foot"], float),
        rot_ortho_r_thigh=np.array(qc["rot_ortho"].get(
            "r_thigh", []), float) if "r_thigh" in qc["rot_ortho"] else np.array([], float),
        rot_ortho_r_shank=np.array(qc["rot_ortho"].get(
            "r_shank", []), float) if "r_shank" in qc["rot_ortho"] else np.array([], float),
        rot_ortho_r_foot=np.array(qc["rot_ortho"].get(
            "r_foot", []), float) if "r_foot" in qc["rot_ortho"] else np.array([], float),

        rot_det_pelvis=np.array(qc["rot_det"]["pelvis"], float),
        rot_det_thigh=np.array(qc["rot_det"]["l_thigh"], float),
        rot_det_shank=np.array(qc["rot_det"]["l_shank"], float),
        rot_det_foot=np.array(qc["rot_det"]["l_foot"], float),
        rot_det_r_thigh=np.array(qc["rot_det"].get(
            "r_thigh", []), float) if "r_thigh" in qc["rot_det"] else np.array([], float),
        rot_det_r_shank=np.array(qc["rot_det"].get(
            "r_shank", []), float) if "r_shank" in qc["rot_det"] else np.array([], float),
        rot_det_r_foot=np.array(qc["rot_det"].get(
            "r_foot", []), float) if "r_foot" in qc["rot_det"] else np.array([], float),

        rot_jump_pelvis=np.array(qc["rot_jump_deg"]["pelvis"], float),
        rot_jump_thigh=np.array(qc["rot_jump_deg"]["l_thigh"], float),
        rot_jump_shank=np.array(qc["rot_jump_deg"]["l_shank"], float),
        rot_jump_foot=np.array(qc["rot_jump_deg"]["l_foot"], float),
        rot_jump_r_thigh=np.array(qc["rot_jump_deg"].get(
            "r_thigh", []), float) if "r_thigh" in qc["rot_jump_deg"] else np.array([], float),
        rot_jump_r_shank=np.array(qc["rot_jump_deg"].get(
            "r_shank", []), float) if "r_shank" in qc["rot_jump_deg"] else np.array([], float),
        rot_jump_r_foot=np.array(qc["rot_jump_deg"].get(
            "r_foot", []), float) if "r_foot" in qc["rot_jump_deg"] else np.array([], float),

        kneeL_from_thigh=np.array(knee_qc_L["from_thigh"], float),
        kneeL_from_shank=np.array(knee_qc_L["from_shank"], float),
        kneeR_from_thigh=np.array(knee_qc_R["from_thigh"], float),
        kneeR_from_shank=np.array(knee_qc_R["from_shank"], float),
    )
    # Pelvis-axis sanity metrics
    ycos, zcos = pelvis_axis_sanity(acs_R_lists, acs_O_lists)
    print(f"pelvis y·forward median cos = {
          np.nanmedian(ycos):.3f} (→ ~+1 good)")
    print(f"pelvis z·up      median cos = {
          np.nanmedian(zcos):.3f} (→ ~+1 good)")
    _plot_timeseries(ycos, f"{out_base} pelvis y·forward cos", "cos", os.path.join(
        output_dir, f"{out_base}_pelvis_y_forward_cos.png"))
    _plot_timeseries(zcos, f"{out_base} pelvis z·up cos", "cos", os.path.join(
        output_dir, f"{out_base}_pelvis_z_up_cos.png"))
    # Quick PNGs
    _plot_timeseries(qc["fit_mean_mm"]["pelvis"], f"{
                     out_base} pelvis fit (mm)", "mean error (mm)", os.path.join(output_dir, f"{out_base}_pelvis_fit.png"))
    _plot_timeseries(qc["fit_mean_mm"]["l_thigh"], f"{
                     out_base} thigh fit (mm)", "mean error (mm)", os.path.join(output_dir, f"{out_base}_thigh_fit.png"))
    _plot_timeseries(qc["fit_mean_mm"]["l_shank"], f"{
                     out_base} shank fit (mm)", "mean error (mm)", os.path.join(output_dir, f"{out_base}_shank_fit.png"))
    _plot_timeseries(qc["fit_mean_mm"]["l_foot"],  f"{
                     out_base} foot fit (mm)",  "mean error (mm)", os.path.join(output_dir, f"{out_base}_foot_fit.png"))
    if "r_thigh" in qc["fit_mean_mm"]:
        _plot_timeseries(qc["fit_mean_mm"]["r_thigh"], f"{
                         out_base} right thigh fit (mm)", "mean error (mm)", os.path.join(output_dir, f"{out_base}_r_thigh_fit.png"))
    if "r_shank" in qc["fit_mean_mm"]:
        _plot_timeseries(qc["fit_mean_mm"]["r_shank"], f"{
                         out_base} right shank fit (mm)", "mean error (mm)", os.path.join(output_dir, f"{out_base}_r_shank_fit.png"))
    if "r_foot" in qc["fit_mean_mm"]:
        _plot_timeseries(qc["fit_mean_mm"]["r_foot"],  f"{
                         out_base} right foot fit (mm)",  "mean error (mm)", os.path.join(output_dir, f"{out_base}_r_foot_fit.png"))

    _plot_timeseries(qc["rot_ortho"]["pelvis"], f"{
                     out_base} pelvis R ortho", "||R^T R - I||_F", os.path.join(output_dir, f"{out_base}_pelvis_Rortho.png"))
    _plot_timeseries(qc["rot_jump_deg"]["l_thigh"], f"{
                     out_base} thigh ΔR (deg)", "deg/frame", os.path.join(output_dir, f"{out_base}_thigh_rotjump.png"))
    if "r_thigh" in qc["rot_jump_deg"]:
        _plot_timeseries(qc["rot_jump_deg"]["r_thigh"], f"{
                         out_base} right thigh ΔR (deg)", "deg/frame", os.path.join(output_dir, f"{out_base}_r_thigh_rotjump.png"))

    _plot_timeseries(knee_qc_L["from_thigh"], f"{out_base} knee QC L (thigh)", "mm", os.path.join(
        output_dir, f"{out_base}_kneeQC_L_thigh.png"))
    _plot_timeseries(knee_qc_L["from_shank"], f"{out_base} knee QC L (shank)", "mm", os.path.join(
        output_dir, f"{out_base}_kneeQC_L_shank.png"))
    _plot_timeseries(knee_qc_R["from_thigh"], f"{out_base} knee QC R (thigh)", "mm", os.path.join(
        output_dir, f"{out_base}_kneeQC_R_thigh.png"))
    _plot_timeseries(knee_qc_R["from_shank"], f"{out_base} knee QC R (shank)", "mm", os.path.join(
        output_dir, f"{out_base}_kneeQC_R_shank.png"))

    save_results(WALK_C3D, output_dir, num_frames, global_T_arr, relative_T_arr,
                 fit_err_arr, acs_R_arr, acs_O_arr, knee_qc_L, knee_qc_R, angles)

# ----- MAIN -----


def main():
    global WALK_C3D
    if len(sys.argv) >= 2:
        WALK_C3D = sys.argv[1]
    run_pipeline()
    return


if __name__ == "__main__":
    main()
