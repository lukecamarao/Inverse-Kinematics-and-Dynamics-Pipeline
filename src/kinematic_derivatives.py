# -*- coding: utf-8 -*-
"""
Compute segment COM positions in global (lab) frame using segment data and ACS
from inertial_segments (static calibration). Compute and plot global COM trajectories
from dynamic trial bilateral chain results (e.g. from svd_kabsch).

When a walk trial C3D includes force-plate analogs, loads GRF/COP via
forceplate_preprocess.read_grf_com_free_moment (same time base as marker frames),
**low-pass filters GRF/COP** (default ``DEFAULT_GRF_CUTOFF_HZ``) before stance QC
and before saving to the COM kinematics NPZ.

**Marker side:** segment and whole-body COM from bilateral ACS are **filtered in the
time domain before** velocity/acceleration (and angular ω before α); raw ACS is not
double-filtered at the COM stage.

@author: lmcam
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

from inertial_segments import (
    SEGMENT_ACS_TEMPLATE_MAP,
    load_segment_acs_from_static,
    export_inertial_segments,
    load_joint_centers,
)
from inertial_segments import _foot_markers_both_from_c3d

try:
    from forceplate_preprocess import read_grf_com_free_moment as _read_grf_com_free_moment
    HAVE_FORCEPLATE = True
except Exception:
    HAVE_FORCEPLATE = False
    _read_grf_com_free_moment = None  # type: ignore

# Default GRF low-pass (Hz) before stance detection / save; 0 to disable
DEFAULT_GRF_CUTOFF_HZ = 20.0
# Marker-derived COM: Butterworth cutoffs (Hz) before finite differences — see compute_segment_kinematics
DEFAULT_CUTOFF_POS_HZ = 6.0
DEFAULT_CUTOFF_VEL_HZ = 6.0
DEFAULT_CUTOFF_ANG_HZ = 8.0

# Force-plate stance (Fz > threshold, N): used by COP stance filtering defaults and checkpoints.
# Defined here so ``_filter_cop_mm_stance_spans(..., stance_fz_threshold_n=...)`` resolves at import.
CHECKPOINT_STANCE_FZ_THRESHOLD_N = 50.0

# Segment names in bilateral_chain_results.npz (svd_kabsch) -> inertial_segments
BILATERAL_TO_INERTIAL_SEG = {
    "l_thigh": "L_thigh",
    "r_thigh": "R_thigh",
    "l_shank": "L_shank",
    "r_shank": "R_shank",
    "l_foot": "L_foot",
    "r_foot": "R_foot",
}


def butter_lowpass_filtfilt(
    x: np.ndarray,
    cutoff_hz: float,
    fs_hz: float,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase low-pass Butterworth filter applied along time axis (axis=0).
    """
    if cutoff_hz <= 0.0 or fs_hz <= 0.0:
        return x
    nyq = 0.5 * fs_hz
    wn = cutoff_hz / nyq
    b, a = butter(order, wn, btype="low", analog=False)
    return filtfilt(b, a, x, axis=0)


def _filter_with_nan_support(
    x: np.ndarray,
    cutoff_hz: float,
    fs_hz: float,
    order: int = 4,
    keep_nan_mask: np.ndarray | None = None,
) -> np.ndarray:
    """NaN-tolerant low-pass filter by interpolating finite values first."""
    arr = np.asarray(x, dtype=float).copy()
    if arr.ndim == 1:
        arr2 = arr.reshape(-1, 1)
        squeeze = True
    else:
        arr2 = arr
        squeeze = False
    n = int(arr2.shape[0])
    t = np.arange(n, dtype=float)
    for j in range(arr2.shape[1]):
        v = arr2[:, j]
        finite = np.isfinite(v)
        if np.count_nonzero(finite) < max(8, order + 2):
            continue
        filled = np.interp(t, t[finite], v[finite])
        arr2[:, j] = butter_lowpass_filtfilt(
            filled, cutoff_hz, fs_hz, order=order)
    if keep_nan_mask is not None:
        km = np.asarray(keep_nan_mask, dtype=bool).reshape(-1)
        if km.shape[0] == n:
            arr2[km, :] = np.nan
    return arr2[:, 0] if squeeze else arr2


def _contiguous_true_windows(mask: np.ndarray) -> list[tuple[int, int]]:
    """Inclusive index intervals where ``mask`` is True (contiguous runs)."""
    m = np.asarray(mask, dtype=bool).reshape(-1)
    n = int(m.shape[0])
    if n == 0 or not np.any(m):
        return []
    out: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if not m[i]:
            i += 1
            continue
        j = i
        while j < n and m[j]:
            j += 1
        out.append((i, j - 1))
        i = j
    return out


def _filter_cop_mm_stance_spans(
    cop_mm: np.ndarray,
    fz_N: np.ndarray,
    cutoff_hz: float,
    fs_hz: float,
    order: int = 4,
    stance_fz_threshold_n: float = CHECKPOINT_STANCE_FZ_THRESHOLD_N,
) -> np.ndarray:
    """Filter COP (mm) only in stance spans; keep non-contact rows as NaN."""
    cop = np.asarray(cop_mm, dtype=float).copy().reshape(-1, 3)
    if cutoff_hz <= 0.0 or fs_hz <= 0.0:
        return cop
    fz = np.asarray(fz_N, dtype=float).reshape(-1)
    n = min(cop.shape[0], fz.shape[0])
    cop = cop[:n]
    fz = fz[:n]
    out = np.full_like(cop, np.nan, dtype=float)
    stance = fz > float(stance_fz_threshold_n)
    windows = _contiguous_true_windows(stance)
    min_needed = max(8, order + 2)
    for s, e in windows:
        seg = cop[s: e + 1].copy()
        finite_rows = np.isfinite(seg).all(axis=1)
        if int(np.sum(finite_rows)) < min_needed:
            continue
        t = np.arange(seg.shape[0], dtype=float)
        for j in range(3):
            v = seg[:, j]
            finite = np.isfinite(v)
            if int(np.sum(finite)) < min_needed:
                continue
            filled = np.interp(t, t[finite], v[finite])
            seg[:, j] = butter_lowpass_filtfilt(
                filled, cutoff_hz, fs_hz, order=order)
        seg[~finite_rows, :] = np.nan
        out[s: e + 1] = seg
    return out


def _filtered_linear_com_kinematics(
    pos: np.ndarray,
    dt: float,
    cutoff_pos_hz: float,
    cutoff_vel_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter COM position → central difference velocity → filter velocity → acceleration.
    ``pos`` is (N, 3) in mm; returns filtered position, filtered velocity, acceleration (same units/s).
    """
    fs = 1.0 / dt if dt > 0.0 else 0.0
    pos = np.asarray(pos, dtype=float)
    pos_filt = butter_lowpass_filtfilt(
        pos, cutoff_pos_hz, fs) if fs > 0.0 else pos
    vel = finite_diff_central(pos_filt, dt)
    vel_filt = butter_lowpass_filtfilt(
        vel, cutoff_vel_hz, fs) if fs > 0.0 else vel
    acc = finite_diff_central(vel_filt, dt)
    return pos_filt, vel_filt, acc


def _trim_or_pad_1d_to_length(arr: np.ndarray, n_use: int) -> np.ndarray:
    """Length ``n_use``; pad with NaN if shorter, trim if longer."""
    a = np.asarray(arr, dtype=float).ravel()
    if a.size >= n_use:
        return a[:n_use].copy()
    pad = np.full(n_use, np.nan, dtype=float)
    pad[: a.size] = a
    return pad


def _matplotlib_save_or_show(fig, out_path: str | None, saved_label: str = "Saved") -> None:
    if out_path:
        plt.savefig(out_path, dpi=150)
        print("{}: {}".format(saved_label, os.path.abspath(out_path)))
        plt.close(fig)
    else:
        plt.show()


def finite_diff_central(x: np.ndarray, dt: float) -> np.ndarray:
    """
    Central finite-difference derivative with forward/backward at endpoints.
    x: (N, ...) array, derivative along axis 0.
    """
    dx = np.zeros_like(x)
    if x.shape[0] < 2 or dt <= 0.0:
        return dx
    dx[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
    dx[0] = (x[1] - x[0]) / dt
    dx[-1] = (x[-1] - x[-2]) / dt
    return dx


def angular_velocity_from_R(R_seq: np.ndarray, dt: float) -> np.ndarray:
    """
    Approximate angular velocity (rad/s) from sequence of rotation matrices.
    Uses skew(R_rel - R_rel.T) / (2*dt) where R_rel = R_i^T R_{i+1}.

    Parameters
    ----------
    R_seq : (N, 3, 3)
        Rotation matrices for one segment over time.
    dt : float
        Time step in seconds.
    """
    N = R_seq.shape[0]
    w = np.zeros((N, 3), dtype=float)
    if N < 2 or dt <= 0.0:
        return w
    for i in range(1, N):
        Ra = R_seq[i - 1]
        Rb = R_seq[i]
        if np.any(np.isnan(Ra)) or np.any(np.isnan(Rb)):
            w[i] = np.array([np.nan, np.nan, np.nan], dtype=float)
            continue
        R_rel = Ra.T @ Rb
        omega_mat = (R_rel - R_rel.T) / (2.0 * dt)
        w[i] = np.array(
            [omega_mat[2, 1], omega_mat[0, 2], omega_mat[1, 0]], dtype=float
        )
    w[0] = w[1]
    return w


def compute_segment_com_global(
    inertial_export: dict[str, dict],
    template_dir: str,
    base: str,
) -> dict[str, np.ndarray]:
    """
    Compute segment COM positions in global (lab) frame.

    Uses O_A_static (segment origin in lab, mm), R_A_static (segment axes in lab),
    and r_com_seg (COM in segment frame, m) from inertial export:
      com_global_mm = O_A + R_A @ (r_com_seg * 1000)

    Parameters
    ----------
    inertial_export : dict
        Per-segment dict from export_inertial_segments; each value has "r_com_seg" (3,) in m.
    template_dir : str
        Directory containing static calibration npz templates (e.g. base_femur_tcs_template.npz).
    base : str
        Base name for template files (e.g. "S_Cal02").

    Returns
    -------
    dict[str, np.ndarray]
        segment_id -> com_global_mm, shape (3,), units mm.
    """
    out = {}
    for seg_id, d in inertial_export.items():
        O_A, R_A = load_segment_acs_from_static(template_dir, base, seg_id)
        if O_A is None or R_A is None:
            continue
        r_com_seg = np.asarray(d["r_com_seg"], dtype=float).reshape(3)  # m
        r_com_mm = r_com_seg * 1000.0
        com_global_mm = O_A + R_A @ r_com_mm
        out[seg_id] = np.asarray(com_global_mm, dtype=float).reshape(3)
    return out


def compute_com_trajectories(
    bilateral_npz_path: str,
    inertial_export: dict[str, dict],
    frame_rate_hz: float | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, np.ndarray, float]:
    """
    Compute segment and whole-body COM trajectories in global (lab) frame from
    bilateral chain results (per-frame ACS) and static inertial export.

    Parameters
    ----------
    bilateral_npz_path : str
        Path to *_bilateral_chain_results.npz (from svd_kabsch pipeline).
    inertial_export : dict
        Per-segment dict from export_inertial_segments (mass_kg, r_com_seg in m).
    frame_rate_hz : float, optional
        Point rate (Hz) for time vector. If None, time = frame index.

    Returns
    -------
    segment_trajs : dict[str, np.ndarray]
        segment_id -> (n_frames, 3) COM in lab frame, mm.
    segment_R : dict[str, np.ndarray]
        segment_id -> (n_frames, 3, 3) rotation matrices (ACS in lab).
    com_wb_traj : np.ndarray
        (n_frames, 3) whole-body COM in lab frame, mm (mass-weighted sum of limb segments).
    time_vec : np.ndarray
        (n_frames,) time in seconds.
    rate_used : float
        Frame rate used (Hz).
    """
    data = np.load(bilateral_npz_path, allow_pickle=True)
    n_frames = None
    for key in ("l_thigh_acs_O", "l_thigh_acs_R"):
        if key in data:
            arr = data[key]
            n_frames = arr.shape[0] if arr.ndim >= 1 else len(arr)
            break
    if n_frames is None:
        raise ValueError(f"No segment ACS arrays found in {
                         bilateral_npz_path}")

    rate_used = float(frame_rate_hz) if frame_rate_hz is not None else 100.0
    time_vec = np.arange(n_frames, dtype=float) / rate_used

    segment_trajs: dict[str, np.ndarray] = {}
    segment_R: dict[str, np.ndarray] = {}
    total_mass = 0.0
    for kabsch_seg, inertial_seg in BILATERAL_TO_INERTIAL_SEG.items():
        if inertial_seg not in inertial_export:
            continue
        r_key = f"{kabsch_seg}_acs_R"
        o_key = f"{kabsch_seg}_acs_O"
        if r_key not in data or o_key not in data:
            continue
        R_arr = np.asarray(data[r_key], dtype=float)   # (n_frames, 3, 3)
        O_arr = np.asarray(data[o_key], dtype=float)   # (n_frames, 3)
        if R_arr.ndim == 2:
            R_arr = R_arr[np.newaxis, ...]
        if O_arr.ndim == 1:
            O_arr = O_arr.reshape(1, -1)
        r_com_mm = np.asarray(
            inertial_export[inertial_seg]["r_com_seg"], dtype=float).reshape(3) * 1000.0
        mass_kg = float(inertial_export[inertial_seg]["mass_kg"])
        # com[f] = O[f] + R[f] @ r_com_mm
        com_traj = O_arr + (R_arr @ r_com_mm)  # (n_frames, 3)
        segment_trajs[inertial_seg] = com_traj
        segment_R[inertial_seg] = R_arr
        total_mass += mass_kg

    # Whole-body COM (mass-weighted average of segment COMs)
    com_wb_traj = np.zeros((n_frames, 3), dtype=float)
    if total_mass > 0:
        for seg_id, traj in segment_trajs.items():
            m = float(inertial_export[seg_id]["mass_kg"])
            com_wb_traj += m * traj
        com_wb_traj /= total_mass
    return segment_trajs, segment_R, com_wb_traj, time_vec, rate_used


def plot_com_trajectories(
    com_wb_traj: np.ndarray,
    time_vec: np.ndarray,
    title: str = "Whole-body COM (raw)",
    out_path: str | None = None,
) -> None:
    """
    Plot raw COM trajectories: X, Y, Z (lab frame, mm) vs time (s).
    """
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6))
    labels = ["X (mm)", "Y (mm)", "Z (mm)"]
    for i, (ax, lab) in enumerate(zip(axes, labels)):
        ax.plot(time_vec, com_wb_traj[:, i], color="C0", linewidth=0.8)
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    plt.tight_layout()
    _matplotlib_save_or_show(
        fig, out_path, saved_label="Saved COM trajectory plot")


def plot_segment_kinematics_components(
    seg_id: str,
    seg_kin: dict[str, dict[str, np.ndarray]],
    time_vec: np.ndarray,
    out_path: str | None = None,
) -> None:
    """
    Plot position, velocity, and acceleration for one segment: X, Y, Z components
    (not magnitude) vs time for inspection of bad segments.
    """
    if seg_id not in seg_kin:
        print("  Segment {} not in seg_kin, skip plot.".format(seg_id))
        return
    kin = seg_kin[seg_id]
    pos = np.asarray(kin["pos"], dtype=float)   # (N, 3) mm
    vel = np.asarray(kin["vel"], dtype=float)   # (N, 3) mm/s
    acc = np.asarray(kin["acc"], dtype=float)   # (N, 3) mm/s^2
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    comp_labels = ["X", "Y", "Z"]
    colors = ["C0", "C1", "C2"]
    # Row 0: position (mm)
    for i in range(3):
        axes[0].plot(time_vec, pos[:, i], color=colors[i],
                     label=comp_labels[i], linewidth=0.9)
    axes[0].set_ylabel("Position (mm)")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)
    # Row 1: velocity (mm/s)
    for i in range(3):
        axes[1].plot(time_vec, vel[:, i], color=colors[i],
                     label=comp_labels[i], linewidth=0.9)
    axes[1].set_ylabel("Velocity (mm/s)")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.3)
    # Row 2: acceleration (mm/s^2)
    for i in range(3):
        axes[2].plot(time_vec, acc[:, i], color=colors[i],
                     label=comp_labels[i], linewidth=0.9)
    axes[2].set_ylabel("Acceleration (mm/s²)")
    axes[2].set_xlabel("Time (s)")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, alpha=0.3)
    fig.suptitle(
        "{} COM: position, velocity, acceleration (components)".format(seg_id))
    plt.tight_layout()
    _matplotlib_save_or_show(fig, out_path, saved_label="  Saved")


def compute_segment_kinematics(
    segment_trajs: dict[str, np.ndarray],
    segment_R: dict[str, np.ndarray],
    com_wb_traj: np.ndarray,
    dt: float,
    cutoff_pos_hz: float = DEFAULT_CUTOFF_POS_HZ,
    cutoff_vel_hz: float = DEFAULT_CUTOFF_VEL_HZ,
    cutoff_ang_hz: float = DEFAULT_CUTOFF_ANG_HZ,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    """
    Compute COM position, velocity, acceleration and angular velocity/acceleration
    for each segment, plus whole-body COM kinematics.

    Pipeline (linear kinematics):
      1. Low-pass filter COM position
      2. Compute velocity (finite difference of filtered position)
      3. Low-pass filter velocity
      4. Compute acceleration (finite difference of filtered velocity)

    Angular: angular velocity from R(t), then filter, then differentiate for alpha.

    Returns
    -------
    seg_kin : dict
        seg_id -> {
            "pos": (N,3) filtered COM position (mm),
            "vel": (N,3) COM velocity (mm/s),
            "acc": (N,3) COM acceleration (mm/s^2),
            "omega": (N,3) angular velocity (rad/s),
            "alpha": (N,3) angular acceleration (rad/s^2),
        }
    wb_kin : dict
        Whole-body COM kinematics:
        {
            "pos": (N,3),
            "vel": (N,3),
            "acc": (N,3),
        }
    """
    fs = 1.0 / dt if dt > 0.0 else 0.0
    seg_kin: dict[str, dict[str, np.ndarray]] = {}

    # Per-segment kinematics: filter pos -> velocity -> filter velocity -> acceleration
    for seg_id, pos in segment_trajs.items():
        pos_filt, vel_filt, acc = _filtered_linear_com_kinematics(
            pos, dt, cutoff_pos_hz, cutoff_vel_hz
        )

        R_seq = segment_R.get(seg_id)
        if R_seq is not None:
            R_seq = np.asarray(R_seq, dtype=float)
            omega = angular_velocity_from_R(R_seq, dt)
            # Filter angular velocity before computing angular acceleration
            omega_filt = butter_lowpass_filtfilt(
                omega, cutoff_ang_hz, fs) if fs > 0.0 else omega
            alpha = finite_diff_central(omega_filt, dt)
        else:
            N = pos.shape[0]
            omega_filt = np.full((N, 3), np.nan, dtype=float)
            alpha = np.full((N, 3), np.nan, dtype=float)

        seg_kin[seg_id] = {
            "pos": pos_filt,
            "vel": vel_filt,
            "acc": acc,
            "omega": omega_filt,
            "alpha": alpha,
        }

    com_wb_traj = np.asarray(com_wb_traj, dtype=float)
    wb_pos, wb_vel_filt, wb_acc = _filtered_linear_com_kinematics(
        com_wb_traj, dt, cutoff_pos_hz, cutoff_vel_hz
    )
    wb_kin = {"pos": wb_pos, "vel": wb_vel_filt, "acc": wb_acc}
    return seg_kin, wb_kin


# Checkpoint thresholds (tunable for your data)
CHECKPOINT_MAX_ACC_MM_S2 = 5000.0       # whole-body COM acc magnitude (mm/s^2)
# max frame-to-frame change in |acc| (mm/s^3 proxy)
CHECKPOINT_ACC_JUMP_MM_S3 = 15000.0
# angular velocity magnitude (rad/s) for walking
CHECKPOINT_MAX_OMEGA_RAD_S = 8.0
# "stance" = frames in bottom 40% of |wb_vel_z|
CHECKPOINT_STANCE_LOW_VEL_FRAC = 0.4
# WB |acc| in vel-heuristic stance (quiet mid-stance proxy)
CHECKPOINT_STANCE_MAX_ACC_MM_S2 = 3000.0
# segment |ω| in vel-heuristic stance only
CHECKPOINT_STANCE_MAX_OMEGA_RAD_S = 5.0
# When stance = Fz on plate, swing leg still moves; WB COM acc stays gait-like — use looser WB limit only.
CHECKPOINT_STANCE_FP_MAX_ACC_MM_S2 = 5500.0
# Distal segment COM accelerates much faster than WB COM (swing, foot strike); do not use WB limit.
# (CHECKPOINT_STANCE_FZ_THRESHOLD_N is defined with module defaults above.)


def stance_mask_from_fz(
    fz_N: np.ndarray,
    n_frames_kin: int,
    stance_fz_threshold_n: float = CHECKPOINT_STANCE_FZ_THRESHOLD_N,
) -> np.ndarray:
    """
    Stance boolean mask, **same rule** as ``checkpoint_kinematics`` when force-plate
    Fz is available: on the overlap, ``fz > stance_fz_threshold_n``. Length is
    ``n_frames_kin``; frames without overlapping Fz are False.
    """
    fz = np.asarray(fz_N, dtype=float).ravel()
    n = int(n_frames_kin)
    stance_mask = np.zeros(max(0, n), dtype=bool)
    if fz.size == 0 or n <= 0:
        return stance_mask
    n_st = min(n, fz.shape[0])
    stance_mask[:n_st] = fz[:n_st] > stance_fz_threshold_n
    return stance_mask


def stance_windows_from_fz(
    fz_N: np.ndarray,
    n_frames_kin: int,
    stance_fz_threshold_n: float = CHECKPOINT_STANCE_FZ_THRESHOLD_N,
) -> list[tuple[int, int]]:
    """
    Contiguous stance index intervals using the same rule as ``checkpoint_kinematics``
    (force-plate Fz: ``fz > threshold`` on overlap). For ankle / ID QC, pass the same
    ``fz_N`` and frame count as used for COM kinematics checkpoint.
    """
    mask = stance_mask_from_fz(fz_N, n_frames_kin, stance_fz_threshold_n)
    return _contiguous_true_windows(mask)


CHECKPOINT_SEGMENT_MAX_ACC_MM_S2: dict[str, float] = {
    "L_thigh": 16000.0,
    "R_thigh": 16000.0,
    "L_shank": 26000.0,
    "R_shank": 26000.0,
    "L_foot": 45000.0,
    "R_foot": 45000.0,
}


def load_forceplate_aligned(
    walk_c3d_path: str,
    n_frames_kin: int,
    grf_cutoff_hz: float = DEFAULT_GRF_CUTOFF_HZ,
    verbose: bool = False,
) -> dict[str, np.ndarray | float | str] | None:
    """
    Load GRF/COP from the same walk C3D using forceplate_preprocess.read_grf_com_free_moment.

    Trims to min(n_frames_kin, n_frames_fp). Optionally low-pass filters GRF and COP
    at grf_cutoff_hz using the returned sampling_rate_hz (typically analog rate).
    """
    if not HAVE_FORCEPLATE or _read_grf_com_free_moment is None:
        if verbose:
            print("Force plate: forceplate_preprocess not available (import failed).")
        return None
    if not walk_c3d_path or not os.path.isfile(walk_c3d_path):
        if verbose:
            print("Force plate: walk C3D missing or invalid: {}".format(walk_c3d_path))
        return None
    try:
        fp = _read_grf_com_free_moment(walk_c3d_path)
    except Exception as e:
        if verbose:
            print("Force plate: read_grf_com_free_moment failed: {}".format(e))
        return None
    grf = np.asarray(fp.get("grf"), dtype=float)
    if grf.size == 0 or grf.ndim != 2 or grf.shape[1] < 3:
        if verbose:
            print("Force plate: GRF array empty or wrong shape (need Nx3).")
        return None
    n_fp = int(grf.shape[0])
    n_use = min(int(n_frames_kin), n_fp)
    if n_use <= 0:
        return None
    grf = grf[:n_use].copy()
    cop_mm = (
        np.asarray(fp.get("cop_mm"), dtype=float)
        if fp.get("cop_mm") is not None
        else np.asarray(fp.get("cop"), dtype=float) * 1000.0
    )
    if cop_mm.shape[0] >= n_use:
        cop_mm = cop_mm[:n_use].copy()
    else:
        cop_mm = np.full((n_use, 3), np.nan, dtype=float)
    fm = _trim_or_pad_1d_to_length(np.asarray(
        fp.get("free_moment"), dtype=float), n_use)
    fm_cop = _trim_or_pad_1d_to_length(np.asarray(
        fp.get("free_moment_about_cop"), dtype=float), n_use)
    fs = float(fp.get("sampling_rate_hz") or 0.0)
    if grf_cutoff_hz > 0.0 and fs > 0.0:
        grf = butter_lowpass_filtfilt(grf, grf_cutoff_hz, fs)
        cop_mm = _filter_cop_mm_stance_spans(
            cop_mm, grf[:, 2], grf_cutoff_hz, fs)
        fm = butter_lowpass_filtfilt(
            fm.reshape(-1, 1), grf_cutoff_hz, fs).ravel()
        fm_cop = _filter_with_nan_support(
            fm_cop.reshape(-1, 1), grf_cutoff_hz, fs).ravel()
    cop = cop_mm * 1.0e-3
    units = fp.get("units") or {}
    return {
        "grf_N": grf,
        "cop_m": cop,
        "cop_mm": cop_mm,
        "free_moment_Nm": fm,  # Mz about plate origin, 1D (n_use,)
        "free_moment_about_cop_Nm": fm_cop,  # Mz about COP, 1D (n_use,)
        "sampling_rate_hz": fs,
        "n_frames_used": float(n_use),
        "n_frames_kin": float(n_frames_kin),
        "n_frames_fp": float(n_fp),
        "force_unit": str(units.get("force", "N")),
        "cop_unit": str(units.get("point", "m")),
        "transform_qc": fp.get("transform_qc"),
        "plate_axes": fp.get("plate_axes"),
    }


def checkpoint_kinematics(
    seg_kin: dict[str, dict[str, np.ndarray]],
    wb_kin: dict[str, np.ndarray],
    verbose: bool = True,
    fz_N: np.ndarray | None = None,
    stance_fz_threshold_n: float = CHECKPOINT_STANCE_FZ_THRESHOLD_N,
) -> dict:
    """
    Checkpoint: no wild spikes in accelerations, angular velocity magnitudes
    reasonable for walking, and smooth through stance (where numerical noise
    often shows up).

    Parameters
    ----------
    fz_N : array, optional
        Vertical GRF (N), length <= n_kin; from forceplate_preprocess. When
        provided, stance frames use Fz > stance_fz_threshold_n on the overlap.
    stance_fz_threshold_n : float
        Stance when fz_N > this (Newtons).

    Returns
    -------
    dict with all_ok (bool), details (per-check results), issues (list of str).
    """
    issues = []
    details = {}

    # 1) No wild spikes in accelerations
    wb_acc = np.asarray(wb_kin["acc"], dtype=float)
    acc_mag = np.linalg.norm(wb_acc, axis=1)
    acc_mag_valid = acc_mag[~np.isnan(acc_mag)]
    max_wb_acc = np.nanmax(acc_mag) if acc_mag_valid.size else 0.0
    if max_wb_acc > CHECKPOINT_MAX_ACC_MM_S2:
        issues.append(
            "Whole-body COM acceleration spike: max |acc| = {:.1f} mm/s^2 (limit {:.0f})".format(
                max_wb_acc, CHECKPOINT_MAX_ACC_MM_S2
            )
        )
    details["wb_acc_max_mm_s2"] = float(max_wb_acc)

    # Frame-to-frame jump in acceleration magnitude (spike detector)
    if acc_mag.shape[0] >= 2:
        d_acc = np.abs(np.diff(acc_mag))
        max_jump = np.nanmax(d_acc) if np.any(~np.isnan(d_acc)) else 0.0
        if max_jump > CHECKPOINT_ACC_JUMP_MM_S3:
            issues.append(
                "Large frame-to-frame jump in |acc|: max Δ = {:.1f} (limit {:.0f})".format(
                    max_jump, CHECKPOINT_ACC_JUMP_MM_S3
                )
            )
        details["wb_acc_max_jump"] = float(max_jump)

    for seg_id, kin in seg_kin.items():
        acc = np.asarray(kin["acc"], dtype=float)
        mag = np.linalg.norm(acc, axis=1)
        m = np.nanmax(mag) if np.any(~np.isnan(mag)) else 0.0
        seg_lim = CHECKPOINT_SEGMENT_MAX_ACC_MM_S2.get(
            seg_id, CHECKPOINT_MAX_ACC_MM_S2
        )
        if m > seg_lim:
            issues.append(
                "{} COM acceleration spike: max |acc| = {:.1f} mm/s^2 (limit {:.0f})".format(
                    seg_id, m, seg_lim
                )
            )
        details[f"{seg_id}_acc_max_mm_s2"] = float(m)
        details[f"{seg_id}_acc_limit_mm_s2"] = float(seg_lim)

        # Angular velocity magnitudes reasonable for walking
        omega = np.asarray(kin["omega"], dtype=float)
        if not np.all(np.isnan(omega)):
            omag = np.linalg.norm(omega, axis=1)
            p95 = np.nanpercentile(omag, 95)
            if p95 > CHECKPOINT_MAX_OMEGA_RAD_S:
                issues.append(
                    "{} angular velocity high: 95th percentile |ω| = {:.2f} rad/s (limit {:.1f})".format(
                        seg_id, p95, CHECKPOINT_MAX_OMEGA_RAD_S
                    )
                )
            details[f"{seg_id}_omega_p95_rad_s"] = float(p95)

    # 3) Smooth through stance (where numerical noise often shows up)
    wb_vel = np.asarray(wb_kin["vel"], dtype=float)
    wb_vel_z = wb_vel[:, 2]
    n = wb_vel_z.shape[0]
    if n >= 10:
        fz = None if fz_N is None else np.asarray(fz_N, dtype=float).ravel()
        if fz is not None and fz.shape[0] > 0:
            stance_mask = stance_mask_from_fz(fz, n, stance_fz_threshold_n)
            n_st = min(n, fz.shape[0])
            details["stance_method"] = "force_plate_Fz"
            details["stance_fz_threshold_N"] = float(stance_fz_threshold_n)
            if n_st < n:
                details["stance_note"] = (
                    "Fz length {} < kin {}; stance mask true only on overlap".format(
                        n_st, n)
                )
        else:
            # Heuristic: |vertical velocity| in bottom fraction (near mid-stance)
            abs_vel_z = np.abs(wb_vel_z)
            thresh = np.nanpercentile(
                abs_vel_z, 100.0 * CHECKPOINT_STANCE_LOW_VEL_FRAC)
            stance_mask = abs_vel_z <= thresh
            details["stance_method"] = "low_|vel_z|"
        n_stance = int(np.sum(stance_mask))
        details["stance_frames"] = n_stance
        details["stance_frac"] = float(n_stance / n)

        if n_stance > 0:
            acc_mag_stance = acc_mag[stance_mask]
            max_acc_stance = np.nanmax(acc_mag_stance) if np.any(
                ~np.isnan(acc_mag_stance)) else 0.0
            fp_stance = details.get("stance_method") == "force_plate_Fz"
            stance_acc_limit = (
                CHECKPOINT_STANCE_FP_MAX_ACC_MM_S2
                if fp_stance
                else CHECKPOINT_STANCE_MAX_ACC_MM_S2
            )
            if max_acc_stance > stance_acc_limit:
                issues.append(
                    "Stance: max |wb acc| = {:.1f} mm/s^2 (limit {:.0f})".format(
                        max_acc_stance, stance_acc_limit
                    )
                )
            details["stance_acc_max_mm_s2"] = float(max_acc_stance)
            details["stance_acc_limit_mm_s2"] = float(stance_acc_limit)

            # Fz>threshold = plate loaded during gait; swing limb is not quiet — skip segment ω here.
            if not fp_stance:
                for seg_id, kin in seg_kin.items():
                    omega = np.asarray(kin["omega"], dtype=float)
                    if np.all(np.isnan(omega)):
                        continue
                    omag = np.linalg.norm(omega, axis=1)
                    max_omega_stance = np.nanmax(omag[stance_mask])
                    if max_omega_stance > CHECKPOINT_STANCE_MAX_OMEGA_RAD_S:
                        issues.append(
                            "Stance: {} |ω| max = {:.2f} rad/s (limit {:.1f})".format(
                                seg_id, max_omega_stance, CHECKPOINT_STANCE_MAX_OMEGA_RAD_S
                            )
                        )
                    details[f"{seg_id}_stance_omega_max_rad_s"] = float(
                        max_omega_stance)
            else:
                details["stance_seg_omega_check"] = "skipped (FP stance: swing leg still moves)"

    all_ok = len(issues) == 0
    out = {"all_ok": all_ok, "details": details, "issues": issues}

    if verbose:
        print("\n" + "=" * 60)
        print("Checkpoint: kinematics quality (no spikes, reasonable ω, smooth stance)")
        print("=" * 60)
        print("  Whole-body: max |acc| = {:.1f} mm/s^2 (limit {:.0f})".format(
            details.get("wb_acc_max_mm_s2", 0), CHECKPOINT_MAX_ACC_MM_S2))
        if "wb_acc_max_jump" in details:
            print("  Whole-body: max frame Δ|acc| = {:.1f} (limit {:.0f})".format(
                details["wb_acc_max_jump"], CHECKPOINT_ACC_JUMP_MM_S3))
        for seg_id in sorted(seg_kin.keys()):
            k = "{}_omega_p95_rad_s".format(seg_id)
            if k in details:
                print("  {}: 95th % |ω| = {:.2f} rad/s (limit {:.1f})".format(
                    seg_id, details[k], CHECKPOINT_MAX_OMEGA_RAD_S))
        if "stance_frames" in details:
            sm = details.get("stance_method", "stance")
            print("  Stance ({}): {} frames ({:.0%})".format(
                sm, details["stance_frames"], details.get("stance_frac", 0)))
            if "stance_note" in details:
                print("    Note: {}".format(details["stance_note"]))
            if "stance_acc_max_mm_s2" in details:
                lim = details.get(
                    "stance_acc_limit_mm_s2", CHECKPOINT_STANCE_MAX_ACC_MM_S2
                )
                print("  Stance: max |wb acc| = {:.1f} mm/s^2 (limit {:.0f})".format(
                    details["stance_acc_max_mm_s2"], lim))
            if details.get("stance_seg_omega_check"):
                print("    {}".format(details["stance_seg_omega_check"]))
        if all_ok:
            print("  Overall: PASS")
        else:
            print("  Overall: FAIL")
            for s in issues:
                print("    - {}".format(s))
        print("=" * 60)

    return out


def _get_point_rate_hz(c3d_path: str) -> float | None:
    """Read point rate (Hz) from C3D if possible."""
    try:
        import ezc3d
        c = ezc3d.c3d(c3d_path)
        rate = c["parameters"]["POINT"]["RATE"]["value"]
        # ezc3d may return array; extract scalar to avoid NumPy deprecation
        rate = float(np.asarray(rate).ravel()[0])
        return rate
    except Exception:
        return None


def resolve_walk_c3d_from_bilateral_npz(
    bilateral_npz: str,
    c3d_dir: str,
) -> str | None:
    """``<stem>.c3d`` next to the bilateral NPZ or under ``c3d_dir`` (``stem`` = basename without suffix)."""
    bn = os.path.basename(bilateral_npz)
    suf = "_bilateral_chain_results.npz"
    if not bn.endswith(suf):
        return None
    wb = bn[: -len(suf)]
    for cand in (
        os.path.join(os.path.dirname(bilateral_npz), wb + ".c3d"),
        os.path.join(c3d_dir, wb + ".c3d"),
    ):
        if os.path.isfile(cand):
            return os.path.abspath(cand)
    return None


def discover_bilateral_npz_and_walk_c3d(
    template_dir: str,
    c3d_dir: str,
    script_dir: str,
) -> tuple[str | None, str | None]:
    """
    Find ``*_bilateral_chain_results.npz`` and the matching walk ``<WalkName>.c3d``.

    Search order: ``template_dir`` (e.g. subject 02 - S_Cal02), ``c3d_dir``, ``script_dir``.
    The walk trial name is taken from the bilateral filename (e.g. Walk_R04).
    The C3D is resolved next to the bilateral NPZ first, then ``c3d_dir/<WalkName>.c3d``.

    If multiple bilateral NPZs exist, the most recently modified file is used.
    """
    paths_found: list[str] = []
    for d in (template_dir, c3d_dir, script_dir):
        if not os.path.isdir(d):
            continue
        paths_found.extend(glob.glob(os.path.join(
            d, "*_bilateral_chain_results.npz")))
    seen: set[str] = set()
    uniq: list[str] = []
    for p in paths_found:
        ap = os.path.abspath(p)
        key = os.path.normcase(ap)
        if key not in seen and os.path.isfile(ap):
            seen.add(key)
            uniq.append(ap)
    if not uniq:
        return None, None
    bilateral_npz = sorted(uniq, key=os.path.getmtime, reverse=True)[0]
    return bilateral_npz, resolve_walk_c3d_from_bilateral_npz(bilateral_npz, c3d_dir)


def main(argv: list[str] | None = None) -> None:
    """Load segment data from inertial_segments, compute COM in global frame, then COM trajectories and plot."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    subject, base = "subject 02", "S_Cal02"
    folder = f"{subject} - {base}"
    template_dir = os.path.join(script_dir, folder)
    csv_path = os.path.join(template_dir, f"{base}_joint_centers.csv")
    c3d_path = os.path.abspath(os.path.join(
        script_dir, "..", "..", "c3d", subject, f"{base}.c3d"))
    c3d_dir = os.path.join(script_dir, "..", "..", "c3d", subject)

    p = argparse.ArgumentParser(
        description="COM kinematics + optional force plate (same walk C3D as bilateral trial).")
    p.add_argument(
        "--walk-c3d",
        type=str,
        default=None,
        help="Walk trial C3D path (force plate + point rate). Default: match bilateral NPZ basename.",
    )
    p.add_argument(
        "--bilateral-npz",
        type=str,
        default=None,
        help="Path to *_bilateral_chain_results.npz. Default: discover next to subject folder or c3d.",
    )
    args = p.parse_args(argv)

    bilateral_npz: str | None = None
    walk_c3d_path: str | None = None
    if args.bilateral_npz:
        bilateral_npz = os.path.abspath(args.bilateral_npz)
    if args.walk_c3d:
        walk_c3d_path = os.path.abspath(args.walk_c3d)

    if bilateral_npz is None or walk_c3d_path is None:
        disc_b, disc_w = discover_bilateral_npz_and_walk_c3d(
            template_dir, c3d_dir, script_dir)
        if bilateral_npz is None:
            bilateral_npz = disc_b
        if walk_c3d_path is None:
            walk_c3d_path = disc_w

    if bilateral_npz is not None and walk_c3d_path is None:
        walk_c3d_path = resolve_walk_c3d_from_bilateral_npz(
            bilateral_npz, c3d_dir)

    walk_trial = os.path.basename(
        walk_c3d_path) if walk_c3d_path else "unknown.c3d"
    if bilateral_npz:
        print(
            "Resolved bilateral NPZ: {}\nResolved walk C3D: {}".format(
                os.path.abspath(bilateral_npz),
                os.path.abspath(
                    walk_c3d_path) if walk_c3d_path else "(none — frame rate may default)",
            )
        )

    if not os.path.isfile(csv_path):
        print(f"Joint centers CSV not found: {csv_path}")
        print("Run static_calibration and inertial_segments first.")
        return

    jc = load_joint_centers(csv_path)
    body_mass_kg = 70.0
    foot_markers_both = _foot_markers_both_from_c3d(
        c3d_path, 0) if os.path.isfile(c3d_path) else None

    inertial_export = export_inertial_segments(
        body_mass_kg, jc, foot_markers_both,
        template_dir=template_dir,
        base=base,
        out_path=None,
    )
    if not inertial_export:
        print("No inertial export segments available (missing templates or joint centers).")
        return

    # Static COM positions (single frame)
    com_global = compute_segment_com_global(
        inertial_export, template_dir, base)
    print("Segment COM positions in global (lab) frame [mm]:")
    print("  (X, Y, Z) in lab coordinates")
    for seg_id in sorted(com_global.keys()):
        pos = com_global[seg_id]
        print(f"  {seg_id}: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

    # Global COM trajectories from dynamic trial (bilateral_chain_results.npz from discovery above).
    # Do not re-search only under walk_c3d dirname — bilateral may live in subject 02 - S_Cal02
    # while the walk C3D is under c3d/subject 02.
    if bilateral_npz:
        walk_base = os.path.basename(bilateral_npz).replace(
            "_bilateral_chain_results.npz", "")
        rate_hz = _get_point_rate_hz(walk_c3d_path) if walk_c3d_path else None
        segment_trajs, segment_R, com_wb_traj, time_vec, rate_used = compute_com_trajectories(
            bilateral_npz, inertial_export, frame_rate_hz=rate_hz
        )
        n_f = com_wb_traj.shape[0]
        print(f"\nGlobal COM trajectories: {
              n_f} frames @ {rate_used:.1f} Hz (from {walk_trial})")
        print("  Whole-body COM (mass-weighted limbs): min/mean/max X = {:.2f} / {:.2f} / {:.2f} mm".format(
            np.nanmin(com_wb_traj[:, 0]), np.nanmean(com_wb_traj[:, 0]), np.nanmax(com_wb_traj[:, 0])))
        print("  Y = {:.2f} / {:.2f} / {:.2f} mm".format(
            np.nanmin(com_wb_traj[:, 1]), np.nanmean(com_wb_traj[:, 1]), np.nanmax(com_wb_traj[:, 1])))
        print("  Z = {:.2f} / {:.2f} / {:.2f} mm".format(
            np.nanmin(com_wb_traj[:, 2]), np.nanmean(com_wb_traj[:, 2]), np.nanmax(com_wb_traj[:, 2])))

        # Full kinematics: filter marker-derived COM & ω, then finite-difference (no raw derivatives).
        dt = 1.0 / rate_used if rate_used > 0.0 else 0.0
        seg_kin, wb_kin = compute_segment_kinematics(
            segment_trajs, segment_R, com_wb_traj, dt,
            cutoff_pos_hz=DEFAULT_CUTOFF_POS_HZ,
            cutoff_vel_hz=DEFAULT_CUTOFF_VEL_HZ,
            cutoff_ang_hz=DEFAULT_CUTOFF_ANG_HZ,
        )
        print(
            "\nFiltering (markers → COM/ω): low-pass before derivatives — "
            "pos {:.1f} Hz, vel {:.1f} Hz, angular {:.1f} Hz @ {:.1f} Hz sample rate.".format(
                DEFAULT_CUTOFF_POS_HZ, DEFAULT_CUTOFF_VEL_HZ, DEFAULT_CUTOFF_ANG_HZ, rate_used,
            )
        )

        fp_aligned = load_forceplate_aligned(
            walk_c3d_path, n_f, grf_cutoff_hz=DEFAULT_GRF_CUTOFF_HZ, verbose=True
        )
        fz_for_ck = None
        if fp_aligned is not None:
            fz_for_ck = fp_aligned["grf_N"][:, 2]
            print(
                "\nForce plate (forceplate_preprocess): {} frames used (kin {}, fp raw {}) @ {:.1f} Hz; "
                "GRF/COP/free moment **filtered** @ {:.1f} Hz before save/QC.".format(
                    int(fp_aligned["n_frames_used"]),
                    int(fp_aligned["n_frames_kin"]),
                    int(fp_aligned["n_frames_fp"]),
                    fp_aligned["sampling_rate_hz"],
                    DEFAULT_GRF_CUTOFF_HZ,
                )
            )
        elif HAVE_FORCEPLATE and walk_c3d_path and os.path.isfile(walk_c3d_path):
            print(
                "\nForce plate: no usable GRF in C3D (or read failed); stance uses |vel_z| heuristic.")

        # Checkpoint: no wild spikes, reasonable angular velocity, smooth through stance
        checkpoint_kinematics(
            seg_kin, wb_kin, verbose=True,
            fz_N=fz_for_ck,
            stance_fz_threshold_n=CHECKPOINT_STANCE_FZ_THRESHOLD_N,
        )

        # Save kinematics to NPZ next to bilateral chain results
        kin_npz = os.path.join(os.path.dirname(bilateral_npz), f"{
                               walk_base}_COM_kinematics.npz")
        save_kw: dict = {
            "time": time_vec,
            "wb_pos_mm": wb_kin["pos"],
            "wb_vel_mm_s": wb_kin["vel"],
            "wb_acc_mm_s2": wb_kin["acc"],
        }
        for seg, v in seg_kin.items():
            save_kw[f"{seg}_pos_mm"] = v["pos"]
            save_kw[f"{seg}_vel_mm_s"] = v["vel"]
            save_kw[f"{seg}_acc_mm_s2"] = v["acc"]
            save_kw[f"{seg}_omega_rad_s"] = v["omega"]
            save_kw[f"{seg}_alpha_rad_s2"] = v["alpha"]
        if fp_aligned is not None:
            nu = int(fp_aligned["n_frames_used"])
            save_kw["grf_N"] = fp_aligned["grf_N"]
            save_kw["cop_lab_m"] = fp_aligned["cop_m"]
            save_kw["cop_lab_mm"] = fp_aligned["cop_mm"]
            save_kw["free_moment_Nm"] = fp_aligned["free_moment_Nm"]
            save_kw["free_moment_about_cop_Nm"] = fp_aligned["free_moment_about_cop_Nm"]
            save_kw["fp_sampling_rate_hz"] = np.array(
                [fp_aligned["sampling_rate_hz"]], dtype=float)
            save_kw["fp_n_frames_used"] = np.array([nu], dtype=np.int64)
            save_kw["fp_grf_filtered_hz"] = np.array(
                [DEFAULT_GRF_CUTOFF_HZ], dtype=float)
            fz = fp_aligned["grf_N"][:, 2]
            save_kw["stance_mask_fp"] = (
                fz > CHECKPOINT_STANCE_FZ_THRESHOLD_N).astype(np.int8)
        np.savez(kin_npz, **save_kw)
        print(f"Saved COM kinematics: {os.path.abspath(kin_npz)}")

        out_plot = os.path.join(os.path.dirname(bilateral_npz), f"{
                                walk_base}_COM_trajectory.png")
        plot_com_trajectories(
            wb_kin["pos"], time_vec,
            title=f"Whole-body COM (filtered pos) — {walk_base}",
            out_path=out_plot,
        )

        # Per-component plots for inspection (R_foot, R_shank)
        plot_dir = os.path.dirname(bilateral_npz)
        for seg in ("R_foot", "R_shank"):
            plot_segment_kinematics_components(
                seg, seg_kin, time_vec,
                out_path=os.path.join(plot_dir, f"{walk_base}_{
                                      seg}_pos_vel_acc_components.png"),
            )
    else:
        print("\nBilateral chain results not found. Searched directories for *_bilateral_chain_results.npz:")
        for d in (template_dir, c3d_dir, script_dir):
            print("  {}".format(os.path.abspath(d)))
        print(
            "Run svd_kabsch on a walk trial or pass an explicit file, e.g.\n"
            '  python kinematic_derivatives.py --bilateral-npz "'
            '{0}\\Walk_R04_bilateral_chain_results.npz"'.format(template_dir)
        )


if __name__ == "__main__":
    main()
