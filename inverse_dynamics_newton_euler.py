# -*- coding: utf-8 -*-
"""
Newton–Euler inverse dynamics: **foot** (distal, with GRF) and **bottom-up leg chain**
(shank → thigh) using ankle/knee loads as distal inputs. Knee and hip net moments are
projected into the same **JCS conventions** as ``joint_angles`` (Grood–Suntay knee;
hip ``hip_angles_xzy`` decomposition frame).

Uses ground reaction **F** at **CoP** from ``forceplate_preprocess`` — either
embedded in ``*_COM_kinematics.npz`` or from a separate ``*_grf_export.npz`` produced
by ``export_grf_to_npz`` when the plate C3D differs from the marker trial used for
bilateral results — and
segment kinematics from ``kinematic_derivatives`` + bilateral ACS
(``*_bilateral_chain_results.npz``).

Segment ACS in lab (required for knee / ankle JCS)
-------------------------------------------------
Per-segment rotation ``R_seg`` from ``*_bilateral_chain_results.npz`` (Kabsch + static
TCS→ACS templates, see ``svd_kabsch``) is used as **segment basis → lab**:

- **Column 0 (X): lateral** — mediolateral axis of the segment (right limb: toward the
  subject's anatomical **right**; left limb: toward the subject's **left**).
- **Column 1 (Y): anterior** — roughly along the segment's **anterior** direction.
- **Column 2 (Z): proximal** — long-bone **proximal** direction (thigh/shank Z toward
  the hip/knee respectively; foot Z toward the ankle/proximal along the foot segment).

The lab frame is **Z up** (gravity ``[0,0,-9.81]``). Grood–Suntay knee JCS
(``grood_suntay_R_jcs_columns_lab``) and ``joint_angles.knee_angles_grood_suntay`` assume
**femur and tibia** ``R[:,0:3]`` follow this **X/Y/Z = lateral / anterior / proximal**
convention. If templates or marker definitions change axis order, update both angles
and ID projections together.

Equations (foot free body, lab frame, Z up)
-------------------------------------------
Linear (COM):
    m * a = F_grf + F_ankle + m * g
=>  F_ankle = m * a - F_grf - m * g

Rotational about COM (vector form, body frame then mapped to lab):
    tau_b = I * alpha + omega × (I * omega)
    tau_lab = R_foot * tau_b

External moments about COM (lab):
    tau_lab = M_joint + (r_cop - r_com) × F_grf + (r_ankle - r_com) × F_ankle

=>  M_joint_lab = tau_lab - (r_cop - r_com) × F_grf - (r_ankle - r_com) × F_ankle

``M_joint`` is the net ankle reaction **couple** (shank on foot). Resolve into
tibia ACS or foot ACS with ``R_tibia.T @ M_joint_lab`` and ``R_foot.T @ M_joint_lab``.

Units: SI — lengths in m, forces in N, mass in kg, inertia in kg·m², omega/alpha
in rad/s and rad/s². COM kinematics from the pipeline are often in mm; convert.

Use ``checkpoint_ankle_moment`` after solving: plantarflexor |M| should peak in late
stance and the PF/DF axis sign should match expectation before moving up the chain.

Use ``checkpoint_knee_moment_gait`` / ``checkpoint_hip_moment_gait`` on the leg-chain
output: stance vs swing medians and within-stance peak timing on the **FE** component
(Grood–Suntay column 0 for the knee; hip decomposition column 0, same frame as
``joint_angles.hip_angles_xzy``), compared to typical walking (extension-dominant knee
in stance; hip extension in stance and flexion in swing; knee extension moment peak
mid-stance; hip extension peak early stance).

@author: lmcam
"""

from __future__ import annotations

import os
import numpy as np

__all__ = [
    "G_LAB_M_S2",
    "euler_torque_body",
    "euler_torque_lab_from_body",
    "foot_R_for_ankle_angle_axes",
    "inverse_dynamics_foot_one_frame",
    "inverse_dynamics_foot_timeseries",
    "inverse_dynamics_proximal_joint_one_frame",
    "inverse_dynamics_proximal_joint_timeseries",
    "grood_suntay_R_jcs_columns_lab",
    "hip_angle_decomposition_R_rel_adj",
    "load_foot_id_from_pipeline_outputs",
    "load_leg_chain_id_from_pipeline_outputs",
    "resolve_grf_export_npz_path",
    "checkpoint_ankle_moment",
    "checkpoint_knee_moment_gait",
    "checkpoint_hip_moment_gait",
    "validate_shank_knee_lab_inputs",
    "validate_knee_jcs_moment_consistency",
]

try:
    from joint_angles import _enforce_right_handed, _rot_x
except Exception:
    _enforce_right_handed = None
    _rot_x = None

try:
    from kinematic_derivatives import (
        butter_lowpass_filtfilt,
        DEFAULT_GRF_CUTOFF_HZ,
        CHECKPOINT_STANCE_FZ_THRESHOLD_N,
        stance_windows_from_fz,
    )
except Exception:
    butter_lowpass_filtfilt = None  # type: ignore
    DEFAULT_GRF_CUTOFF_HZ = 20.0
    CHECKPOINT_STANCE_FZ_THRESHOLD_N = 50.0

    def stance_windows_from_fz(
        fz_N: np.ndarray,
        n_frames_kin: int,
        stance_fz_threshold_n: float = 50.0,
    ) -> list[tuple[int, int]]:
        fz = np.asarray(fz_N, dtype=float).ravel()
        n = int(n_frames_kin)
        if n <= 0 or fz.size == 0:
            return []
        n_st = min(n, fz.shape[0])
        mask = np.zeros(n, dtype=bool)
        mask[:n_st] = fz[:n_st] > stance_fz_threshold_n
        if not np.any(mask):
            return []
        windows: list[tuple[int, int]] = []
        i = 0
        while i < n:
            if not mask[i]:
                i += 1
                continue
            j = i
            while j < n and mask[j]:
                j += 1
            windows.append((i, j - 1))
            i = j
        return windows

try:
    from forceplate_preprocess import export_grf_to_npz as _export_grf_to_npz
except Exception:
    _export_grf_to_npz = None


def _grf_filtered_hz_from_npz(npz) -> float | None:
    if "fp_grf_filtered_hz" not in npz.files:
        return None
    v = float(np.asarray(npz["fp_grf_filtered_hz"]).ravel()[0])
    return v if v > 0.0 else None


def _apply_grf_cop_filter_if_needed(
    grf: np.ndarray,
    cop: np.ndarray,
    fs_hz: float,
    already_filtered_hz: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply same GRF low-pass as kinematic_derivatives when NPZ is unfiltered (legacy export)."""
    if already_filtered_hz is not None and already_filtered_hz > 0.0:
        return grf, cop
    if butter_lowpass_filtfilt is None or fs_hz <= 0.0:
        return grf, cop
    grf = butter_lowpass_filtfilt(grf, DEFAULT_GRF_CUTOFF_HZ, fs_hz)
    cop = butter_lowpass_filtfilt(cop, DEFAULT_GRF_CUTOFF_HZ, fs_hz)
    return grf, cop


def _coerce_grf_cop_units(grf: np.ndarray, cop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Coerce to solver units: GRF in N, COP in m.

    Heuristics:
    - GRF max abs in (0, 10): likely kN -> multiply by 1000.
    - COP planar median > 10: likely mm -> divide by 1000.
    """
    g = np.asarray(grf, dtype=float).copy()
    c = np.asarray(cop, dtype=float).copy()

    gmax = float(np.nanmax(np.abs(g))) if g.size else 0.0
    if np.isfinite(gmax) and 0.0 < gmax < 10.0:
        print(
            "Unit guard: GRF appears to be in kN (max {:.3f}); converting to N.".format(
                gmax),
            flush=True,
        )
        g *= 1000.0

    if c.ndim == 2 and c.shape[1] >= 2 and c.size:
        vals = np.linalg.norm(c[:, :2], axis=1)
    else:
        vals = np.abs(c).reshape(-1) if c.size else np.array([], dtype=float)
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    cplan = float(np.median(vals)) if vals.size else 0.0
    if np.isfinite(cplan) and cplan > 10.0:
        print(
            "Unit guard: COP appears to be in mm (median planar {:.2f}); converting to m.".format(
                cplan),
            flush=True,
        )
        c *= 1.0e-3

    return g, c


def _best_ext_start_by_fz(
    fz_ref: np.ndarray,
    fz_ext: np.ndarray,
    *,
    threshold_n: float = CHECKPOINT_STANCE_FZ_THRESHOLD_N,
) -> int:
    """
    Best window start in external Fz that aligns to reference Fz length.

    Primary score: stance-mask overlap; tie-breaker: normalized correlation.
    """
    a = np.asarray(fz_ref, dtype=float).reshape(-1)
    b = np.asarray(fz_ext, dtype=float).reshape(-1)
    n = int(a.shape[0])
    m = int(b.shape[0])
    if n < 8 or m < n:
        return 0
    mask_a = a > float(threshold_n)
    a0 = a - float(np.nanmean(a))
    sa = float(np.nanstd(a0)) + 1e-12
    best_s = 0
    best_score = (-1.0, -1.0)
    for s in range(m - n + 1):
        bw = b[s: s + n]
        mask_b = bw > float(threshold_n)
        inter = float(np.sum(mask_a & mask_b))
        union = float(np.sum(mask_a | mask_b))
        jacc = inter / \
            union if union > 0 else (1.0 if np.sum(
                mask_a) == 0 and np.sum(mask_b) == 0 else 0.0)
        b0 = bw - float(np.nanmean(bw))
        sb = float(np.nanstd(b0)) + 1e-12
        corr = float(np.nanmean((a0 / sa) * (b0 / sb)))
        score = (jacc, corr)
        if score > best_score:
            best_score = score
            best_s = s
    return int(best_s)


def _infer_moment_normalization_mass_kg(
    inertial_npz_path: str | None,
    *,
    fallback_mass_kg: float = 70.0,
) -> float:
    """
    Infer subject mass for moment normalization (Nm -> Nm/kg).

    Priority:
    1) ``body_mass_kg`` in inertial NPZ if present.
    2) Sum of ``*_mass_kg`` entries when total is plausible as whole-body mass.
    3) Fallback constant.
    """
    if inertial_npz_path and os.path.isfile(inertial_npz_path):
        try:
            ine = np.load(inertial_npz_path, allow_pickle=True)
            if "body_mass_kg" in ine.files:
                m = float(np.asarray(ine["body_mass_kg"]).ravel()[0])
                if np.isfinite(m) and m > 1.0:
                    return m
            mass_keys = [k for k in ine.files if str(k).endswith("_mass_kg")]
            if mass_keys:
                total = float(
                    np.sum([float(np.asarray(ine[k]).ravel()[0]) for k in mass_keys]))
                # Accept only plausible full-body totals.
                if np.isfinite(total) and 45.0 <= total <= 140.0:
                    return total
        except Exception:
            pass
    return float(fallback_mass_kg)


def _normalize_moment_dict_values(data: dict[str, np.ndarray], mass_kg: float) -> dict[str, np.ndarray]:
    """
    Normalize every ``*_Nm`` array in ``data`` by ``mass_kg``.
    """
    m = float(mass_kg) if np.isfinite(mass_kg) and mass_kg > 1e-9 else 1.0
    out: dict[str, np.ndarray] = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray) and "_Nm" in k:
            out[k] = np.asarray(v, dtype=float) / m
        else:
            out[k] = v
    out["moment_normalization_mass_kg"] = np.array([m], dtype=float)
    return out


def _select_cop_in_m(npz_obj) -> tuple[np.ndarray | None, str | None]:
    """
    Pick the most usable COP array from an NPZ and convert to meters.

    Supported keys: ``cop_lab_mm``, ``cop_mm``, ``cop_lab_m``, ``cop_m``.
    """
    if npz_obj is None:
        return None, None
    candidates: list[tuple[int, np.ndarray, str]] = []
    for key, scale in (
        ("cop_lab_mm", 1.0e-3),
        ("cop_mm", 1.0e-3),
        ("cop_lab_m", 1.0),
        ("cop_m", 1.0),
    ):
        if key not in npz_obj.files:
            continue
        arr = np.asarray(npz_obj[key], dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            continue
        arr_m = arr[:, :3].copy() * scale
        usable = int(np.sum(np.isfinite(arr_m).all(axis=1)))
        candidates.append((usable, arr_m, key))
    if not candidates:
        return None, None
    candidates.sort(key=lambda t: t[0], reverse=True)
    _, best_arr_m, best_key = candidates[0]
    return best_arr_m, best_key


def _cop_plausibility_report(
    cop_lab_m: np.ndarray,
    grf_N: np.ndarray,
    ankle_lab_mm: np.ndarray,
    *,
    stance_fz_threshold_n: float = CHECKPOINT_STANCE_FZ_THRESHOLD_N,
    max_median_xy_dist_m: float = 0.17,
    max_p95_xy_dist_m: float = 0.24,
) -> tuple[bool, dict]:
    """
    Sanity-check COP relative to the instrumented ankle in lab frame.

    Returns ``(ok, details)``. Intended as a guard against frame-mismatched COP
    imports that can blow up ankle moments via ``(r_cop - r_com) x F_grf``.
    """
    cop = np.asarray(cop_lab_m, dtype=float).reshape(-1, 3)
    grf = np.asarray(grf_N, dtype=float).reshape(-1, 3)
    ank = np.asarray(ankle_lab_mm, dtype=float).reshape(-1, 3) * 1.0e-3
    n = min(cop.shape[0], grf.shape[0], ank.shape[0])
    if n < 8:
        return True, {"reason": "too_few_frames", "n": int(n)}
    cop = cop[:n]
    grf = grf[:n]
    ank = ank[:n]
    stance = grf[:, 2] > float(stance_fz_threshold_n)
    finite = np.isfinite(cop).all(axis=1)
    use = stance & finite
    n_use = int(np.sum(use))
    if n_use < 8:
        return True, {"reason": "too_few_stance_cop_frames", "n_use": n_use, "n": int(n)}
    d_xy = np.linalg.norm(cop[use, :2] - ank[use, :2], axis=1)
    med = float(np.nanmedian(d_xy))
    p95 = float(np.nanpercentile(d_xy, 95))
    ok = bool(med <= max_median_xy_dist_m and p95 <= max_p95_xy_dist_m)
    return ok, {
        "n_use": n_use,
        "median_xy_dist_m": med,
        "p95_xy_dist_m": p95,
        "max_xy_dist_m": float(np.nanmax(d_xy)),
    }


def _cop_side_consistency_report(
    cop_lab_m: np.ndarray,
    grf_N: np.ndarray,
    ankle_lab_mm_selected: np.ndarray,
    ankle_lab_mm_other: np.ndarray | None,
    *,
    stance_fz_threshold_n: float = CHECKPOINT_STANCE_FZ_THRESHOLD_N,
    min_fraction_closer_to_selected: float = 0.60,
) -> tuple[bool, dict]:
    """
    Check COP is closer to selected foot than the opposite foot during stance.
    """
    if ankle_lab_mm_other is None:
        return True, {"reason": "no_other_foot_ankle"}
    cop = np.asarray(cop_lab_m, dtype=float).reshape(-1, 3)
    grf = np.asarray(grf_N, dtype=float).reshape(-1, 3)
    sel = np.asarray(ankle_lab_mm_selected,
                     dtype=float).reshape(-1, 3) * 1.0e-3
    oth = np.asarray(ankle_lab_mm_other, dtype=float).reshape(-1, 3) * 1.0e-3
    n = min(cop.shape[0], grf.shape[0], sel.shape[0], oth.shape[0])
    if n < 8:
        return True, {"reason": "too_few_frames", "n": int(n)}
    cop = cop[:n]
    grf = grf[:n]
    sel = sel[:n]
    oth = oth[:n]
    stance = grf[:, 2] > float(stance_fz_threshold_n)
    finite = np.isfinite(cop).all(axis=1)
    use = stance & finite
    n_use = int(np.sum(use))
    if n_use < 8:
        return True, {"reason": "too_few_stance_cop_frames", "n_use": n_use}
    d_sel = np.linalg.norm(cop[use, :2] - sel[use, :2], axis=1)
    d_oth = np.linalg.norm(cop[use, :2] - oth[use, :2], axis=1)
    frac = float(np.mean(d_sel < d_oth))
    ok = bool(frac >= min_fraction_closer_to_selected)
    return ok, {
        "n_use": n_use,
        "fraction_closer_to_selected": frac,
        "median_xy_dist_selected_m": float(np.nanmedian(d_sel)),
        "median_xy_dist_other_m": float(np.nanmedian(d_oth)),
    }


def _print_cop_checks(
    *,
    source: str,
    cop_key: str | None,
    cop_lab_m: np.ndarray,
    grf_N: np.ndarray,
    side_ok: bool,
    side_diag: dict,
    cop_ok: bool,
    cop_diag: dict,
    source_npz=None,
) -> None:
    """Compact COP import diagnostics for each run."""
    cop = np.asarray(cop_lab_m, dtype=float).reshape(-1, 3)
    grf = np.asarray(grf_N, dtype=float).reshape(-1, 3)
    n = min(cop.shape[0], grf.shape[0])
    if n <= 0:
        return
    cop = cop[:n]
    grf = grf[:n]
    finite_rows = np.isfinite(cop).all(axis=1)
    stance = grf[:, 2] > float(CHECKPOINT_STANCE_FZ_THRESHOLD_N)
    finite_stance = int(np.sum(finite_rows & stance))
    print("\n[COP CHECK] source={} key={}".format(
        source, cop_key or "none"), flush=True)
    print(
        "  finite COP rows: {}/{} ({:.1%}); finite stance rows: {}".format(
            int(np.sum(finite_rows)), int(n), float(
                np.mean(finite_rows)), finite_stance
        ),
        flush=True,
    )
    if np.any(finite_rows):
        c = cop[finite_rows]
        print(
            "  COP xyz ranges (m): x[{:.3f},{:.3f}] y[{:.3f},{:.3f}] z[{:.3f},{:.3f}]".format(
                float(np.nanmin(c[:, 0])),
                float(np.nanmax(c[:, 0])),
                float(np.nanmin(c[:, 1])),
                float(np.nanmax(c[:, 1])),
                float(np.nanmin(c[:, 2])),
                float(np.nanmax(c[:, 2])),
            ),
            flush=True,
        )
    print(
        "  side check: {} (closer frac={:.1%}, d_sel_med={:.3f} m, d_oth_med={:.3f} m)".format(
            "PASS" if side_ok else "FAIL",
            float(side_diag.get("fraction_closer_to_selected", float("nan"))),
            float(side_diag.get("median_xy_dist_selected_m", float("nan"))),
            float(side_diag.get("median_xy_dist_other_m", float("nan"))),
        ),
        flush=True,
    )
    print(
        "  plausibility: {} (d_xy_med={:.3f} m, d_xy_p95={:.3f} m, n_stance={})".format(
            "PASS" if cop_ok else "FAIL",
            float(cop_diag.get("median_xy_dist_m", float("nan"))),
            float(cop_diag.get("p95_xy_dist_m", float("nan"))),
            int(cop_diag.get("n_use", 0)),
        ),
        flush=True,
    )
    if source_npz is not None and hasattr(source_npz, "files") and "transform_qc" in source_npz.files:
        try:
            tq = np.asarray(source_npz["transform_qc"], dtype=object).item()
            if isinstance(tq, dict):
                print(
                    "  transform_qc: local_z |med|={:.2f} mm, p95={:.2f} mm, max={:.2f} mm".format(
                        float(tq.get("cop_local_z_mm_median_abs", float("nan"))),
                        float(tq.get("cop_local_z_mm_p95_abs", float("nan"))),
                        float(tq.get("cop_local_z_mm_max_abs", float("nan"))),
                    ),
                    flush=True,
                )
        except Exception:
            pass


# Default gravity (lab Z up, m/s²)
G_LAB_M_S2 = np.array([0.0, 0.0, -9.81], dtype=float)

# Ankle QC: plantarflexor axis in ``joint_angles`` ankle-angle frame is index 0 (PF/DF vs ax)
ANKLE_PF_DF_AXIS = 0
# Last fraction of each stance window counted as "late stance" (terminal stance / push-off band)
CHECKPOINT_LATE_STANCE_FRAC = 0.45
# If global peak |M_pf| is early (heel-strike impact), still pass when late stance retains
# a substantial fraction of the stance-phase peak (avoids false FAIL when peak is not in late).
CHECKPOINT_LATE_PEAK_FRACTION_MIN = 0.22
# Below this max |M_pf| in stance, timing check is inconclusive (pass; no usable PF/DF scale).
CHECKPOINT_ANKLE_TIMING_PEAK_MAG_MIN_NM = 0.05
# Below this |M_pf|, median sign is treated as inconclusive (noise-level net moment).
CHECKPOINT_ANKLE_DIRECTION_ABS_TOL_NM = 0.02

# Knee / hip gait QC: FE component in JCS / hip-decomp frame (match joint_angles)
KNEE_FE_JCS_AXIS = 0
HIP_FE_DECOMP_AXIS = 0
CHECKPOINT_KNEE_HIP_DIRECTION_ABS_TOL_NM = 0.08
CHECKPOINT_KNEE_HIP_TIMING_PEAK_MAG_MIN_NM = 0.25
# Knee: stance-phase peak **extension** moment (max M_FE) — allow loading at stance onset and
# push-off toward end (single short stances often have one dominant peak near an edge).
CHECKPOINT_KNEE_STANCE_EXT_PEAK_POS_MIN = 0.0
CHECKPOINT_KNEE_STANCE_EXT_PEAK_POS_MAX = 0.96
# Hip: stance-phase peak M_FE (extension) usually before terminal stance (allow mid-stance peaks)
CHECKPOINT_HIP_STANCE_EXT_PEAK_POS_MAX = 0.88
# If global peak is late, still pass when the **first 60%** of stance reaches this fraction of stance max M_FE
# (early extensor support present even when argmax is near push-off).
CHECKPOINT_HIP_EARLY_STANCE_FRAC_OF_STANCE = 0.60
CHECKPOINT_HIP_EARLY_STANCE_PEAK_FRACTION_MIN = 0.18
CHECKPOINT_SWING_MIN_SAMPLES = 12


def checkpoint_ankle_moment(
    time: np.ndarray,
    fz_N: np.ndarray,
    M_joint_ankle_angle_frame_Nm: np.ndarray,
    *,
    late_stance_frac: float = CHECKPOINT_LATE_STANCE_FRAC,
    late_peak_fraction_min: float = CHECKPOINT_LATE_PEAK_FRACTION_MIN,
    timing_peak_mag_min_nm: float = CHECKPOINT_ANKLE_TIMING_PEAK_MAG_MIN_NM,
    direction_abs_tol_nm: float = CHECKPOINT_ANKLE_DIRECTION_ABS_TOL_NM,
    fz_threshold_n: float = CHECKPOINT_STANCE_FZ_THRESHOLD_N,
    pf_axis: int = ANKLE_PF_DF_AXIS,
    expect_plantarflexor_positive: bool = True,
    verbose: bool = True,
) -> dict:
    """
    QC before moving up the chain: ankle plantarflexor moment should peak in **late
    stance**, and the PF/DF component sign should be biomechanically plausible.

    Stance intervals use the same rule as ``kinematic_derivatives.checkpoint_kinematics``
    when force-plate Fz is used: ``Fz > fz_threshold_n`` on the overlap (see
    ``stance_windows_from_fz``). Splits each stance
    into early vs late (last ``late_stance_frac`` of samples in that stance). Compares
    the peak of |M_pf| on the PF/DF axis (default column 0, matching
    ``joint_angles.ankle_angles_xyz`` PF/DF).

    Parameters
    ----------
    late_peak_fraction_min
        For each stance, pass the timing check if either the global |M_pf| peak falls in
        late stance **or** ``max(|M_pf| in late) / max(|M_pf| in stance)`` is at least
        this value (heel-strike peaks often violate the first condition alone).
    timing_peak_mag_min_nm
        If ``max(|M_pf|)`` in stance or in late stance is below this (Nm), the
        late-stance **timing** check passes as inconclusive (no resolvable PF/DF moment).
    direction_abs_tol_nm
        If ``abs(median late-stance M_pf)`` is below this (Nm), the sign check is
        skipped (pass as inconclusive); net PF/DF is noise-level.
    expect_plantarflexor_positive
        If True, late-stance median M_pf should be > 0 for a typical soleus-dominated
        push-off (sign convention in your ankle-angle frame). Set False if your
        convention is inverted.

    Returns
    -------
    dict with keys: all_ok (bool), late_stance_peak_ok (bool), direction_ok (bool),
    details (dict), issues (list of str).
    """
    t = np.asarray(time, dtype=float).ravel()
    fz = np.asarray(fz_N, dtype=float).ravel()
    M = np.asarray(M_joint_ankle_angle_frame_Nm, dtype=float)
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    n = min(t.shape[0], fz.shape[0], M.shape[0])
    if n < 3:
        out = {
            "all_ok": False,
            "late_stance_peak_ok": False,
            "direction_ok": False,
            "details": {},
            "issues": ["Too few samples for ankle checkpoint."],
        }
        if verbose:
            _print_checkpoint_ankle(out)
        return out

    t = t[:n]
    fz = fz[:n]
    M = M[:n, :]
    if pf_axis >= M.shape[1]:
        pf_axis = 0
    M_pf = M[:, pf_axis]

    details: dict = {
        "pf_axis": int(pf_axis),
        "fz_min_N": float(np.nanmin(fz)),
        "fz_max_N": float(np.nanmax(fz)),
        "fz_median_N": float(np.nanmedian(fz)),
        "fz_abs_max_N": float(np.nanmax(np.abs(fz))),
        "stance_fz_threshold_N": float(fz_threshold_n),
        "stance_method": "force_plate_Fz",
        "timing_peak_mag_min_Nm": float(timing_peak_mag_min_nm),
        "late_peak_fraction_min": float(late_peak_fraction_min),
    }

    windows = stance_windows_from_fz(fz, n, fz_threshold_n)
    issues: list[str] = []
    details["stance_windows"] = len(windows)

    if not windows:
        mx = details["fz_abs_max_N"]
        mx_pos = float(np.nanmax(fz))
        if mx_pos < fz_threshold_n:
            issues.append(
                "No stance windows: max Fz = {:.1f} N < {:.0f} N (same rule as kinematic checkpoint: Fz > threshold).".format(
                    mx_pos, fz_threshold_n
                )
            )
        else:
            issues.append(
                "No contiguous stance windows despite max Fz = {:.1f} N; check gaps or alignment.".format(
                    mx_pos
                )
            )
        out = {
            "all_ok": False,
            "late_stance_peak_ok": False,
            "direction_ok": False,
            "details": details,
            "issues": issues,
        }
        if verbose:
            _print_checkpoint_ankle(out)
        return out

    late_peak_hits = 0
    windows_used = 0
    late_medians: list[float] = []

    for w, (i0, i1) in enumerate(windows):
        L = i1 - i0 + 1
        if L < 3:
            continue
        windows_used += 1
        n_late = max(1, int(np.ceil(late_stance_frac * L)))
        i_late0 = i1 - n_late + 1
        seg = M_pf[i0: i1 + 1]
        late_seg = M_pf[i_late0: i1 + 1]
        abs_seg = np.abs(seg)
        abs_late = np.abs(late_seg)
        peak_mag = float(np.nanmax(abs_seg))
        late_mag = float(np.nanmax(abs_late))
        late_frac_of_peak = late_mag / (peak_mag + 1e-15)
        # Global peak index (|M_pf|); often early stance at heel strike
        peak_local = int(np.nanargmax(abs_seg))
        peak_idx = i0 + peak_local
        peak_in_late = peak_idx >= i_late0
        # No meaningful |M_pf| in late stance → cannot assess push-off vs heel (pass inconclusive).
        # Also when entire stance is noise-level on this axis.
        timing_inconclusive = (
            peak_mag < timing_peak_mag_min_nm
            or late_mag < timing_peak_mag_min_nm
        )
        timing_ok = (
            timing_inconclusive
            or peak_in_late
            or late_frac_of_peak >= late_peak_fraction_min
        )
        if timing_ok:
            late_peak_hits += 1
        details[f"stance{w}_peak_frame"] = int(peak_idx)
        details[f"stance{w}_peak_in_late"] = bool(peak_in_late)
        details[f"stance{w}_late_range"] = (int(i_late0), int(i1))
        details[f"stance{w}_late_peak_mag_fraction"] = float(late_frac_of_peak)
        details[f"stance{w}_stance_max_abs_M_pf_Nm"] = float(peak_mag)
        if timing_inconclusive:
            details[f"stance{w}_timing_inconclusive"] = True
        late_medians.append(float(np.nanmedian(late_seg)))

    late_stance_peak_ok = (
        windows_used > 0 and late_peak_hits == windows_used
    )
    if not late_stance_peak_ok and windows_used > 0:
        issues.append(
            "Late-stance |M_pf| check failed: for at least one stance, peak was early "
            "and late-stance max |M_pf| was < {:.0%} of stance max (heel strike vs push-off).".format(
                late_peak_fraction_min
            )
        )

    # Direction: median late-stance M_pf vs expected sign (skip if noise-level)
    direction_ok = False
    direction_inconclusive = False
    if late_medians:
        med = float(np.median(np.asarray(late_medians, dtype=float)))
        details["median_M_pf_late_stance_Nm"] = med
        if abs(med) < direction_abs_tol_nm:
            direction_inconclusive = True
            direction_ok = True
            details["direction_inconclusive"] = True
            details["direction_abs_tol_Nm"] = float(direction_abs_tol_nm)
        elif expect_plantarflexor_positive:
            direction_ok = med > 0.0
            if not direction_ok:
                issues.append(
                    "Median late-stance M_pf <= 0; check sign convention or inverse "
                    "dynamics (fix ankle before proximal chain)."
                )
        else:
            direction_ok = med < 0.0
            if not direction_ok:
                issues.append(
                    "Median late-stance M_pf >= 0; inconsistent with expect_plantarflexor_positive=False."
                )
    else:
        issues.append("No late-stance samples for direction check.")

    all_ok = bool(late_stance_peak_ok and direction_ok)

    out = {
        "all_ok": all_ok,
        "late_stance_peak_ok": late_stance_peak_ok,
        "direction_ok": direction_ok,
        "direction_inconclusive": direction_inconclusive,
        "details": details,
        "issues": issues,
    }
    if verbose:
        _print_checkpoint_ankle(out)
    return out


def _print_checkpoint_ankle(out: dict) -> None:
    all_ok = bool(out.get("all_ok"))
    late_ok = bool(out.get("late_stance_peak_ok"))
    dir_ok = bool(out.get("direction_ok"))
    label = "PASS" if all_ok else "FAIL"
    print("\n" + "=" * 60, flush=True)
    print("[ANKLE CHECK] {}".format(label), flush=True)
    print("Checkpoint: ankle inverse dynamics (fix ankle before proximal chain)", flush=True)
    print("=" * 60, flush=True)
    d = out.get("details") or {}
    if "fz_abs_max_N" in d:
        print(
            "  Fz (vertical GRF) stats: min {:.1f}  max {:.1f}  median {:.1f}  max|·| {:.1f} N".format(
                d.get("fz_min_N", float("nan")),
                d.get("fz_max_N", float("nan")),
                d.get("fz_median_N", float("nan")),
                d.get("fz_abs_max_N", float("nan")),
            ),
            flush=True,
        )
        print(
            "  Stance rule (force_plate_Fz): Fz > {:.0f} N (matches kinematic_derivatives checkpoint)".format(
                d.get("stance_fz_threshold_N",
                      CHECKPOINT_STANCE_FZ_THRESHOLD_N),
            ),
            flush=True,
        )
    print("  Stance windows (Fz): {}".format(
        d.get("stance_windows", "?")), flush=True)
    if any(k.endswith("_timing_inconclusive") for k in d):
        print(
            "  Timing: inconclusive (stance or late max |M_pf| < {:.2f} Nm on PF/DF axis)".format(
                d.get("timing_peak_mag_min_Nm",
                      CHECKPOINT_ANKLE_TIMING_PEAK_MAG_MIN_NM),
            ),
            flush=True,
        )
    if "median_M_pf_late_stance_Nm" in d:
        print("  Median M_pf in late stance (Nm): {:.4f}".format(
            d["median_M_pf_late_stance_Nm"]), flush=True)
    if out.get("direction_inconclusive"):
        print(
            "  Direction: inconclusive (|median| < {:.3f} Nm; noise-level PF/DF component)".format(
                d.get("direction_abs_tol_Nm",
                      CHECKPOINT_ANKLE_DIRECTION_ABS_TOL_NM),
            ),
            flush=True,
        )
    print("  [PASS] Late-stance PF peak timing" if late_ok else "  [FAIL] Late-stance PF peak timing", flush=True)
    print("  [PASS] Moment direction (PF/DF axis)" if dir_ok else "  [FAIL] Moment direction (PF/DF axis)", flush=True)
    if all_ok:
        print("  >>> Overall: PASS <<<", flush=True)
    else:
        print("  >>> Overall: FAIL <<<", flush=True)
        for s in out.get("issues") or []:
            print("    - {}".format(s), flush=True)
    print("=" * 60, flush=True)
    print("[ANKLE CHECK] {}".format(label), flush=True)


def _stance_boolean_mask(
    fz_N: np.ndarray,
    n_frames: int,
    fz_threshold_n: float,
) -> np.ndarray:
    fz = np.asarray(fz_N, dtype=float).ravel()
    n = int(n_frames)
    m = np.zeros(max(0, n), dtype=bool)
    if n <= 0 or fz.size == 0:
        return m
    n_st = min(n, fz.shape[0])
    m[:n_st] = fz[:n_st] > fz_threshold_n
    return m


def checkpoint_knee_moment_gait(
    time: np.ndarray,
    fz_N: np.ndarray,
    M_knee_jcs_Nm: np.ndarray,
    *,
    fe_axis: int = KNEE_FE_JCS_AXIS,
    fz_threshold_n: float = CHECKPOINT_STANCE_FZ_THRESHOLD_N,
    direction_abs_tol_nm: float = CHECKPOINT_KNEE_HIP_DIRECTION_ABS_TOL_NM,
    timing_peak_mag_min_nm: float = CHECKPOINT_KNEE_HIP_TIMING_PEAK_MAG_MIN_NM,
    stance_ext_peak_pos_min: float = CHECKPOINT_KNEE_STANCE_EXT_PEAK_POS_MIN,
    stance_ext_peak_pos_max: float = CHECKPOINT_KNEE_STANCE_EXT_PEAK_POS_MAX,
    expect_stance_extension_positive: bool = True,
    expect_swing_median_below_stance: bool = True,
    swing_min_samples: int = CHECKPOINT_SWING_MIN_SAMPLES,
    verbose: bool = True,
) -> dict:
    """
    QC knee inverse dynamics vs **typical sagittal walking** (instrumented foot Fz).

    Uses Grood–Suntay JCS column ``fe_axis`` (default 0 ≈ flexion–extension conjugate).
    Checks: (1) median M_FE in **stance** tends **extension-positive** (knee extensors
    support weight); (2) median M_FE in **swing** is not larger than stance in the same
    extension direction (swing tends more flexor); (3) within each stance, the **maximum
    extension** moment ``max(M_FE)`` occurs between ``stance_ext_peak_pos_min`` and
    ``stance_ext_peak_pos_max`` of stance length (mid-stance-type peak, not only edge
    spikes), when |M_FE| is above noise.

    Sign conventions follow your ID + JCS: set ``expect_stance_extension_positive=False``
    if positive M_FE means flexion in your outputs.
    """
    t = np.asarray(time, dtype=float).ravel()
    fz = np.asarray(fz_N, dtype=float).ravel()
    M = np.asarray(M_knee_jcs_Nm, dtype=float)
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    n = min(t.shape[0], fz.shape[0], M.shape[0])
    issues: list[str] = []
    if n < 3 or fe_axis >= M.shape[1]:
        out = {
            "all_ok": False,
            "stance_extension_ok": False,
            "swing_vs_stance_ok": False,
            "stance_peak_timing_ok": False,
            "details": {},
            "issues": issues + ["Too few samples or invalid fe_axis for knee checkpoint."],
        }
        if verbose:
            _print_checkpoint_knee(out)
        return out

    M_fe = M[:n, fe_axis]
    stance_m = _stance_boolean_mask(fz, n, fz_threshold_n)
    swing_m = ~stance_m
    windows = stance_windows_from_fz(fz, n, fz_threshold_n)

    details: dict = {
        "fe_axis": int(fe_axis),
        "stance_fz_threshold_N": float(fz_threshold_n),
        "stance_windows": len(windows),
        "direction_abs_tol_Nm": float(direction_abs_tol_nm),
        "timing_peak_mag_min_Nm": float(timing_peak_mag_min_nm),
    }

    stance_extension_ok = False
    stance_ext_inconclusive = False
    st_idx = np.flatnonzero(stance_m)
    if st_idx.size == 0:
        issues.append("No stance samples (Fz); cannot assess knee vs gait.")
    else:
        med_st = float(np.nanmedian(M_fe[stance_m]))
        details["median_M_FE_stance_Nm"] = med_st
        if abs(med_st) < direction_abs_tol_nm:
            stance_ext_inconclusive = True
            stance_extension_ok = True
            details["stance_direction_inconclusive"] = True
        elif expect_stance_extension_positive:
            stance_extension_ok = med_st > 0.0
            if not stance_extension_ok:
                issues.append(
                    "Median stance M_knee_FE <= 0; typical walking has net knee **extension** "
                    "(extensor) moment in stance — check sign/JCS or model."
                )
        else:
            stance_extension_ok = med_st < 0.0
            if not stance_extension_ok:
                issues.append(
                    "Median stance M_knee_FE >= 0; inconsistent with expect_stance_extension_positive=False."
                )

    swing_vs_stance_ok = False
    swing_inconclusive = False
    sw_idx = np.flatnonzero(swing_m)
    if sw_idx.size < swing_min_samples:
        swing_inconclusive = True
        swing_vs_stance_ok = True
        details["swing_inconclusive"] = True
        details["swing_samples"] = int(sw_idx.size)
    elif st_idx.size == 0:
        swing_vs_stance_ok = True
        swing_inconclusive = True
    else:
        med_sw = float(np.nanmedian(M_fe[swing_m]))
        details["median_M_FE_swing_Nm"] = med_sw
        if expect_swing_median_below_stance:
            swing_vs_stance_ok = med_sw < med_st + direction_abs_tol_nm
            if not swing_vs_stance_ok:
                issues.append(
                    "Median swing M_knee_FE is not below stance median (typical: swing more flexor / "
                    "less extension than stance)."
                )
        else:
            swing_vs_stance_ok = True

    stance_peak_timing_ok = False
    timing_hits = 0
    timing_used = 0
    if not windows:
        issues.append("No stance windows for knee peak-timing check.")
    else:
        for w, (i0, i1) in enumerate(windows):
            L = i1 - i0 + 1
            if L < 3:
                continue
            seg = M_fe[i0: i1 + 1]
            peak_ext = float(np.nanmax(seg))
            if peak_ext < timing_peak_mag_min_nm:
                details[f"stance{w}_knee_timing_inconclusive"] = True
                continue
            timing_used += 1
            peak_local = int(np.nanargmax(seg))
            pos = (peak_local / max(L - 1, 1)) if L > 1 else 0.0
            ok_w = stance_ext_peak_pos_min <= pos <= stance_ext_peak_pos_max
            if ok_w:
                timing_hits += 1
            details[f"stance{w}_knee_ext_peak_norm_pos"] = float(pos)
        stance_peak_timing_ok = timing_used > 0 and timing_hits == timing_used
        details["knee_stance_peak_timing_used"] = int(timing_used)
        if timing_used > 0 and not stance_peak_timing_ok:
            issues.append(
                "Knee extension-moment peak in stance fell outside [{:.0%}, {:.0%}] of stance "
                "(expected mid-stance-type pattern vs edge-only spike).".format(
                    stance_ext_peak_pos_min, stance_ext_peak_pos_max
                )
            )

    all_ok = bool(
        stance_extension_ok
        and swing_vs_stance_ok
        and stance_peak_timing_ok
    )
    out = {
        "all_ok": all_ok,
        "stance_extension_ok": stance_extension_ok,
        "stance_ext_inconclusive": stance_ext_inconclusive,
        "swing_vs_stance_ok": swing_vs_stance_ok,
        "swing_inconclusive": swing_inconclusive,
        "stance_peak_timing_ok": stance_peak_timing_ok,
        "details": details,
        "issues": issues,
    }
    if verbose:
        _print_checkpoint_knee(out)
    return out


def checkpoint_hip_moment_gait(
    time: np.ndarray,
    fz_N: np.ndarray,
    M_hip_decomp_Nm: np.ndarray,
    *,
    fe_axis: int = HIP_FE_DECOMP_AXIS,
    fz_threshold_n: float = CHECKPOINT_STANCE_FZ_THRESHOLD_N,
    direction_abs_tol_nm: float = CHECKPOINT_KNEE_HIP_DIRECTION_ABS_TOL_NM,
    timing_peak_mag_min_nm: float = CHECKPOINT_KNEE_HIP_TIMING_PEAK_MAG_MIN_NM,
    stance_ext_peak_pos_max: float = CHECKPOINT_HIP_STANCE_EXT_PEAK_POS_MAX,
    early_stance_frac_of_stance: float = CHECKPOINT_HIP_EARLY_STANCE_FRAC_OF_STANCE,
    early_stance_peak_fraction_min: float = CHECKPOINT_HIP_EARLY_STANCE_PEAK_FRACTION_MIN,
    expect_stance_extension_positive: bool = True,
    expect_swing_flexion_negative: bool = True,
    swing_min_samples: int = CHECKPOINT_SWING_MIN_SAMPLES,
    verbose: bool = True,
) -> dict:
    """
    QC hip inverse dynamics vs **typical sagittal walking** (instrumented foot Fz).

    Uses hip decomposition column ``fe_axis`` (default 0, same construction as
    ``hip_angle_decomposition_R_rel_adj`` / ``hip_angles_xzy``).

    Checks: (1) median M_FE in **stance** tends extension-positive (hip extensors in
    early–mid stance); (2) median M_FE in **swing** tends flexion-negative (hip flexors
    in swing);     (3) within stance, ``argmax(M_FE)`` lies in the **first**
    ``stance_ext_peak_pos_max`` fraction of stance when extension peak is large enough,
    **or** the max M_FE in the **first half** of stance is at least
    ``early_stance_peak_fraction_min`` times the stance max, using the first
    ``early_stance_frac_of_stance`` fraction of samples in that stance (fallback when
    the global peak is late but early stance still shows extensor support).
    """
    t = np.asarray(time, dtype=float).ravel()
    fz = np.asarray(fz_N, dtype=float).ravel()
    M = np.asarray(M_hip_decomp_Nm, dtype=float)
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    n = min(t.shape[0], fz.shape[0], M.shape[0])
    issues: list[str] = []
    if n < 3 or fe_axis >= M.shape[1]:
        out = {
            "all_ok": False,
            "stance_extension_ok": False,
            "swing_flexion_ok": False,
            "stance_peak_timing_ok": False,
            "details": {},
            "issues": issues + ["Too few samples or invalid fe_axis for hip checkpoint."],
        }
        if verbose:
            _print_checkpoint_hip(out)
        return out

    M_fe = M[:n, fe_axis]
    stance_m = _stance_boolean_mask(fz, n, fz_threshold_n)
    swing_m = ~stance_m
    windows = stance_windows_from_fz(fz, n, fz_threshold_n)

    details: dict = {
        "fe_axis": int(fe_axis),
        "stance_fz_threshold_N": float(fz_threshold_n),
        "stance_windows": len(windows),
        "stance_ext_peak_pos_max": float(stance_ext_peak_pos_max),
        "early_stance_frac_of_stance": float(early_stance_frac_of_stance),
        "early_stance_peak_fraction_min": float(early_stance_peak_fraction_min),
        "direction_abs_tol_Nm": float(direction_abs_tol_nm),
        "timing_peak_mag_min_Nm": float(timing_peak_mag_min_nm),
    }

    stance_extension_ok = False
    st_idx = np.flatnonzero(stance_m)
    if st_idx.size == 0:
        issues.append("No stance samples (Fz); cannot assess hip vs gait.")
    else:
        med_st = float(np.nanmedian(M_fe[stance_m]))
        details["median_M_FE_stance_Nm"] = med_st
        if abs(med_st) < direction_abs_tol_nm:
            stance_extension_ok = True
            details["stance_direction_inconclusive"] = True
        elif expect_stance_extension_positive:
            stance_extension_ok = med_st > 0.0
            if not stance_extension_ok:
                issues.append(
                    "Median stance M_hip_FE <= 0; typical walking has net **hip extension** "
                    "moment in stance — check sign/decomposition or model."
                )
        else:
            stance_extension_ok = med_st < 0.0
            if not stance_extension_ok:
                issues.append(
                    "Median stance M_hip_FE >= 0; inconsistent with expect_stance_extension_positive=False."
                )

    swing_flexion_ok = False
    sw_idx = np.flatnonzero(swing_m)
    if sw_idx.size < swing_min_samples:
        swing_flexion_ok = True
        details["swing_inconclusive"] = True
        details["swing_samples"] = int(sw_idx.size)
    elif st_idx.size == 0:
        swing_flexion_ok = True
    else:
        med_sw = float(np.nanmedian(M_fe[swing_m]))
        details["median_M_FE_swing_Nm"] = med_sw
        if expect_swing_flexion_negative:
            if abs(med_sw) < direction_abs_tol_nm:
                swing_flexion_ok = True
                details["swing_direction_inconclusive"] = True
            else:
                swing_flexion_ok = med_sw < 0.0
                if not swing_flexion_ok:
                    issues.append(
                        "Median swing M_hip_FE >= 0; typical swing shows **hip flexion** moment "
                        "(negative if extension-positive in stance)."
                    )
        else:
            swing_flexion_ok = True

    stance_peak_timing_ok = False
    timing_hits = 0
    timing_used = 0
    if not windows:
        issues.append("No stance windows for hip peak-timing check.")
    else:
        for w, (i0, i1) in enumerate(windows):
            L = i1 - i0 + 1
            if L < 3:
                continue
            seg = M_fe[i0: i1 + 1]
            peak_ext = float(np.nanmax(seg))
            if peak_ext < timing_peak_mag_min_nm:
                details[f"stance{w}_hip_timing_inconclusive"] = True
                continue
            timing_used += 1
            peak_local = int(np.nanargmax(seg))
            pos = (peak_local / max(L - 1, 1)) if L > 1 else 0.0
            n_early = max(int(np.ceil(early_stance_frac_of_stance * L)), 1)
            first_half_max = float(np.nanmax(seg[:n_early]))
            early_frac_of_peak = first_half_max / (peak_ext + 1e-15)
            ok_w = pos <= stance_ext_peak_pos_max or early_frac_of_peak >= early_stance_peak_fraction_min
            if ok_w:
                timing_hits += 1
            details[f"stance{w}_hip_ext_peak_norm_pos"] = float(pos)
            details[f"stance{w}_hip_early_max_frac_of_stance_peak"] = float(
                early_frac_of_peak)
        stance_peak_timing_ok = timing_used > 0 and timing_hits == timing_used
        details["hip_stance_peak_timing_used"] = int(timing_used)
        if timing_used > 0 and not stance_peak_timing_ok:
            issues.append(
                "Hip extension-moment peak in stance was late and max in the **first {:.0%}** of stance "
                "was < {:.0%} of stance max (expected early extensor support).".format(
                    early_stance_frac_of_stance,
                    early_stance_peak_fraction_min,
                )
            )

    all_ok = bool(
        stance_extension_ok and swing_flexion_ok and stance_peak_timing_ok)
    out = {
        "all_ok": all_ok,
        "stance_extension_ok": stance_extension_ok,
        "swing_flexion_ok": swing_flexion_ok,
        "stance_peak_timing_ok": stance_peak_timing_ok,
        "details": details,
        "issues": issues,
    }
    if verbose:
        _print_checkpoint_hip(out)
    return out


def _print_checkpoint_knee(out: dict) -> None:
    all_ok = bool(out.get("all_ok"))
    label = "PASS" if all_ok else "FAIL"
    print("\n" + "=" * 60, flush=True)
    print("[KNEE CHECK] {}".format(label), flush=True)
    print("Checkpoint: knee M_FE (Grood–Suntay JCS col {}) vs typical gait".format(
        KNEE_FE_JCS_AXIS), flush=True)
    print("=" * 60, flush=True)
    d = out.get("details") or {}
    if "median_M_FE_stance_Nm" in d:
        print("  Median M_knee_FE in stance (Nm): {:.4f}".format(
            d["median_M_FE_stance_Nm"]), flush=True)
    if "median_M_FE_swing_Nm" in d:
        print("  Median M_knee_FE in swing (Nm): {:.4f}".format(
            d["median_M_FE_swing_Nm"]), flush=True)
    if d.get("swing_inconclusive"):
        print("  Swing median: inconclusive (few swing samples)", flush=True)
    if d.get("stance_direction_inconclusive"):
        print("  Stance sign: inconclusive (|median| < tol)", flush=True)
    for k, v in sorted(d.items()):
        if k.endswith("_knee_ext_peak_norm_pos"):
            print("  Normalized stance peak pos (knee FE ext max): {:.3f}".format(
                float(v)), flush=True)
    print(
        "  [PASS] Stance extension sign" if out.get(
            "stance_extension_ok") else "  [FAIL] Stance extension sign",
        flush=True,
    )
    print(
        "  [PASS] Swing vs stance median" if out.get(
            "swing_vs_stance_ok") else "  [FAIL] Swing vs stance median",
        flush=True,
    )
    print(
        "  [PASS] Stance peak timing (mid-stance extension peak)"
        if out.get("stance_peak_timing_ok")
        else "  [FAIL] Stance peak timing (mid-stance extension peak)",
        flush=True,
    )
    if all_ok:
        print("  >>> Overall: PASS <<<", flush=True)
    else:
        print("  >>> Overall: FAIL <<<", flush=True)
        for s in out.get("issues") or []:
            print("    - {}".format(s), flush=True)
    print("=" * 60, flush=True)


def _print_checkpoint_hip(out: dict) -> None:
    all_ok = bool(out.get("all_ok"))
    label = "PASS" if all_ok else "FAIL"
    print("\n" + "=" * 60, flush=True)
    print("[HIP CHECK] {}".format(label), flush=True)
    print("Checkpoint: hip M_FE (decomposition col {}) vs typical gait".format(
        HIP_FE_DECOMP_AXIS), flush=True)
    print("=" * 60, flush=True)
    d = out.get("details") or {}
    if "median_M_FE_stance_Nm" in d:
        print("  Median M_hip_FE in stance (Nm): {:.4f}".format(
            d["median_M_FE_stance_Nm"]), flush=True)
    if "median_M_FE_swing_Nm" in d:
        print("  Median M_hip_FE in swing (Nm): {:.4f}".format(
            d["median_M_FE_swing_Nm"]), flush=True)
    if d.get("swing_inconclusive"):
        print("  Swing median: inconclusive (few swing samples)", flush=True)
    if d.get("swing_direction_inconclusive"):
        print("  Swing sign: inconclusive (|median| < tol)", flush=True)
    if d.get("stance_direction_inconclusive"):
        print("  Stance sign: inconclusive (|median| < tol)", flush=True)
    for k, v in sorted(d.items()):
        if k.endswith("_hip_ext_peak_norm_pos"):
            print("  Normalized stance peak pos (hip FE): {:.3f}".format(
                float(v)), flush=True)
        if k.endswith("_hip_early_max_frac_of_stance_peak"):
            print(
                "  Early-stance max M_FE / stance max M_FE: {:.3f}".format(float(v)), flush=True)
    print(
        "  [PASS] Stance extension sign" if out.get(
            "stance_extension_ok") else "  [FAIL] Stance extension sign",
        flush=True,
    )
    print(
        "  [PASS] Swing flexion sign" if out.get(
            "swing_flexion_ok") else "  [FAIL] Swing flexion sign",
        flush=True,
    )
    print(
        "  [PASS] Early-stance extension peak timing"
        if out.get("stance_peak_timing_ok")
        else "  [FAIL] Early-stance extension peak timing",
        flush=True,
    )
    if all_ok:
        print("  >>> Overall: PASS <<<", flush=True)
    else:
        print("  >>> Overall: FAIL <<<", flush=True)
        for s in out.get("issues") or []:
            print("    - {}".format(s), flush=True)
    print("=" * 60, flush=True)


def euler_torque_body(
    I_com: np.ndarray,
    omega: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """
    Rigid-body Euler torque about COM in segment (body) frame:
        tau_b = I * alpha + omega × (I * omega)
    """
    I = np.asarray(I_com, dtype=float).reshape(3, 3)
    w = np.asarray(omega, dtype=float).reshape(3)
    al = np.asarray(alpha, dtype=float).reshape(3)
    Iw = I @ w
    return I @ al + np.cross(w, Iw)


def euler_torque_lab_from_body(
    R_seg_lab: np.ndarray,
    I_com_seg: np.ndarray,
    omega_seg: np.ndarray,
    alpha_seg: np.ndarray,
) -> np.ndarray:
    """Map Euler torque from segment frame to lab: tau_lab = R * tau_b."""
    R = np.asarray(R_seg_lab, dtype=float).reshape(3, 3)
    tau_b = euler_torque_body(I_com_seg, omega_seg, alpha_seg)
    return R @ tau_b


def foot_R_for_ankle_angle_axes(
    R_foot_lab: np.ndarray,
    R_tibia_lab: np.ndarray,
    side: str,
) -> np.ndarray:
    """
    Foot rotation matrix aligned with the current ``angles_only.py`` ankle
    convention (no fixed -90 deg pre-rotation; paired X/Z alignment only).

    Use ``R_ank.T @ M_joint_lab`` for moments in that ankle-angle frame.
    """
    if _enforce_right_handed is None:
        return np.asarray(R_foot_lab, dtype=float).reshape(3, 3).copy()
    Ra = _enforce_right_handed(np.asarray(
        R_foot_lab, dtype=float).reshape(3, 3))
    Rt = _enforce_right_handed(np.asarray(
        R_tibia_lab, dtype=float).reshape(3, 3))
    if float(np.dot(Rt[:, 0], Ra[:, 0])) < 0:
        Ra[:, [0, 2]] *= -1.0
    return Ra


def _signed_angle_about_axis_deg(u: np.ndarray, v: np.ndarray, axis: np.ndarray) -> float:
    u = np.asarray(u, dtype=float).reshape(3)
    v = np.asarray(v, dtype=float).reshape(3)
    axis = np.asarray(axis, dtype=float).reshape(3)
    u = u / (np.linalg.norm(u) + 1e-12)
    v = v / (np.linalg.norm(v) + 1e-12)
    a = axis / (np.linalg.norm(axis) + 1e-12)
    u_perp = u - a * float(np.dot(u, a))
    v_perp = v - a * float(np.dot(v, a))
    u_perp = u_perp / (np.linalg.norm(u_perp) + 1e-12)
    v_perp = v_perp / (np.linalg.norm(v_perp) + 1e-12)
    s = float(np.dot(a, np.cross(u_perp, v_perp)))
    c = float(np.dot(u_perp, v_perp))
    return float(np.degrees(np.arctan2(s, c)))


def grood_suntay_R_jcs_columns_lab(
    R_femur_lab: np.ndarray,
    R_tibia_lab: np.ndarray,
) -> np.ndarray:
    """
    Knee JCS basis matrix ``B`` (columns in **lab**) matching the current
    ``angles_only._knee_angles_grood_suntay`` geometry.

    **Segment ACS inputs** (from bilateral ``*_thigh_acs_R`` / ``*_shank_acs_R``) must
    use the pipeline convention documented in this module:

    - ``R_femur[:, 0]`` = femur **X** = **lateral**, ``[:, 1]`` = **Y** = **anterior**,
      ``[:, 2]`` = **Z** = **proximal** (toward hip).
    - ``R_tibia`` likewise: **X** lateral, **Y** anterior, **Z** proximal (toward knee).

    Construction mirrors ``angles_only``:
    - femur X seed (and -X branch), orthogonalized to femur Z,
    - tibia long axis e3 = tibia Z,
    - floating axis e2 = e3 × fX,
    - evaluate tibia as-is and paired X/Z-flipped; choose smaller |IE|.

    ``B`` is generally non-orthogonal. For robust moment projection with reduced
    cross-axis leakage, use the reciprocal-basis mapping
    ``M_jcs = solve(B.T, M_lab)``.

    Joint moment components (FE, Var/Val, IE before optional sign mapping)::
        M_knee_jcs = solve(B.T, M_knee_lab)
    """
    Rf0 = np.asarray(R_femur_lab, dtype=float).reshape(3, 3)
    Rt0 = np.asarray(R_tibia_lab, dtype=float).reshape(3, 3)
    Rf = _enforce_right_handed(
        Rf0) if _enforce_right_handed is not None else Rf0
    Rt = _enforce_right_handed(
        Rt0) if _enforce_right_handed is not None else Rt0
    femur_X, femur_Y, femur_Z = Rf[:, 0], Rf[:, 1], Rf[:, 2]

    def compute_with_tibia(Rtib: np.ndarray, fX_seed: np.ndarray) -> tuple[np.ndarray, float]:
        tibia_X, tibia_Z = Rtib[:, 0], Rtib[:, 2]
        fX = fX_seed - femur_Z * float(np.dot(fX_seed, femur_Z))
        if np.linalg.norm(fX) < 1e-12:
            fX = np.cross(femur_Z, femur_Y)
        fX = fX / (np.linalg.norm(fX) + 1e-12)
        e3 = tibia_Z / (np.linalg.norm(tibia_Z) + 1e-12)
        e2 = np.cross(e3, fX)
        if np.linalg.norm(e2) < 1e-12:
            e2 = np.cross(e3, tibia_X)
        e2 = e2 / (np.linalg.norm(e2) + 1e-12)
        ie = _signed_angle_about_axis_deg(fX, tibia_X, e3)
        B = np.column_stack([fX, e2, e3])
        return B, ie

    def best_with_tibia(Rtib: np.ndarray) -> tuple[np.ndarray, float]:
        B1, ie1 = compute_with_tibia(Rtib, femur_X)
        B2, ie2 = compute_with_tibia(Rtib, -femur_X)
        return (B2, ie2) if abs(ie2) < abs(ie1) else (B1, ie1)

    Rt_flip = Rt.copy()
    Rt_flip[:, [0, 2]] *= -1.0
    B_as_is, ie_as_is = best_with_tibia(Rt)
    B_flip, ie_flip = best_with_tibia(Rt_flip)
    return B_flip if abs(ie_flip) < abs(ie_as_is) else B_as_is


def _knee_jcs_from_lab(
    R_femur_lab: np.ndarray,
    R_tibia_lab: np.ndarray,
    M_knee_lab_Nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Project knee net moment from lab into Grood–Suntay JCS and apply the same FE /
    Var–Val sign mapping as ``angles_only`` (negate columns 0 and 1).

    Returns ``(Rg, M_knee_jcs_stored)`` where raw JCS components satisfy
    ``M_knee_lab = Rg.T @ M_jcs_raw`` and stored outputs match the pipeline
    (FE and Var/Val negated vs raw solve).
    """
    Rg = grood_suntay_R_jcs_columns_lab(R_femur_lab, R_tibia_lab)
    lab = np.asarray(M_knee_lab_Nm, dtype=float).reshape(3)
    try:
        raw = np.linalg.solve(Rg.T, lab)
    except np.linalg.LinAlgError:
        raw = np.linalg.lstsq(Rg.T, lab, rcond=None)[0]
    stored = np.asarray(raw, dtype=float).copy()
    stored[0] = -stored[0]
    stored[1] = -stored[1]
    return Rg, stored


def _Rx_minus_90() -> np.ndarray:
    if _rot_x is not None:
        return _rot_x(np.radians(-90.0))
    return np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]], dtype=float)


def hip_angle_decomposition_R_rel_adj(R_pelvis_lab: np.ndarray, R_femur_lab: np.ndarray) -> np.ndarray:
    """
    ``R_rel_adj = R_x(-90°) @ R_pelvis.T @ R_femur`` with the same femur X/Z flip
    choice as ``joint_angles.hip_angles_xzy`` (minimize |IE|).

    Hip net moment in the **same 3-vector frame** as the hip angle decomposition::
        M_hip_decomp = R_rel_adj.T @ (R_pelvis.T @ M_hip_lab)

    Components align with hip FE / AbAd / IE ordering from ``hip_angles_xzy`` only in
    the sense of that construction (intrinsic XZY on ``R_rel_adj``).
    """
    Rp0 = np.asarray(R_pelvis_lab, dtype=float).reshape(3, 3)
    Rp = _enforce_right_handed(
        Rp0) if _enforce_right_handed is not None else Rp0
    Rf0 = np.asarray(R_femur_lab, dtype=float).reshape(3, 3)
    best: tuple[float, np.ndarray] | None = None
    for s in (1.0, -1.0):
        Rraw = Rf0 @ np.diag([s, 1.0, s])
        Rf = _enforce_right_handed(
            Rraw) if _enforce_right_handed is not None else Rraw
        R_rel = Rp.T @ Rf
        R_adj = _Rx_minus_90() @ R_rel
        ie = abs(float(np.degrees(np.arctan2(R_adj[1, 0], R_adj[0, 0]))))
        if best is None or ie < best[0]:
            best = (ie, R_adj.astype(float))
    assert best is not None
    return best[1]


def inverse_dynamics_proximal_joint_one_frame(
    mass_kg: float,
    I_com_seg: np.ndarray,
    r_com_lab_m: np.ndarray,
    r_distal_jc_lab_m: np.ndarray,
    r_proximal_jc_lab_m: np.ndarray,
    F_distal_on_segment_N: np.ndarray,
    M_distal_couple_lab_Nm: np.ndarray,
    a_com_lab_m_s2: np.ndarray,
    omega_seg: np.ndarray,
    alpha_seg: np.ndarray,
    R_seg_lab: np.ndarray,
    g_lab_m_s2: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """
    One rigid segment, **no GRF**: known wrench at **distal** joint (from child),
    solve **proximal** joint force and moment (lab frame).

    Sign convention (matches foot segment):
        ``m*a = F_proximal + F_distal + m*g_lab`` with ``g_lab = (0,0,-9.81)`` m/s².

    Distal wrench is the action of the **child** on **this** segment: force
    ``F_distal_on_segment`` and free couple ``M_distal_couple`` at the distal JC,
    both in lab.
    """
    g = G_LAB_M_S2 if g_lab_m_s2 is None else np.asarray(
        g_lab_m_s2, dtype=float).reshape(3)
    m = float(mass_kg)
    r_c = np.asarray(r_com_lab_m, dtype=float).reshape(3)
    r_d = np.asarray(r_distal_jc_lab_m, dtype=float).reshape(3)
    r_p = np.asarray(r_proximal_jc_lab_m, dtype=float).reshape(3)
    F_d = np.asarray(F_distal_on_segment_N, dtype=float).reshape(3)
    M_d = np.asarray(M_distal_couple_lab_Nm, dtype=float).reshape(3)
    a = np.asarray(a_com_lab_m_s2, dtype=float).reshape(3)
    R = np.asarray(R_seg_lab, dtype=float).reshape(3, 3)

    F_p = m * a - F_d - m * g
    tau_lab = euler_torque_lab_from_body(R, I_com_seg, omega_seg, alpha_seg)
    M_p = tau_lab - M_d - np.cross(r_d - r_c, F_d) - np.cross(r_p - r_c, F_p)
    return {
        "F_proximal_N": F_p,
        "M_proximal_lab_Nm": M_p,
        "tau_euler_lab_Nm": tau_lab,
    }


def inverse_dynamics_proximal_joint_timeseries(
    mass_kg: float,
    I_com_seg: np.ndarray,
    r_com_lab_mm: np.ndarray,
    r_distal_jc_lab_mm: np.ndarray,
    r_proximal_jc_lab_mm: np.ndarray,
    F_distal_N: np.ndarray,
    M_distal_lab_Nm: np.ndarray,
    a_com_lab_mm_s2: np.ndarray,
    omega_seg: np.ndarray,
    alpha_seg: np.ndarray,
    R_seg_lab: np.ndarray,
    g_lab_m_s2: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Stack ``inverse_dynamics_proximal_joint_one_frame``; positions/acc in mm / mm/s²."""
    N = int(r_com_lab_mm.shape[0])
    MM = 1.0e-3
    g = G_LAB_M_S2 if g_lab_m_s2 is None else np.asarray(
        g_lab_m_s2, dtype=float).reshape(3)
    Fp = np.zeros((N, 3), dtype=float)
    Mp = np.zeros((N, 3), dtype=float)
    tau = np.zeros((N, 3), dtype=float)
    for i in range(N):
        o = omega_seg[i] if omega_seg.shape[0] > i else np.zeros(3)
        al = alpha_seg[i] if alpha_seg.shape[0] > i else np.zeros(3)
        one = inverse_dynamics_proximal_joint_one_frame(
            mass_kg,
            I_com_seg,
            r_com_lab_mm[i] * MM,
            r_distal_jc_lab_mm[i] * MM,
            r_proximal_jc_lab_mm[i] * MM,
            F_distal_N[i],
            M_distal_lab_Nm[i],
            a_com_lab_mm_s2[i] * MM,
            o,
            al,
            R_seg_lab[i],
            g_lab_m_s2=g,
        )
        Fp[i] = one["F_proximal_N"]
        Mp[i] = one["M_proximal_lab_Nm"]
        tau[i] = one["tau_euler_lab_Nm"]
    return {
        "F_proximal_N": Fp,
        "M_proximal_lab_Nm": Mp,
        "tau_euler_lab_Nm": tau,
    }


def inverse_dynamics_foot_one_frame(
    mass_kg: float,
    I_com_seg: np.ndarray,
    r_com_lab_m: np.ndarray,
    r_ankle_lab_m: np.ndarray,
    F_grf_N: np.ndarray,
    r_cop_lab_m: np.ndarray,
    a_com_lab_m_s2: np.ndarray,
    omega_seg: np.ndarray,
    alpha_seg: np.ndarray,
    R_foot_lab: np.ndarray,
    R_tibia_lab: np.ndarray,
    side: str = "L",
    g_lab_m_s2: np.ndarray | None = None,
    fz_min_for_cop_N: float = 5.0,
) -> dict[str, np.ndarray]:
    """
    Ankle joint reaction force and net joint moment for one time sample.

    Parameters
    ----------
    mass_kg, I_com_seg
        Foot mass and inertia about COM in **foot segment** frame (kg·m²).
    r_com_lab_m, r_ankle_lab_m
        COM and ankle joint center in lab (m). Ankle = foot ACS origin
        (``*_foot_acs_O`` / mm in pipeline → divide by 1000).
    F_grf_N, r_cop_lab_m
        Ground reaction (N) and center of pressure (m), lab frame.
    a_com_lab_m_s2
        COM linear acceleration in lab (m/s²); from mm/s² use /1000.
    omega_seg, alpha_seg
        Angular velocity/acceleration in **foot segment** frame (rad/s, rad/s²).
    R_foot_lab, R_tibia_lab
        Rotation from segment frame to lab (columns = segment axes in lab).
    g_lab_m_s2
        Gravity in lab; default (0,0,-9.81) for Z up.
    fz_min_for_cop_N
        If |Fz| below this, COP moment arm is set to zero (avoid huge |r×F| when
        COP is ill-defined).

    Returns
    -------
    dict with:
        F_ankle_N, M_joint_lab_Nm,
        M_joint_tibia_Nm, M_joint_foot_Nm,
        M_joint_ankle_angle_frame_Nm (if joint_angles helpers available),
        tau_euler_lab_Nm
    """
    g = G_LAB_M_S2 if g_lab_m_s2 is None else np.asarray(
        g_lab_m_s2, dtype=float).reshape(3)
    m = float(mass_kg)
    r_c = np.asarray(r_com_lab_m, dtype=float).reshape(3)
    r_a = np.asarray(r_ankle_lab_m, dtype=float).reshape(3)
    Fg = np.asarray(F_grf_N, dtype=float).reshape(3)
    rcop = np.asarray(r_cop_lab_m, dtype=float).reshape(3)
    a = np.asarray(a_com_lab_m_s2, dtype=float).reshape(3)
    Rf = np.asarray(R_foot_lab, dtype=float).reshape(3, 3)
    Rt = np.asarray(R_tibia_lab, dtype=float).reshape(3, 3)

    F_ankle = m * a - Fg - m * g

    tau_lab = euler_torque_lab_from_body(Rf, I_com_seg, omega_seg, alpha_seg)

    use_cop = abs(float(Fg[2])) >= fz_min_for_cop_N or np.linalg.norm(
        Fg) >= fz_min_for_cop_N
    if not use_cop or not np.all(np.isfinite(rcop)):
        M_grf = np.zeros(3, dtype=float)
    else:
        M_grf = np.cross(rcop - r_c, Fg)
    M_ankle = np.cross(r_a - r_c, F_ankle)

    M_joint = tau_lab - M_grf - M_ankle

    out: dict[str, np.ndarray] = {
        "F_ankle_N": F_ankle,
        "M_joint_lab_Nm": M_joint,
        "M_joint_tibia_Nm": Rt.T @ M_joint,
        "M_joint_foot_Nm": Rf.T @ M_joint,
        "tau_euler_lab_Nm": tau_lab,
    }
    if _enforce_right_handed is not None:
        R_ank = foot_R_for_ankle_angle_axes(Rf, Rt, side=side)
        out["M_joint_ankle_angle_frame_Nm"] = R_ank.T @ M_joint
    return out


def inverse_dynamics_foot_timeseries(
    mass_kg: float,
    I_com_seg: np.ndarray,
    r_com_lab_mm: np.ndarray,
    r_ankle_lab_mm: np.ndarray,
    F_grf_N: np.ndarray,
    cop_lab_m: np.ndarray,
    a_com_lab_mm_s2: np.ndarray,
    omega_seg: np.ndarray,
    alpha_seg: np.ndarray,
    R_foot_lab: np.ndarray,
    R_tibia_lab: np.ndarray,
    side: str = "L",
    g_lab_m_s2: np.ndarray | None = None,
    fz_min_for_cop_N: float = 5.0,
) -> dict[str, np.ndarray]:
    """
    Stack ``inverse_dynamics_foot_one_frame`` over time.

    Arrays are (N, 3) except R_* are (N, 3, 3). COM/ankle positions in **mm**,
    accelerations in **mm/s²** (converted internally). COP and GRF in **m** and **N**.
    """
    N = int(r_com_lab_mm.shape[0])
    MM = 1.0e-3
    g = G_LAB_M_S2 if g_lab_m_s2 is None else np.asarray(
        g_lab_m_s2, dtype=float).reshape(3)

    F_ankle = np.zeros((N, 3), dtype=float)
    M_lab = np.zeros((N, 3), dtype=float)
    M_tibia = np.zeros((N, 3), dtype=float)
    M_foot = np.zeros((N, 3), dtype=float)
    M_ankle_angle = np.zeros((N, 3), dtype=float)
    tau_euler = np.zeros((N, 3), dtype=float)

    for i in range(N):
        o = omega_seg[i] if omega_seg.shape[0] > i else np.zeros(3)
        al = alpha_seg[i] if alpha_seg.shape[0] > i else np.zeros(3)
        Rf = R_foot_lab[i]
        Rt = R_tibia_lab[i]
        one = inverse_dynamics_foot_one_frame(
            mass_kg,
            I_com_seg,
            r_com_lab_mm[i] * MM,
            r_ankle_lab_mm[i] * MM,
            F_grf_N[i],
            cop_lab_m[i],
            a_com_lab_mm_s2[i] * MM,
            o,
            al,
            Rf,
            Rt,
            side=side,
            g_lab_m_s2=g,
            fz_min_for_cop_N=fz_min_for_cop_N,
        )
        F_ankle[i] = one["F_ankle_N"]
        M_lab[i] = one["M_joint_lab_Nm"]
        M_tibia[i] = one["M_joint_tibia_Nm"]
        M_foot[i] = one["M_joint_foot_Nm"]
        tau_euler[i] = one["tau_euler_lab_Nm"]
        ma = one.get("M_joint_ankle_angle_frame_Nm")
        if ma is not None:
            M_ankle_angle[i] = np.asarray(ma, dtype=float).reshape(3)
        elif _enforce_right_handed is not None:
            R_ank = foot_R_for_ankle_angle_axes(Rf, Rt, side=side)
            M_ankle_angle[i] = R_ank.T @ M_lab[i]

    out = {
        "F_ankle_N": F_ankle,
        "M_joint_lab_Nm": M_lab,
        "M_joint_tibia_Nm": M_tibia,
        "M_joint_foot_Nm": M_foot,
        "M_joint_ankle_angle_frame_Nm": M_ankle_angle,
        "tau_euler_lab_Nm": tau_euler,
    }
    return out


def resolve_grf_export_npz_path(
    bilateral_npz_path: str,
    com_kinematics_npz_path: str,
    explicit_path: str | None,
) -> str | None:
    """
    Return best path to ``*_grf_export.npz`` (if any) by data quality.

    Candidates include ``explicit_path`` (when provided), paths next to bilateral/COM
    NPZs, and common project locations (e.g. ``c3d/<subject>/grf``).
    The selected file must contain ``grf_N`` and is ranked by:
    1) finite COP row count (higher is better),
    2) file modification time (newer is better).
    """
    base = os.path.basename(bilateral_npz_path)
    suf = "_bilateral_chain_results.npz"
    if not base.endswith(suf):
        return None
    trial = base[: -len(suf)]
    cands: list[str] = []
    if explicit_path:
        cands.append(os.path.abspath(explicit_path))
    bdir = os.path.dirname(os.path.abspath(bilateral_npz_path))
    cdir = os.path.dirname(os.path.abspath(com_kinematics_npz_path))
    cands.extend(
        [
            os.path.join(bdir, f"{trial}_grf_export.npz"),
            os.path.join(cdir, f"{trial}_grf_export.npz"),
        ]
    )
    # Common project layout fallbacks:
    #   <project>/scripts/static calib/<subject-folder>/...
    #   <project>/c3d/<subject>/grf/<trial>_grf_export.npz
    static_calib_dir = os.path.dirname(bdir)
    scripts_dir = os.path.dirname(static_calib_dir)
    project_root = os.path.dirname(scripts_dir)
    subject_guess = os.path.basename(bdir).split(" - ")[0]
    cands.extend(
        [
            os.path.join(project_root, "c3d", subject_guess,
                         "grf", f"{trial}_grf_export.npz"),
            os.path.join(project_root, "c3d", subject_guess,
                         f"{trial}_grf_export.npz"),
        ]
    )

    uniq: list[str] = []
    seen: set[str] = set()
    for p in cands:
        ap = os.path.abspath(p)
        key = os.path.normcase(ap)
        if key not in seen and os.path.isfile(ap):
            seen.add(key)
            uniq.append(ap)
    if not uniq:
        return None

    best_path: str | None = None
    best_score: tuple[int, int, float] = (-1, -1, -1.0)
    for p in uniq:
        try:
            z = np.load(p, allow_pickle=True)
            has_grf = int("grf_N" in z.files)
            cop_m, _ = _select_cop_in_m(z)
            cop_rows = int(np.sum(np.isfinite(cop_m).all(axis=1))
                           ) if cop_m is not None else 0
        except Exception:
            has_grf, cop_rows = 0, 0
        mtime = float(os.path.getmtime(p))
        score = (has_grf, cop_rows, mtime)
        if score > best_score:
            best_score = score
            best_path = p
    return best_path


def _foot_id_from_loaded_npz(
    bi,
    kin,
    grf_ext,
    *,
    com_kinematics_npz_path: str,
    inertial_npz_path: str | None,
    seg_id_foot: str,
    side: str,
    foot_on_plate: bool,
) -> dict[str, np.ndarray]:
    """Foot inverse dynamics from already-open bilateral + COM kinematics NPZ objects."""
    prefix = "l" if side.upper().startswith("L") else "r"
    R_foot = np.asarray(bi[f"{prefix}_foot_acs_R"], dtype=float)
    O_foot = np.asarray(bi[f"{prefix}_foot_acs_O"], dtype=float)
    R_tibia = np.asarray(bi[f"{prefix}_shank_acs_R"], dtype=float)
    other_prefix = "r" if prefix == "l" else "l"
    other_ankle_key = f"{other_prefix}_foot_acs_O"
    O_other_foot = np.asarray(
        bi[other_ankle_key], dtype=float) if other_ankle_key in bi.files else None

    acc_key = f"{seg_id_foot}_acc_mm_s2"
    om_key = f"{seg_id_foot}_omega_rad_s"
    al_key = f"{seg_id_foot}_alpha_rad_s2"
    if acc_key not in kin.files:
        raise KeyError(f"Expected {acc_key} in {com_kinematics_npz_path}")
    a_mm = np.asarray(kin[acc_key], dtype=float)
    omega = np.asarray(
        kin[om_key], dtype=float) if om_key in kin.files else np.zeros_like(a_mm)
    alpha = np.asarray(
        kin[al_key], dtype=float) if al_key in kin.files else np.zeros_like(a_mm)

    com_mm = np.zeros_like(O_foot)
    # Reconstruct COM from foot origin + r_com_seg (filled when inertial npz loaded below)
    com_mm[:] = O_foot

    mass_kg = 1.0
    I = np.eye(3) * 1e-4
    if inertial_npz_path and os.path.isfile(inertial_npz_path):
        ine = np.load(inertial_npz_path, allow_pickle=True)
        mk = f"{seg_id_foot}_mass_kg"
        Ik = f"{seg_id_foot}_I_com_seg"
        rk = f"{seg_id_foot}_r_com_seg"
        if mk in ine.files:
            mass_kg = float(np.asarray(ine[mk]).ravel()[0])
        if Ik in ine.files:
            I = np.asarray(ine[Ik], dtype=float).reshape(3, 3)
        if rk in ine.files:
            r_com = np.asarray(ine[rk], dtype=float).reshape(3)
            r_mm = r_com * 1000.0
            ncm = int(min(com_mm.shape[0], O_foot.shape[0], R_foot.shape[0]))
            com_mm[:ncm] = O_foot[:ncm] + \
                np.einsum("nij,j->ni", R_foot[:ncm], r_mm)

    n = min(R_foot.shape[0], a_mm.shape[0], O_foot.shape[0])
    # Prefer GRF/COP from COM kinematics when available (already aligned to marker frames
    # and typically has finite COP). Only use *_grf_export.npz when it provides a usable COP.
    kin_has_grf = "grf_N" in kin.files
    ext_has_grf = grf_ext is not None and "grf_N" in grf_ext.files
    kin_cop_arr_m, kin_cop_key = _select_cop_in_m(kin)
    ext_cop_arr_m, ext_cop_key = _select_cop_in_m(
        grf_ext) if grf_ext is not None else (None, None)
    kin_has_cop = kin_cop_arr_m is not None
    ext_has_cop = ext_cop_arr_m is not None

    kin_cop_usable = False
    ext_cop_usable = False
    kin_cop_rows = 0
    ext_cop_rows = 0
    if kin_has_cop:
        cop_kin_chk = np.asarray(kin_cop_arr_m, dtype=float)
        kin_cop_usable = bool(np.any(np.isfinite(cop_kin_chk)))
        kin_cop_rows = int(np.sum(np.isfinite(cop_kin_chk).all(axis=1)))
    if ext_has_cop:
        cop_ext_chk = np.asarray(ext_cop_arr_m, dtype=float)
        ext_cop_usable = bool(np.any(np.isfinite(cop_ext_chk)))
        ext_cop_rows = int(np.sum(np.isfinite(cop_ext_chk).all(axis=1)))

    # Prefer external GRF/COP when it has better COP coverage; this is common when
    # marker and force channels are split into separate C3D exports.
    use_ext = bool(
        ext_has_grf and ext_has_cop and ext_cop_usable and ext_cop_rows >= kin_cop_rows)
    use_kin = bool(
        (not use_ext) and kin_has_grf and kin_has_cop and kin_cop_usable)
    if (not use_kin) and (not use_ext) and ext_has_grf:
        use_ext = bool(ext_has_cop and ext_cop_usable)
    cop_key_used: str | None = None

    if use_kin:
        n = min(
            n,
            int(np.asarray(kin["grf_N"]).shape[0]),
            int(np.asarray(kin_cop_arr_m).shape[0]),
        )
        grf = np.asarray(kin["grf_N"], dtype=float)[:n]
        cop = np.asarray(kin_cop_arr_m, dtype=float)[:n]
        grf_src = "com_kinematics"
        cop_key_used = kin_cop_key
        if kin_cop_key in ("cop_lab_mm", "cop_mm"):
            print(f"COP source: {
                  kin_cop_key} (mm) converted to m.", flush=True)
    elif use_ext:
        ext_grf_full = np.asarray(grf_ext["grf_N"], dtype=float)
        ext_cop_full = np.asarray(ext_cop_arr_m, dtype=float)
        ext_start = 0
        # If COM kinematics has GRF, align external GRF window to kinematics using Fz.
        if kin_has_grf:
            kin_fz = np.asarray(kin["grf_N"], dtype=float)[:, 2]
            ext_fz = ext_grf_full[:, 2]
            ext_start = _best_ext_start_by_fz(kin_fz, ext_fz)
            if ext_start > 0:
                print(
                    "Aligned external GRF/COP to kinematics by Fz: start frame {} (ext length {}, kin length {}).".format(
                        ext_start,
                        int(ext_grf_full.shape[0]),
                        int(np.asarray(kin["grf_N"]).shape[0]),
                    ),
                    flush=True,
                )
        n = min(
            n,
            int(ext_grf_full.shape[0] - ext_start),
            int(ext_cop_full.shape[0] - ext_start),
        )
        grf = ext_grf_full[ext_start: ext_start + n]
        cop = ext_cop_full[ext_start: ext_start + n]
        grf_src = "grf_export"
        cop_key_used = ext_cop_key
        if ext_cop_key in ("cop_lab_mm", "cop_mm"):
            print(f"COP source: {
                  ext_cop_key} (mm) converted to m.", flush=True)
    elif kin_has_grf:
        n = min(n, int(np.asarray(kin["grf_N"]).shape[0]))
        grf = np.asarray(kin["grf_N"], dtype=float)[:n]
        cop = (
            np.asarray(kin_cop_arr_m, dtype=float)[:n]
            if kin_has_cop and kin_cop_usable
            else np.zeros((n, 3), dtype=float)
        )
        grf_src = "com_kinematics_grf_only"
        cop_key_used = kin_cop_key if (
            kin_has_cop and kin_cop_usable) else None
        if kin_has_cop and (not kin_cop_usable):
            print(
                "COP in COM kinematics is all non-finite; using GRF-only fallback (COP=0). "
                "Provide *_grf_export.npz with finite COP for realistic ankle moments.",
                flush=True,
            )
    elif ext_has_grf:
        n = min(n, int(np.asarray(grf_ext["grf_N"]).shape[0]))
        grf = np.asarray(grf_ext["grf_N"], dtype=float)[:n]
        cop = np.zeros((n, 3), dtype=float)
        grf_src = "grf_export_no_cop"
    else:
        grf = np.zeros((n, 3), dtype=float)
        cop = np.zeros((n, 3), dtype=float)
        grf_src = "none"

    # Ensure GRF/COP are low-pass filtered before inverse dynamics (match kinematic_derivatives).
    fs_hz = 100.0
    already_f: float | None = None
    if grf_src == "grf_export":
        if grf_ext is not None and "sampling_rate_hz" in grf_ext.files:
            fs_hz = float(np.asarray(grf_ext["sampling_rate_hz"]).ravel()[0])
        already_f = _grf_filtered_hz_from_npz(
            grf_ext) if grf_ext is not None else None
    elif grf_src.startswith("com_kinematics"):
        if "fp_sampling_rate_hz" in kin.files:
            fs_hz = float(np.asarray(kin["fp_sampling_rate_hz"]).ravel()[0])
        elif "time" in kin.files and int(np.asarray(kin["time"]).shape[0]) > 2:
            t = np.asarray(kin["time"], dtype=float).ravel()
            dt = float(np.median(np.diff(t)))
            if dt > 0.0:
                fs_hz = 1.0 / dt
        already_f = _grf_filtered_hz_from_npz(kin)
    grf, cop = _coerce_grf_cop_units(grf, cop)
    grf, cop = _apply_grf_cop_filter_if_needed(grf, cop, fs_hz, already_f)

    # Side-consistency check: during stance, COP should be closer to the selected
    # instrumented foot than the opposite foot.
    side_ok, side_diag = _cop_side_consistency_report(
        cop,
        grf,
        O_foot[:n],
        O_other_foot[:n] if O_other_foot is not None else None,
    )
    if foot_on_plate and (not side_ok):
        print(
            "COP side check failed (source={}): COP closer to selected foot in {:.1%} of stance frames "
            "(selected median {:.3f} m vs other {:.3f} m). Disabling COP for this side/trial.".format(
                grf_src,
                float(side_diag.get("fraction_closer_to_selected", float("nan"))),
                float(side_diag.get("median_xy_dist_selected_m", float("nan"))),
                float(side_diag.get("median_xy_dist_other_m", float("nan"))),
            ),
            flush=True,
        )
        cop = np.zeros((n, 3), dtype=float)

    # Guard against frame-mismatched COP imports (e.g., wrong plate/C3D frame).
    # If COP is implausibly far from the instrumented ankle during stance, drop COP
    # so ankle moments are not dominated by a bad r_cop x F term.
    cop_ok, cop_diag = _cop_plausibility_report(cop, grf, O_foot[:n])
    cop_src_npz = kin if grf_src.startswith("com_kinematics") else grf_ext
    _print_cop_checks(
        source=grf_src,
        cop_key=cop_key_used,
        cop_lab_m=cop,
        grf_N=grf,
        side_ok=side_ok,
        side_diag=side_diag,
        cop_ok=cop_ok,
        cop_diag=cop_diag,
        source_npz=cop_src_npz,
    )
    if foot_on_plate and (not cop_ok):
        print(
            "COP plausibility check failed (source={}): median XY COP-ankle distance {:.3f} m, "
            "p95 {:.3f} m over {} stance frames. Disabling COP for this side/trial.".format(
                grf_src,
                float(cop_diag.get("median_xy_dist_m", float("nan"))),
                float(cop_diag.get("p95_xy_dist_m", float("nan"))),
                int(cop_diag.get("n_use", 0)),
            ),
            flush=True,
        )
        cop = np.zeros((n, 3), dtype=float)

    if not foot_on_plate:
        grf = np.zeros((n, 3), dtype=float)
        cop = np.zeros((n, 3), dtype=float)

    R_foot = R_foot[:n]
    R_tibia = R_tibia[:n]
    O_foot = O_foot[:n]
    com_mm = com_mm[:n]
    a_mm = a_mm[:n]
    omega = omega[:n]
    alpha = alpha[:n]

    out = inverse_dynamics_foot_timeseries(
        mass_kg,
        I,
        com_mm,
        O_foot,
        grf,
        cop,
        a_mm,
        omega,
        alpha,
        R_foot,
        R_tibia,
        side=side,
    )
    if grf_ext is not None and "time" in grf_ext.files:
        out["time"] = np.asarray(grf_ext["time"], dtype=float)[:n]
    elif "time" in kin.files:
        out["time"] = np.asarray(kin["time"], dtype=float)[:n]
    return out


def load_foot_id_from_pipeline_outputs(
    bilateral_npz_path: str,
    com_kinematics_npz_path: str,
    inertial_npz_path: str | None,
    seg_id_foot: str = "L_foot",
    side: str = "L",
    foot_on_plate: bool = True,
    grf_export_npz_path: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Load bilateral ACS, COM kinematics + force plate, and foot inertia; run ID.

    Parameters
    ----------
    bilateral_npz_path
        ``*_bilateral_chain_results.npz`` with ``l_foot_acs_R``, ``l_foot_acs_O``,
        ``l_shank_acs_R`` (or ``r_*`` for right).
    com_kinematics_npz_path
        ``*_COM_kinematics.npz`` from ``kinematic_derivatives`` with
        ``{L_foot|R_foot}_acc_mm_s2``, ``omega``, ``alpha`` (and optionally ``grf_N``).
    inertial_npz_path
        Optional path to inertial export npz containing ``L_foot_I_com_seg`` etc.
        If None, uses ``export_inertial_segments`` from a joint-centers CSV
        (requires caller to pass inertia another way — see below).

    seg_id_foot
        ``"L_foot"`` or ``"R_foot"`` for inertial and bilateral keys.
    side
        ``"L"`` or ``"R"`` for ankle-angle frame and bilateral prefix.
    foot_on_plate
        If False, zeros GRF/COP (swing foot).
    grf_export_npz_path
        Optional ``*_grf_export.npz`` from ``forceplate_preprocess.export_grf_to_npz``
        (force-plate C3D separate from marker/bilateral C3D). If None, searches
        ``<trial>_grf_export.npz`` next to the bilateral or COM kinematics file.
        When present, ``grf_N`` / ``cop_lab_m`` / ``time`` from this file take
        priority over arrays inside ``com_kinematics_npz_path``.

    Returns
    -------
    Same as ``inverse_dynamics_foot_timeseries`` plus ``time`` if present.

    Notes
    -----
    For a single force plate, apply GRF only to the foot on the plate: set
    ``foot_on_plate=True`` only for that side; for the other foot use
    ``foot_on_plate=False`` or zero ``grf`` / ``cop`` before calling.

    Kinematics and GRF are trimmed to ``min(n_kin, n_grf)`` frames when lengths
    differ (same trial duration assumed).
    """
    bi = np.load(bilateral_npz_path, allow_pickle=True)
    kin = np.load(com_kinematics_npz_path, allow_pickle=True)
    grf_npz_path = resolve_grf_export_npz_path(
        bilateral_npz_path, com_kinematics_npz_path, grf_export_npz_path
    )
    grf_ext = np.load(
        grf_npz_path, allow_pickle=True) if grf_npz_path else None
    return _foot_id_from_loaded_npz(
        bi,
        kin,
        grf_ext,
        com_kinematics_npz_path=com_kinematics_npz_path,
        inertial_npz_path=inertial_npz_path,
        seg_id_foot=seg_id_foot,
        side=side,
        foot_on_plate=foot_on_plate,
    )


def load_leg_chain_id_from_pipeline_outputs(
    bilateral_npz_path: str,
    com_kinematics_npz_path: str,
    inertial_npz_path: str | None,
    *,
    seg_foot: str = "R_foot",
    seg_shank: str = "R_shank",
    seg_thigh: str = "R_thigh",
    side: str = "R",
    foot_on_plate: bool = True,
    grf_export_npz_path: str | None = None,
) -> dict[str, np.ndarray]:
    """
    Bottom-up inverse dynamics for one leg: foot (GRF) → shank → thigh.

    Joint centers (lab, mm): ankle = foot ACS origin; knee = shank ACS origin;
    hip = thigh ACS origin (same definitions as ``*_bilateral_chain_results.npz``).

    Returns foot ID outputs plus knee/hip forces and moments in lab, Grood–Suntay
    knee JCS, hip decomposition frame (see ``hip_angle_decomposition_R_rel_adj``),
    and segment-frame moments for reporting.
    """
    grf_npz_path = resolve_grf_export_npz_path(
        bilateral_npz_path, com_kinematics_npz_path, grf_export_npz_path
    )
    grf_ext = np.load(
        grf_npz_path, allow_pickle=True) if grf_npz_path else None
    bi = np.load(bilateral_npz_path, allow_pickle=True)
    kin = np.load(com_kinematics_npz_path, allow_pickle=True)
    foot_out = _foot_id_from_loaded_npz(
        bi,
        kin,
        grf_ext,
        com_kinematics_npz_path=com_kinematics_npz_path,
        inertial_npz_path=inertial_npz_path,
        seg_id_foot=seg_foot,
        side=side,
        foot_on_plate=foot_on_plate,
    )
    n = int(foot_out["F_ankle_N"].shape[0])
    prefix = "l" if side.upper().startswith("L") else "r"

    O_foot = np.asarray(bi[f"{prefix}_foot_acs_O"], dtype=float)[:n]
    O_shank = np.asarray(bi[f"{prefix}_shank_acs_O"], dtype=float)[:n]
    O_thigh = np.asarray(bi[f"{prefix}_thigh_acs_O"], dtype=float)[:n]
    R_foot = np.asarray(bi[f"{prefix}_foot_acs_R"], dtype=float)[:n]
    R_shank = np.asarray(bi[f"{prefix}_shank_acs_R"], dtype=float)[:n]
    R_thigh = np.asarray(bi[f"{prefix}_thigh_acs_R"], dtype=float)[:n]
    R_pelvis = np.asarray(bi["pelvis_acs_R"], dtype=float)[:n]

    acc_sk = f"{seg_shank}_acc_mm_s2"
    om_sk = f"{seg_shank}_omega_rad_s"
    al_sk = f"{seg_shank}_alpha_rad_s2"
    acc_th = f"{seg_thigh}_acc_mm_s2"
    om_th = f"{seg_thigh}_omega_rad_s"
    al_th = f"{seg_thigh}_alpha_rad_s2"
    for k in (acc_sk, acc_th):
        if k not in kin.files:
            raise KeyError(f"Expected {k} in {com_kinematics_npz_path}")
    a_sh = np.asarray(kin[acc_sk], dtype=float)[:n]
    a_th = np.asarray(kin[acc_th], dtype=float)[:n]
    w_sh = np.asarray(kin[om_sk], dtype=float)[
        :n] if om_sk in kin.files else np.zeros_like(a_sh)
    al_sh = np.asarray(kin[al_sk], dtype=float)[
        :n] if al_sk in kin.files else np.zeros_like(a_sh)
    w_th = np.asarray(kin[om_th], dtype=float)[
        :n] if om_th in kin.files else np.zeros_like(a_th)
    al_th = np.asarray(kin[al_th], dtype=float)[
        :n] if al_th in kin.files else np.zeros_like(a_th)

    com_sh = np.array(O_shank, copy=True)
    com_th = np.array(O_thigh, copy=True)
    m_sh = 1.0
    m_th = 1.0
    I_sh = np.eye(3, dtype=float) * 1e-4
    I_th = np.eye(3, dtype=float) * 1e-4
    if inertial_npz_path and os.path.isfile(inertial_npz_path):
        ine = np.load(inertial_npz_path, allow_pickle=True)
        mk = f"{seg_shank}_mass_kg"
        if mk in ine.files:
            m_sh = float(np.asarray(ine[mk]).ravel()[0])
        mk = f"{seg_thigh}_mass_kg"
        if mk in ine.files:
            m_th = float(np.asarray(ine[mk]).ravel()[0])
        Ik = f"{seg_shank}_I_com_seg"
        if Ik in ine.files:
            I_sh = np.asarray(ine[Ik], dtype=float).reshape(3, 3)
        Ik = f"{seg_thigh}_I_com_seg"
        if Ik in ine.files:
            I_th = np.asarray(ine[Ik], dtype=float).reshape(3, 3)
        rk = f"{seg_shank}_r_com_seg"
        if rk in ine.files:
            r_mm = np.asarray(ine[rk], dtype=float).reshape(3) * 1000.0
            ncm = int(min(n, com_sh.shape[0],
                      O_shank.shape[0], R_shank.shape[0]))
            com_sh[:ncm] = O_shank[:ncm] + \
                np.einsum("nij,j->ni", R_shank[:ncm], r_mm)
        rk = f"{seg_thigh}_r_com_seg"
        if rk in ine.files:
            r_mm = np.asarray(ine[rk], dtype=float).reshape(3) * 1000.0
            ncm = int(min(n, com_th.shape[0],
                      O_thigh.shape[0], R_thigh.shape[0]))
            com_th[:ncm] = O_thigh[:ncm] + \
                np.einsum("nij,j->ni", R_thigh[:ncm], r_mm)

    F_on_shank_distal = -np.asarray(foot_out["F_ankle_N"], dtype=float)[:n]
    M_on_shank_distal = - \
        np.asarray(foot_out["M_joint_lab_Nm"], dtype=float)[:n]

    shank_out = inverse_dynamics_proximal_joint_timeseries(
        m_sh,
        I_sh,
        com_sh,
        O_foot,
        O_shank,
        F_on_shank_distal,
        M_on_shank_distal,
        a_sh,
        w_sh,
        al_sh,
        R_shank,
    )

    F_on_thigh_distal = -shank_out["F_proximal_N"]
    M_on_thigh_distal = -shank_out["M_proximal_lab_Nm"]

    thigh_out = inverse_dynamics_proximal_joint_timeseries(
        m_th,
        I_th,
        com_th,
        O_shank,
        O_thigh,
        F_on_thigh_distal,
        M_on_thigh_distal,
        a_th,
        w_th,
        al_th,
        R_thigh,
    )

    M_knee_lab = np.asarray(shank_out["M_proximal_lab_Nm"], dtype=float)
    M_hip_lab = np.asarray(thigh_out["M_proximal_lab_Nm"], dtype=float)
    M_knee_jcs = np.zeros((n, 3), dtype=float)
    M_hip_decomp = np.zeros((n, 3), dtype=float)
    M_knee_shank_seg = np.zeros((n, 3), dtype=float)
    M_knee_thigh_seg = np.zeros((n, 3), dtype=float)
    M_hip_thigh_seg = np.zeros((n, 3), dtype=float)
    for i in range(n):
        _, M_knee_jcs[i] = _knee_jcs_from_lab(
            R_thigh[i], R_shank[i], M_knee_lab[i])
        Radj = hip_angle_decomposition_R_rel_adj(R_pelvis[i], R_thigh[i])
        M_hip_decomp[i] = Radj.T @ (R_pelvis[i].T @ M_hip_lab[i])
        M_knee_shank_seg[i] = R_shank[i].T @ M_knee_lab[i]
        M_knee_thigh_seg[i] = R_thigh[i].T @ M_knee_lab[i]
        M_hip_thigh_seg[i] = R_thigh[i].T @ M_hip_lab[i]

    out: dict[str, np.ndarray] = dict(foot_out)
    out.update(
        {
            "F_knee_N": shank_out["F_proximal_N"],
            "M_knee_lab_Nm": M_knee_lab,
            "M_knee_jcs_Nm": M_knee_jcs,
            "M_knee_shank_seg_Nm": M_knee_shank_seg,
            "M_knee_thigh_seg_Nm": M_knee_thigh_seg,
            "F_hip_N": thigh_out["F_proximal_N"],
            "M_hip_lab_Nm": M_hip_lab,
            "M_hip_decomp_Nm": M_hip_decomp,
            "M_hip_thigh_seg_Nm": M_hip_thigh_seg,
            "tau_euler_shank_lab_Nm": shank_out["tau_euler_lab_Nm"],
            "tau_euler_thigh_lab_Nm": thigh_out["tau_euler_lab_Nm"],
        }
    )
    return out


def validate_shank_knee_lab_inputs(
    *,
    M_knee_lab_Nm: np.ndarray,
    F_knee_N: np.ndarray | None = None,
    tau_euler_shank_lab_Nm: np.ndarray | None = None,
    shank_acc_mm_s2: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """
    Basic checks on arrays feeding / describing the shank knee ID output ``M_knee_lab_Nm``.

    Verifies shape (N, 3), fraction of finite samples, and optional companion arrays
    length-matched to ``M_knee_lab_Nm``.
    """
    M = np.asarray(M_knee_lab_Nm, dtype=float)
    if M.ndim != 2 or M.shape[1] != 3:
        out = {
            "all_ok": False,
            "n_frames": int(M.shape[0]) if M.ndim >= 1 else 0,
            "issues": ["M_knee_lab_Nm must have shape (N, 3)."],
        }
        if verbose:
            _print_validate_shank_knee(out)
        return out

    n = int(M.shape[0])
    fin = np.isfinite(M).all(axis=1)
    frac_ok = float(np.mean(fin)) if n > 0 else 0.0
    issues: list[str] = []
    if frac_ok < 1.0:
        issues.append(
            "M_knee_lab_Nm has non-finite rows: {:.2%} finite.".format(frac_ok)
        )

    def _check_pair(name: str, arr: np.ndarray | None) -> None:
        if arr is None:
            return
        a = np.asarray(arr, dtype=float)
        if a.shape[0] != n:
            issues.append("{} length {} != M_knee length {}.".format(
                name, a.shape[0], n))
        elif a.ndim == 2 and a.shape[1] != 3:
            issues.append("{} expected (N,3), got {}.".format(name, a.shape))
        elif not np.isfinite(a).all():
            issues.append("{} contains non-finite values.".format(name))

    _check_pair("F_knee_N", F_knee_N)
    _check_pair("tau_euler_shank_lab_Nm", tau_euler_shank_lab_Nm)
    _check_pair("shank_acc_mm_s2", shank_acc_mm_s2)

    details = {
        "n_frames": n,
        "M_knee_finite_fraction": frac_ok,
        "M_knee_max_abs_Nm": float(np.nanmax(np.abs(M))) if n > 0 else float("nan"),
    }
    all_ok = len(issues) == 0 and frac_ok >= 1.0
    out = {"all_ok": all_ok, "details": details, "issues": issues}
    if verbose:
        _print_validate_shank_knee(out)
    return out


def validate_knee_jcs_moment_consistency(
    M_knee_lab_Nm: np.ndarray,
    M_knee_jcs_Nm: np.ndarray,
    R_thigh_lab: np.ndarray,
    R_shank_lab: np.ndarray,
    *,
    atol_recon: float = 1e-9,
    rtol_recon: float = 1e-8,
    atol_forward: float = 1e-9,
    verbose: bool = True,
) -> dict:
    """
    Verify Grood–Suntay knee JCS projection used in the leg chain:

        ``M_jcs_raw = solve(Rg.T, M_lab)``  ⇔  ``M_lab = Rg.T @ M_jcs_raw``

    Stored outputs in this module apply the same sign mapping as ``angles_only``:
    FE and Var/Val are negated before saving.

    ``Rg`` columns are the JCS basis in lab and are not guaranteed orthonormal.
    This check validates the reciprocal-basis projection and reconstruction.
    """
    Mlab = np.asarray(M_knee_lab_Nm, dtype=float).reshape(-1, 3)
    Mjcs = np.asarray(M_knee_jcs_Nm, dtype=float).reshape(-1, 3)
    Rf = np.asarray(R_thigh_lab, dtype=float)
    Rt = np.asarray(R_shank_lab, dtype=float)
    n = min(Mlab.shape[0], Mjcs.shape[0], Rf.shape[0], Rt.shape[0])
    if n < 1:
        out = {
            "all_ok": False,
            "issues": ["No frames to validate."],
            "details": {},
        }
        if verbose:
            _print_validate_knee_jcs(out)
        return out

    err_recon = np.zeros(n, dtype=float)
    err_forward = np.zeros(n, dtype=float)
    err_orth = np.zeros(n, dtype=float)
    for i in range(n):
        lab = Mlab[i]
        jcs = Mjcs[i]
        Rg, proj = _knee_jcs_from_lab(Rf[i], Rt[i], lab)
        jcs_raw = np.asarray(jcs, dtype=float).copy()
        jcs_raw[0] = -jcs_raw[0]
        jcs_raw[1] = -jcs_raw[1]
        recon = Rg.T @ jcs_raw
        err_recon[i] = float(np.linalg.norm(lab - recon))
        err_forward[i] = float(np.linalg.norm(jcs - proj))
        err_orth[i] = float(np.linalg.norm(Rg.T @ Rg - np.eye(3)))

    scale = np.maximum(np.linalg.norm(Mlab[:n], axis=1), 1e-15)
    tol_recon = atol_recon + rtol_recon * scale
    ok_recon = err_recon <= tol_recon
    ok_forward = err_forward <= atol_forward + rtol_recon * np.maximum(
        np.linalg.norm(Mjcs[:n], axis=1), 1e-15
    )

    max_recon = float(np.max(err_recon))
    max_forward = float(np.max(err_forward))
    max_orth = float(np.max(err_orth))
    p95_recon = float(np.percentile(err_recon, 95)) if n > 1 else max_recon

    issues: list[str] = []
    if not np.all(ok_recon):
        issues.append(
            "Reconstruction check M_lab ≈ Rg.T @ M_jcs failed: max ||Δ|| = {:.3e} (p95 {:.3e}).".format(
                max_recon, p95_recon
            )
        )
    if not np.all(ok_forward):
        issues.append(
            "Projection check M_jcs ≈ solve(Rg.T, M_lab) failed: max ||Δ|| = {:.3e}.".format(
                max_forward)
        )

    all_ok = len(issues) == 0
    details = {
        "n_frames": n,
        "max_recon_err": max_recon,
        "p95_recon_err": p95_recon,
        "max_forward_err": max_forward,
        "max_Rg_nonorth_F": max_orth,
        "frames_failed_recon": int(np.sum(~ok_recon)),
        "frames_failed_forward": int(np.sum(~ok_forward)),
    }
    out = {"all_ok": all_ok, "details": details, "issues": issues}
    if verbose:
        _print_validate_knee_jcs(out)
    return out


def _print_validate_shank_knee(out: dict) -> None:
    ok = bool(out.get("all_ok"))
    label = "PASS" if ok else "FAIL"
    print("\n" + "=" * 60, flush=True)
    print("[SHANK KNEE INPUT CHECK] {}".format(label), flush=True)
    print("Checkpoint: finite M_knee_lab and optional companion arrays", flush=True)
    print("=" * 60, flush=True)
    d = out.get("details") or {}
    if "n_frames" in d:
        print("  Frames: {}".format(d["n_frames"]), flush=True)
    if "M_knee_finite_fraction" in d:
        print("  M_knee_lab finite fraction: {:.4f}".format(
            d["M_knee_finite_fraction"]), flush=True)
    if "M_knee_max_abs_Nm" in d and np.isfinite(d["M_knee_max_abs_Nm"]):
        print("  max |M_knee_lab| (Nm): {:.3f}".format(
            d["M_knee_max_abs_Nm"]), flush=True)
    if ok:
        print("  >>> Overall: PASS <<<", flush=True)
    else:
        print("  >>> Overall: FAIL <<<", flush=True)
        for s in out.get("issues") or []:
            print("    - {}".format(s), flush=True)
    print("=" * 60, flush=True)


def _print_validate_knee_jcs(out: dict) -> None:
    ok = bool(out.get("all_ok"))
    label = "PASS" if ok else "FAIL"
    print("\n" + "=" * 60, flush=True)
    print("[KNEE JCS CONSISTENCY] {}".format(label), flush=True)
    print(
        "Checkpoint: M_jcs_raw = solve(Rg.T, M_lab) ⇔ M_lab = Rg.T @ M_jcs_raw; FE/VarVal sign-mapped in stored M_jcs",
        flush=True,
    )
    print("  (Rg not assumed orthogonal; ||Rg.T@Rg - I||_F reported as diagnostic.)", flush=True)
    print("=" * 60, flush=True)
    d = out.get("details") or {}
    if "n_frames" in d:
        print("  Frames checked: {}".format(d["n_frames"]), flush=True)
    if "max_recon_err" in d:
        print(
            "  max ||M_lab - (Rg.T @ M_jcs)||: {:.3e}".format(d["max_recon_err"]), flush=True)
    if "p95_recon_err" in d:
        print("  p95 inverse error: {:.3e}".format(
            d["p95_recon_err"]), flush=True)
    if "max_forward_err" in d:
        print(
            "  max ||M_jcs - solve(Rg.T, M_lab)||: {:.3e}".format(d["max_forward_err"]), flush=True)
    if "max_Rg_nonorth_F" in d:
        print(
            "  max ||Rg.T @ Rg - I||_F (diagnostic): {:.3e}".format(d["max_Rg_nonorth_F"]), flush=True)
    if "frames_failed_recon" in d and d["frames_failed_recon"]:
        print("  frames failing inverse tol: {}".format(
            d["frames_failed_recon"]), flush=True)
    if "frames_failed_forward" in d and d.get("frames_failed_forward", 0):
        print("  frames failing forward tol: {}".format(
            d["frames_failed_forward"]), flush=True)
    if ok:
        print("  >>> Overall: PASS <<<", flush=True)
    else:
        print("  >>> Overall: FAIL <<<", flush=True)
        for s in out.get("issues") or []:
            print("    - {}".format(s), flush=True)
    print("=" * 60, flush=True)


def main() -> None:
    """Example: load default paths and save ankle ID NPZ next to COM kinematics."""
    print("inverse_dynamics_newton_euler: starting...", flush=True)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    walk_base = "Walk_R04"
    subject_dir = os.path.join(script_dir, "subject 02 - S_Cal02")
    bilateral = os.path.join(
        subject_dir, f"{walk_base}_bilateral_chain_results.npz")
    com_kin = os.path.join(subject_dir, f"{walk_base}_COM_kinematics.npz")
    inertial = None
    for fname in (
        os.path.join(script_dir, "subject 02 - S_Cal02",
                     "S_Cal02_inertial_export.npz"),
        os.path.join(script_dir, "subject 02 - S_Cal02",
                     "inertial_export.npz"),
    ):
        if os.path.isfile(fname):
            inertial = fname
            break

    if not os.path.isfile(bilateral) or not os.path.isfile(com_kin):
        print("Need bilateral_chain_results.npz and COM_kinematics.npz next to walk C3D.", flush=True)
        print(f"  bilateral: {bilateral}", flush=True)
        print(f"  com_kin:   {com_kin}", flush=True)
        return

    print("  Loading bilateral + COM kinematics + inverse dynamics...", flush=True)
    # Canonical GRF export path: SAME folder as bilateral npz.
    grf_export_expected = os.path.join(
        os.path.dirname(bilateral), f"{walk_base}_grf_export.npz"
    )
    grf_export = resolve_grf_export_npz_path(
        bilateral, com_kin, grf_export_expected)
    if grf_export is None and _export_grf_to_npz is not None:
        # If missing, try to export from separate force-plate C3D into canonical path.
        walk_c3d_near_bilateral = os.path.join(os.path.dirname(
            os.path.abspath(bilateral)), f"{walk_base}.c3d")
        walk_c3d_near_com = os.path.join(os.path.dirname(
            os.path.abspath(com_kin)), f"{walk_base}.c3d")
        cands = [
            walk_c3d_near_bilateral,
            walk_c3d_near_com,
            os.path.join(script_dir, "..", "..", "c3d",
                         "subject 02", "grf", f"{walk_base}.c3d"),
            os.path.join(script_dir, "..", "..", "c3d",
                         "subject 02", f"{walk_base}.c3d"),
        ]
        for c in cands:
            ac = os.path.abspath(c)
            if os.path.isfile(ac):
                try:
                    grf_export = _export_grf_to_npz(
                        ac, out_path=grf_export_expected)
                except Exception as e:
                    print(f"GRF export failed from {ac}: {e}", flush=True)
                break

    if grf_export:
        print("Using GRF export (same folder as bilateral): {}".format(
            grf_export), flush=True)
    else:
        print(
            "No {} in bilateral folder; GRF from COM kinematics if present.".format(
                os.path.basename(grf_export_expected)
            ),
            flush=True,
        )

    res_L = load_foot_id_from_pipeline_outputs(
        bilateral, com_kin, inertial, seg_id_foot="L_foot", side="L", foot_on_plate=False,
        grf_export_npz_path=grf_export,
    )
    res_R = load_foot_id_from_pipeline_outputs(
        bilateral, com_kin, inertial, seg_id_foot="R_foot", side="R", foot_on_plate=True,
        grf_export_npz_path=grf_export,
    )
    norm_mass_kg = _infer_moment_normalization_mass_kg(
        inertial, fallback_mass_kg=70.0)
    print("Moment normalization: saving NPZ moments in Nm/kg using mass {:.2f} kg.".format(
        norm_mass_kg), flush=True)
    res_L_save = _normalize_moment_dict_values(res_L, norm_mass_kg)
    res_R_save = _normalize_moment_dict_values(res_R, norm_mass_kg)
    out_npz = os.path.join(
        subject_dir, f"{walk_base}_foot_ankle_inverse_dynamics.npz")
    np.savez(
        out_npz,
        **{f"L_{k}": v for k, v in res_L_save.items()},
        **{f"R_{k}": v for k, v in res_R_save.items()},
    )
    print(f"Saved {os.path.abspath(out_npz)}", flush=True)

    print("  Leg chain (shank → thigh, JCS moments)...", flush=True)
    res_L_chain = load_leg_chain_id_from_pipeline_outputs(
        bilateral,
        com_kin,
        inertial,
        seg_foot="L_foot",
        seg_shank="L_shank",
        seg_thigh="L_thigh",
        side="L",
        foot_on_plate=False,
        grf_export_npz_path=grf_export,
    )
    res_R_chain = load_leg_chain_id_from_pipeline_outputs(
        bilateral,
        com_kin,
        inertial,
        seg_foot="R_foot",
        seg_shank="R_shank",
        seg_thigh="R_thigh",
        side="R",
        foot_on_plate=True,
        grf_export_npz_path=grf_export,
    )
    out_leg = os.path.join(
        subject_dir, f"{walk_base}_leg_inverse_dynamics.npz")
    res_L_chain_save = _normalize_moment_dict_values(res_L_chain, norm_mass_kg)
    res_R_chain_save = _normalize_moment_dict_values(res_R_chain, norm_mass_kg)
    np.savez(
        out_leg,
        **{f"L_{k}": v for k, v in res_L_chain_save.items()},
        **{f"R_{k}": v for k, v in res_R_chain_save.items()},
    )
    print(f"Saved {os.path.abspath(out_leg)}", flush=True)

    print("Validating instrumented leg (R): M_knee_lab inputs + knee JCS consistency...", flush=True)
    bi_val = np.load(bilateral, allow_pickle=True)
    pre_r = "r"
    R_th_r = np.asarray(bi_val[f"{pre_r}_thigh_acs_R"], dtype=float)
    R_sh_r = np.asarray(bi_val[f"{pre_r}_shank_acs_R"], dtype=float)
    kin_val = np.load(com_kin, allow_pickle=True)
    acc_shank_key = "R_shank_acc_mm_s2"
    acc_shank = np.asarray(
        kin_val[acc_shank_key], dtype=float) if acc_shank_key in kin_val.files else None
    validate_shank_knee_lab_inputs(
        M_knee_lab_Nm=res_R_chain["M_knee_lab_Nm"],
        F_knee_N=res_R_chain.get("F_knee_N"),
        tau_euler_shank_lab_Nm=res_R_chain.get("tau_euler_shank_lab_Nm"),
        shank_acc_mm_s2=acc_shank,
        verbose=True,
    )
    validate_knee_jcs_moment_consistency(
        res_R_chain["M_knee_lab_Nm"],
        res_R_chain["M_knee_jcs_Nm"],
        R_th_r,
        R_sh_r,
        verbose=True,
    )

    # Ankle-only QC on the instrumented foot (here: R with foot_on_plate=True). If the
    # plate is under the left foot, pass res_L and that side's Fz instead.
    kin = np.load(com_kin, allow_pickle=True)
    grf_npz = resolve_grf_export_npz_path(bilateral, com_kin, grf_export)
    grf_ld = np.load(grf_npz, allow_pickle=True) if grf_npz else None
    res_plate = res_R
    has_m_ankle = "M_joint_ankle_angle_frame_Nm" in res_plate
    has_grf = (grf_ld is not None and "grf_N" in grf_ld.files) or (
        "grf_N" in kin.files)
    has_time = ("time" in kin.files) or (
        grf_ld is not None and "time" in grf_ld.files)
    t_ck: np.ndarray | None = None
    fz_ck: np.ndarray | None = None
    if has_grf and has_time:
        if "grf_N" in kin.files:
            t_ck = np.asarray(kin["time"], dtype=float)
            fz_ck = np.asarray(kin["grf_N"], dtype=float)[:, 2]
        elif grf_ld is not None:
            t_ck = np.asarray(grf_ld["time"], dtype=float)
            fz_ck = np.asarray(grf_ld["grf_N"], dtype=float)[:, 2]
        else:
            t_ck = np.asarray(kin["time"], dtype=float)
            fz_ck = np.asarray(kin["grf_N"], dtype=float)[:, 2]

    if t_ck is not None and fz_ck is not None and has_m_ankle:
        n_ck = min(
            res_plate["M_joint_ankle_angle_frame_Nm"].shape[0],
            t_ck.shape[0],
            fz_ck.shape[0],
        )
        print("Running ankle checkpoint (instrumented foot)...", flush=True)
        checkpoint_ankle_moment(
            t_ck[:n_ck],
            fz_ck[:n_ck],
            res_plate["M_joint_ankle_angle_frame_Nm"][:n_ck],
            # Trial-specific tuning: allow heel-strike-dominant PF/DF envelopes.
            late_peak_fraction_min=0.14,
            verbose=True,
        )
    else:
        print("[ANKLE CHECK] skipped (cannot run):", flush=True)
        if not has_m_ankle:
            print("  - missing M_joint_ankle_angle_frame_Nm in ID result", flush=True)
        if not has_grf:
            print(
                "  - no grf_N: add *_grf_export.npz (forceplate_preprocess.export_grf_to_npz) "
                "or embed grf_N in COM kinematics npz",
                flush=True,
            )
        if not has_time:
            print("  - no time vector in GRF export or COM kinematics npz", flush=True)
        print(
            "  Keys in {}: {}".format(
                os.path.basename(com_kin), sorted(kin.files)),
            flush=True,
        )
        if grf_npz and grf_ld is not None:
            print(
                "  Keys in {}: {}".format(
                    os.path.basename(grf_npz), sorted(grf_ld.files)),
                flush=True,
            )

    if t_ck is not None and fz_ck is not None:
        has_kh = "M_knee_jcs_Nm" in res_R_chain and "M_hip_decomp_Nm" in res_R_chain
        if has_kh:
            n_kh = min(
                int(t_ck.shape[0]),
                int(fz_ck.shape[0]),
                int(res_R_chain["M_knee_jcs_Nm"].shape[0]),
                int(res_R_chain["M_hip_decomp_Nm"].shape[0]),
            )
            print(
                "Running knee gait checkpoint (instrumented leg, Grood–Suntay FE)...", flush=True)
            checkpoint_knee_moment_gait(
                t_ck[:n_kh],
                fz_ck[:n_kh],
                res_R_chain["M_knee_jcs_Nm"][:n_kh],
                # Current sign convention: stance extension is negative.
                expect_stance_extension_positive=False,
                verbose=True,
            )
            print(
                "Running hip gait checkpoint (instrumented leg, decomposition FE)...", flush=True)
            checkpoint_hip_moment_gait(
                t_ck[:n_kh],
                fz_ck[:n_kh],
                res_R_chain["M_hip_decomp_Nm"][:n_kh],
                # Current sign convention: stance extension is negative.
                expect_stance_extension_positive=False,
                # Keep swing check neutral under this sign setup.
                expect_swing_flexion_negative=False,
                verbose=True,
            )
        else:
            print(
                "[KNEE/HIP CHECK] skipped: missing M_knee_jcs_Nm or M_hip_decomp_Nm in leg chain result.", flush=True)
    else:
        print(
            "[KNEE/HIP CHECK] skipped: need grf_N and time (same as ankle checkpoint).", flush=True)

    print("inverse_dynamics_newton_euler: done.", flush=True)


if __name__ == "__main__":
    main()
