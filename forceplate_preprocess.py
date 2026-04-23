# -*- coding: utf-8 -*-
"""
forceplate_preprocess

Reads GRF (ground reaction force), COP (center of pressure), and free moment
from a C3D file. Use ``export_grf_to_npz`` to save GRF/COP/time to an NPZ for
trials where the force-plate C3D is separate from the marker C3D used for
bilateral chain / COM kinematics.

@author: lmcam


"""

import os
import re
import warnings
import numpy as np

try:
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False

try:
    import ezc3d
    HAVE_EZC3D = True
except ImportError:
    HAVE_EZC3D = False

try:
    from scipy.signal import butter as _sp_butter, filtfilt as _sp_filtfilt
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False
    _sp_butter = None
    _sp_filtfilt = None

# Match kinematic_derivatives.DEFAULT_GRF_CUTOFF_HZ for exported NPZ
DEFAULT_GRF_EXPORT_CUTOFF_HZ = 20.0
# Stance detection and COP filtering (vertical GRF threshold, N)
DEFAULT_STANCE_FZ_THRESHOLD_N = 50.0


def _uniform_time_seconds(n_frames: int, rate_hz: float) -> np.ndarray:
    """Sample times (s) for ``n_frames`` samples at ``rate_hz`` (0 Hz → unit spacing)."""
    n = int(n_frames)
    fs = float(rate_hz)
    return np.arange(n, dtype=float) / fs if fs > 0.0 else np.arange(n, dtype=float)


def _normalized_analog_labels_upper(analog_labels: list) -> list[str]:
    return [str(lab or "").upper().replace(" ", "").replace("_", "") for lab in analog_labels]


def _find_analog_channel_index(label_u: list[str], cands: list[str]) -> int | None:
    """First channel index whose normalized label contains any candidate token."""
    for token in cands:
        tok = str(token).upper().replace(" ", "").replace("_", "")
        for i, lab in enumerate(label_u):
            if tok in lab:
                return i
    return None


def _point_units_str_from_parameters(params: dict) -> str:
    """POINT:UNITS first component as a string (defaults to ``mm``)."""
    pu = params.get("POINT", {}).get("UNITS", {}).get("value", ["mm"])
    if isinstance(pu, (list, tuple, np.ndarray)) and len(pu) > 0:
        return str(pu[0][0]) if isinstance(pu[0], (list, tuple, np.ndarray)) else str(pu[0])
    return "mm"


def _mpl_finalize_figure(fig, save_path, show: bool) -> None:
    if save_path:
        fig.savefig(save_path, dpi=150)
        print("Saved:", save_path)
    if show:
        plt.show()
    plt.close(fig)


def _normalize_force_platform_corners_3x4(corners) -> np.ndarray | None:
    """
    Coerce FORCE_PLATFORM:CORNERS to shape (3, 4): rows X,Y,Z, columns = corners in lab mm.

    Returns None if the layout cannot be interpreted.
    """
    if corners is None:
        return None
    cr = np.asarray(corners, dtype=float)
    if cr.size < 12:
        return None
    cr = np.squeeze(cr)
    if cr.ndim == 3:
        if cr.shape[0] == 3 and cr.shape[1] == 4:
            cr = cr[:, :, 0]
        elif cr.shape[1] == 3 and cr.shape[2] == 4:
            cr = cr[0, :, :]
        else:
            cr = cr.reshape(-1)
    if cr.ndim == 2:
        if cr.shape == (3, 4):
            pass
        elif cr.shape == (4, 3):
            cr = cr.T
        elif cr.shape[0] == 3 and cr.shape[1] > 4:
            cr = cr[:, :4]
        elif cr.shape[1] == 3 and cr.shape[0] > 4:
            cr = cr[:4, :].T
        else:
            cr = cr.reshape(-1)
    if cr.ndim == 1:
        if cr.size < 12:
            return None
        cr = cr[:12].reshape(3, 4)
    if cr.shape != (3, 4):
        return None
    return cr


def _butter_lowpass_filtfilt(
    x: np.ndarray,
    cutoff_hz: float,
    fs_hz: float,
    order: int = 4,
) -> np.ndarray:
    """Zero-phase low-pass along time (axis=0); same convention as kinematic_derivatives."""
    if not HAVE_SCIPY or cutoff_hz <= 0.0 or fs_hz <= 0.0:
        return x
    nyq = 0.5 * fs_hz
    wn = cutoff_hz / nyq
    b, a = _sp_butter(order, wn, btype="low", analog=False)
    return _sp_filtfilt(b, a, x, axis=0)


def _filter_with_nan_support(
    x: np.ndarray,
    cutoff_hz: float,
    fs_hz: float,
    order: int = 4,
    keep_nan_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Low-pass filter while tolerating NaNs by interpolating finite samples first.

    If ``keep_nan_mask`` is provided, those samples are set back to NaN after
    filtering (useful to preserve non-contact COP gaps).
    """
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
        vf = _butter_lowpass_filtfilt(filled, cutoff_hz, fs_hz, order=order)
        arr2[:, j] = vf
    if keep_nan_mask is not None:
        km = np.asarray(keep_nan_mask, dtype=bool).reshape(-1)
        if km.shape[0] == n:
            arr2[km, :] = np.nan
    return arr2[:, 0] if squeeze else arr2


def _contiguous_true_windows(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.asarray(mask, dtype=bool).reshape(-1)
    out: list[tuple[int, int]] = []
    i = 0
    n = int(m.shape[0])
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
    stance_fz_threshold_n: float = DEFAULT_STANCE_FZ_THRESHOLD_N,
) -> np.ndarray:
    """
    Filter COP in mm only inside stance spans; keep non-contact as NaN.

    This avoids filtering raw NaN COP directly across swing/contact gaps.
    """
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
            seg[:, j] = _butter_lowpass_filtfilt(
                filled, cutoff_hz, fs_hz, order=order)
        seg[~finite_rows, :] = np.nan
        out[s: e + 1] = seg
    return out


C3D_PATH = r"C:\Users\lmcam\Documents\Grad project\c3d\subject 02\grf\Walk_R04.c3d"


def _downsample_analog_to_frames(analog_values, point_rate, analog_rate):
    """Average analog samples so there is one value per 3D frame."""
    # analog_values: (n_channels, n_analog_samples)
    ratio = analog_rate / point_rate
    samples_per_frame = int(round(ratio))
    if abs(ratio - samples_per_frame) > 1e-6:
        raise ValueError(
            "Force plate / mocap sync: ANALOG:RATE / POINT:RATE must be an integer "
            "(got {:.6f}). Check C3D rates.".format(ratio)
        )
    if samples_per_frame < 1:
        raise ValueError(
            "ANALOG:RATE ({}) must be >= POINT:RATE ({})".format(
                analog_rate, point_rate)
        )
    n_channels, n_analog = analog_values.shape[0], analog_values.shape[1]
    n_frames = n_analog // samples_per_frame
    n_keep = n_frames * samples_per_frame
    block = analog_values[:, :n_keep].reshape(
        n_channels, n_frames, samples_per_frame
    )
    return block.mean(axis=2)  # (n_channels, n_frames)


def _force_plate_pose_from_corners(corners, origin):
    """
    Build plate->lab pose from FORCE_PLATFORM CORNERS and ORIGIN.

    Returns
    -------
    (R_pl2lab, center_lab_mm, ref_lab_mm) or (None, None, None) on failure
    """
    cr = _normalize_force_platform_corners_3x4(corners)
    if cr is None:
        return None, None, None

    pts = cr.T.astype(float)  # (4,3)
    center_lab_mm = pts.mean(axis=0)

    # Use adjacent corner edges as local plate X/Y directions.
    ex = pts[1] - pts[0]
    ey = pts[3] - pts[0]
    nx = np.linalg.norm(ex)
    ny = np.linalg.norm(ey)
    if nx < 1e-9 or ny < 1e-9:
        return None, None, None
    ex = ex / nx
    # Normal from corner ordering; then orthonormalize.
    ez = np.cross(ex, ey)
    nz = np.linalg.norm(ez)
    if nz < 1e-9:
        return None, None, None
    ez = ez / nz
    ey = np.cross(ez, ex)
    ny2 = np.linalg.norm(ey)
    if ny2 < 1e-9:
        return None, None, None
    ey = ey / ny2
    R_pl2lab = np.column_stack([ex, ey, ez])  # plate coords -> lab coords

    origin_vec_plate = np.asarray(
        origin).flat[:3] if origin is not None else np.zeros(3, dtype=float)
    # ORIGIN points from moment reference point -> plate geometric center (plate coords)
    # Therefore: ref_lab = center_lab - R_pl2lab @ ORIGIN
    ref_lab_mm = center_lab_mm - (R_pl2lab @ origin_vec_plate)
    return R_pl2lab, center_lab_mm, ref_lab_mm


def _force_plate_center_from_corners(corners):
    """Return geometric center of plate corners in lab mm, or None."""
    cr = _normalize_force_platform_corners_3x4(corners)
    if cr is None:
        return None
    return np.array([cr[0].mean(), cr[1].mean(), cr[2].mean()], dtype=float)


def read_grf_com_free_moment(c3d_path):
    """
    Read GRF (ground reaction force), COP (center of pressure), and free moment
    from a C3D file. COM is interpreted as center of pressure (COP).

    Returns
    -------
    dict with keys:
        grf, cop, free_moment (Mz about plate origin), free_moment_about_cop (Mz about COP;
        standard "free moment"), force_plate_moments, point_rate, analog_rate,
        n_frames, analog_labels, c3d_path, sampling_rate_hz, units, plate_axes,
        time_sync (dict describing force plate / mocap time-base alignment)
    """
    if not HAVE_EZC3D:
        raise ImportError("ezc3d is required. Install with: pip install ezc3d")

    if not os.path.isfile(c3d_path):
        raise FileNotFoundError(f"C3D file not found: {c3d_path}")

    c3d = ezc3d.c3d(c3d_path)
    params = c3d["parameters"]
    data = c3d["data"]

    point_rate = float(params["POINT"]["RATE"]["value"][0])
    analog_rate = float(params["ANALOG"]["RATE"]["value"][0])
    analog_labels = list(params["ANALOG"]["LABELS"]["value"])
    scales = np.asarray(params["ANALOG"]["SCALE"]["value"], dtype=float)
    offsets = np.asarray(params["ANALOG"]["OFFSET"]["value"], dtype=float)

    # analogs: shape (1, n_channels, n_analog_samples) in ezc3d
    raw_analog = data["analogs"]
    if raw_analog.shape[0] == 1:
        raw_analog = raw_analog[0]  # (n_channels, n_analog_samples)
    n_channels, n_analog = raw_analog.shape
    n_frames = data["points"].shape[2]

    # Scale analog channels: stored = scale * (value - offset) => value = stored/scale + offset
    scaled = raw_analog.astype(
        float) * np.asarray(scales)[:n_channels, np.newaxis]
    scaled += np.asarray(offsets)[:n_channels, np.newaxis]

    # One value per 3D frame (average analog samples per frame)
    analog_per_frame = _downsample_analog_to_frames(
        scaled, point_rate, analog_rate
    )  # (n_channels, n_frames_analog)

    # Force plate / mocap time-base alignment (C3D stores one analog block per 3D frame)
    samples_per_frame = int(round(analog_rate / point_rate))
    n_frames_analog = analog_per_frame.shape[1]
    if n_frames_analog != n_frames:
        n_use = min(n_frames, n_frames_analog)
        warnings.warn(
            "Force plate / mocap frame count mismatch: POINT has {} frames, "
            "analog downsample has {}. Using first {} frames only.".format(
                n_frames, n_frames_analog, n_use
            ),
            UserWarning,
            stacklevel=2,
        )
        n_frames = n_use
        analog_per_frame = analog_per_frame[:, :n_use]
    time_sync = {
        "time_base": "hardware",
        "description": (
            "Force plate and mocap share the same time base: C3D stores one block of "
            "analog samples per 3D frame (ANALOG:RATE / POINT:RATE samples per frame). "
            "No software offset or resampling applied."
        ),
        "point_rate_hz": point_rate,
        "analog_rate_hz": analog_rate,
        "samples_per_frame": samples_per_frame,
        "n_frames_point": data["points"].shape[2],
        "n_frames_analog_after_downsample": n_frames_analog,
        "aligned": n_frames_analog == data["points"].shape[2],
    }

    # Map labels to channel indices robustly (important: index 0 is valid).
    label_u = _normalized_analog_labels_upper(analog_labels)

    # Do NOT use `or` on integer indices; 0 is a valid channel index.
    idx_fx = _find_analog_channel_index(
        label_u, ["FORCE.FX", "FX", "F1X", "FORCEX"])
    idx_fy = _find_analog_channel_index(
        label_u, ["FORCE.FY", "FY", "F1Y", "FORCEY"])
    idx_fz = _find_analog_channel_index(
        label_u, ["FORCE.FZ", "FZ", "F1Z", "FORCEZ"])
    idx_mx = _find_analog_channel_index(label_u, ["MOMENT.MX", "MX", "M1X"])
    idx_my = _find_analog_channel_index(label_u, ["MOMENT.MY", "MY", "M1Y"])
    idx_mz = _find_analog_channel_index(label_u, ["MOMENT.MZ", "MZ", "M1Z"])

    if idx_fx is None or idx_fy is None or idx_fz is None:
        warnings.warn(
            "Could not resolve full force channels (Fx,Fy,Fz) from ANALOG labels. "
            "Detected indices: Fx={}, Fy={}, Fz={}. Labels sample: {}".format(
                idx_fx, idx_fy, idx_fz, analog_labels[:12]
            )
        )
    if idx_mx is None or idx_my is None:
        warnings.warn(
            "Could not resolve moment channels (Mx,My) from ANALOG labels; COP may be NaN. "
            "Detected indices: Mx={}, My={}, Mz={}. Labels sample: {}".format(
                idx_mx, idx_my, idx_mz, analog_labels[:12]
            )
        )

    grf = np.zeros((n_frames, 3))
    if idx_fx is not None:
        grf[:, 0] = analog_per_frame[idx_fx, :]
    if idx_fy is not None:
        grf[:, 1] = analog_per_frame[idx_fy, :]
    if idx_fz is not None:
        grf[:, 2] = analog_per_frame[idx_fz, :]

    moments = np.zeros((n_frames, 3))
    if idx_mx is not None:
        moments[:, 0] = analog_per_frame[idx_mx, :]
    if idx_my is not None:
        moments[:, 1] = analog_per_frame[idx_my, :]
    if idx_mz is not None:
        moments[:, 2] = analog_per_frame[idx_mz, :]

    # Mz from plate = moment about plate origin (moment reference point)
    free_moment = moments[:, 2].copy(
    ) if idx_mz is not None else np.zeros(n_frames)

    # Convention: stance vertical GRF positive (flip sign from raw plate output)
    grf = -grf

    # Center of pressure (COP) from forces and moments: COPx = -My/Fz, COPy = Mx/Fz
    # These give COP in *force plate* coordinates (origin = moment reference point).
    # TYPE-2: ORIGIN = vector from moment ref to geometric center; CORNERS = lab-frame corners.
    cop = np.full((n_frames, 3), np.nan)
    fz = grf[:, 2]
    valid = np.abs(fz) > 1e-6
    fp = params.get("FORCE_PLATFORM", {})
    origin = fp.get("ORIGIN", {}).get("value")
    if idx_mx is not None and idx_my is not None:
        cop[valid, 0] = -moments[valid, 1] / \
            fz[valid]  # COPx = -My/Fz (plate coords)
        cop[valid, 1] = moments[valid, 0] / \
            fz[valid]   # COPy = Mx/Fz (plate coords)
    # COP is defined on the plate surface in plate-local coordinates.
    # Keep local z = 0 here; global z comes from the plate pose transform.
    cop[valid, 2] = 0.0

    # Verify free moment: recompute moment about COP and compare to reported Mz
    # Mz_about_origin = Mz_about_COP + (r_cop × F)_z  =>  Mz_about_COP = Mz_origin - (cop_x*Fy - cop_y*Fx)
    # cop here is in plate coords (mm), grf in N, moments in Nmm => (cop_x*Fy - cop_y*Fx) in Nmm
    cop_plate_mm = cop.copy()  # still in plate coords, mm
    free_moment_about_cop_nmm = np.full(n_frames, np.nan)
    if idx_mz is not None:
        mz_origin_nmm = moments[valid, 2].copy()
        term = cop_plate_mm[valid, 0] * grf[valid, 1] - \
            cop_plate_mm[valid, 1] * grf[valid, 0]
        free_moment_about_cop_nmm[valid] = mz_origin_nmm - term

    # Convert COP from force-plate coords to lab frame using CORNERS (lab) and ORIGIN.
    # Apply full plate->lab rigid transform (rotation + translation), not translation only.
    corners = fp.get("CORNERS", {}).get("value")
    R_pl2lab, center_lab_mm, ref_lab_mm = _force_plate_pose_from_corners(
        corners, origin)
    plate_origin_lab_mm = None
    if (R_pl2lab is not None) and np.any(valid):
        plate_origin_lab_mm = ref_lab_mm
        cop_lab = np.full_like(cop, np.nan, dtype=float)
        cop_lab[valid, :] = (
            R_pl2lab @ cop_plate_mm[valid, :].T).T + ref_lab_mm
        cop = cop_lab
    elif np.any(valid):
        # Fallback: still apply global translation using plate center + ORIGIN assumption.
        center_fallback = _force_plate_center_from_corners(corners)
        origin_vec = np.asarray(
            origin).flat[:3] if origin is not None else np.zeros(3, dtype=float)
        if center_fallback is not None:
            ref_lab_mm = center_fallback - origin_vec
        else:
            ref_lab_mm = -origin_vec
        plate_origin_lab_mm = ref_lab_mm
        cop[valid, 0] += ref_lab_mm[0]
        cop[valid, 1] += ref_lab_mm[1]
        cop[valid, 2] += ref_lab_mm[2]

    # Keep COP in mm internally; derive meter copy for legacy plotting/consumers.
    cop_mm = cop.copy()
    cop = cop_mm / 1000.0
    moments = moments / 1000.0
    free_moment = free_moment / 1000.0
    free_moment_about_cop = free_moment_about_cop_nmm / 1000.0  # Nm

    # Sampling rate
    # 3D/GRF frame rate (analog downsampled to this)
    sampling_rate_hz = point_rate

    # Units (from C3D parameters)
    point_units = [_point_units_str_from_parameters(params)]
    analog_units_list = params.get("ANALOG", {}).get(
        "UNITS", {}).get("value", [])
    if not analog_units_list:
        force_units, moment_units = "N", "Nmm"
    else:
        # First 3 channels typically force (N), next 3 moment (Nmm or Nm)
        u = [str(x) if hasattr(x, "strip") else str(x[0])
             if np.size(x) else "?" for x in analog_units_list]
        force_units = u[0] if len(u) > 0 else "N"
        moment_units = u[3] if len(u) > 3 else "Nmm"
    units = {
        "point": "m",
        "point_raw": str(point_units[0]) if point_units else "mm",
        "force": force_units,
        "moment": "Nm",
    }

    # Plate coordinate axes from FORCE_PLATFORM CORNERS (3 x 4: X,Y,Z × 4 corners)
    plate_axes = {"origin_mm": None, "x_range_mm": None,
                  "y_range_mm": None, "z_mm": None, "description": ""}
    corners = fp.get("CORNERS", {}).get("value")
    cr_axes = _normalize_force_platform_corners_3x4(corners)
    if cr_axes is not None:
        x_vals, y_vals, z_vals = cr_axes[0], cr_axes[1], cr_axes[2]
        plate_axes["origin_mm"] = np.asarray(
            origin).flat[:3].tolist() if origin is not None else [0, 0, 0]
        plate_axes["x_range_mm"] = [
            float(np.min(x_vals)), float(np.max(x_vals))]
        plate_axes["y_range_mm"] = [
            float(np.min(y_vals)), float(np.max(y_vals))]
        plate_axes["z_mm"] = float(np.mean(z_vals))
        if (R_pl2lab is not None) and (center_lab_mm is not None):
            plate_axes["description"] = (
                "Plate pose from CORNERS: full plate->lab transform applied for COP "
                "(rotation + translation); center Z = {:.2f} mm.".format(
                    float(center_lab_mm[2]))
            )
        else:
            plate_axes["description"] = (
                "Plate in lab frame: X along {:.2f}–{:.2f} mm, Y along {:.2f}–{:.2f} mm, Z = {:.2f} mm (normal up)."
                .format(plate_axes["x_range_mm"][0], plate_axes["x_range_mm"][1],
                        plate_axes["y_range_mm"][0], plate_axes["y_range_mm"][1], plate_axes["z_mm"])
            )

    transform_qc = {
        "has_pose": bool(R_pl2lab is not None and plate_origin_lab_mm is not None),
        "n_cop_finite_rows": int(np.sum(np.isfinite(cop_mm).all(axis=1))),
    }
    if R_pl2lab is not None and plate_origin_lab_mm is not None:
        finite_rows = np.isfinite(cop_mm).all(axis=1)
        if np.any(finite_rows):
            local = (
                R_pl2lab.T @ (cop_mm[finite_rows] - plate_origin_lab_mm).T).T
            z_res = local[:, 2]
            transform_qc.update(
                {
                    "cop_local_z_mm_median_abs": float(np.nanmedian(np.abs(z_res))),
                    "cop_local_z_mm_p95_abs": float(np.nanpercentile(np.abs(z_res), 95)),
                    "cop_local_z_mm_max_abs": float(np.nanmax(np.abs(z_res))),
                }
            )

    out = {
        "grf": grf,
        "cop": cop,
        "cop_mm": cop_mm,
        "free_moment": free_moment,
        "free_moment_about_cop": free_moment_about_cop,
        "force_plate_moments": moments,
        "point_rate": point_rate,
        "analog_rate": analog_rate,
        "n_frames": n_frames,
        "analog_labels": analog_labels,
        "c3d_path": c3d_path,
        "sampling_rate_hz": sampling_rate_hz,
        "units": units,
        "plate_axes": plate_axes,
        "transform_qc": transform_qc,
        "plate_origin_lab_mm": plate_origin_lab_mm,
        "time_sync": time_sync,
    }
    return out


def export_grf_to_npz(
    c3d_path: str,
    out_path: str | None = None,
    grf_cutoff_hz: float = DEFAULT_GRF_EXPORT_CUTOFF_HZ,
) -> str:
    """
    Load GRF/COP from a C3D via ``read_grf_com_free_moment`` and save a compact NPZ
    for downstream use (e.g. ``inverse_dynamics_newton_euler`` when the plate C3D is
    not the same file as the marker trial used for ``*_bilateral_chain_results.npz``).

    **Analog GRF/COP are low-pass filtered** (default 20 Hz, same as
    ``kinematic_derivatives.load_forceplate_aligned``) before save so inverse
    dynamics uses smoothed loads. Set ``grf_cutoff_hz=0`` to save raw (not recommended).

    Saved arrays (aligned with ``kinematic_derivatives`` COM NPZ naming where possible):

    - ``grf_N`` : (N, 3)
    - ``cop_lab_m`` : (N, 3)  COP in lab frame (m)
    - ``time`` : (N,) seconds, uniform sampling at ``sampling_rate_hz``
    - ``free_moment_Nm``, ``free_moment_about_cop_Nm`` : (N,) scalar moments
    - ``sampling_rate_hz``, ``n_frames``, ``source_c3d`` (str path)
    - ``fp_grf_filtered_hz`` : scalar, cutoff used (0 if raw)

    Parameters
    ----------
    c3d_path
        Path to the C3D that contains analog force-plate data.
    out_path
        Output ``*_grf_export.npz``. Default: same folder as C3D,
        ``<basename>_grf_export.npz``.
    grf_cutoff_hz
        Butterworth low-pass (Hz) on GRF, COP, and scalar moments; 0 disables.

    Returns
    -------
    str
        Absolute path to the written NPZ.
    """
    result = read_grf_com_free_moment(c3d_path)
    n = int(result["n_frames"])
    fs = float(result["sampling_rate_hz"])
    time_s = _uniform_time_seconds(n, fs)
    grf = np.asarray(result["grf"], dtype=float)
    cop_mm = np.asarray(result["cop_mm"], dtype=float) if "cop_mm" in result else (
        np.asarray(result["cop"], dtype=float) * 1000.0)
    fm = np.asarray(result["free_moment"], dtype=float).ravel()
    fm_cop = np.asarray(result["free_moment_about_cop"], dtype=float).ravel()
    filt_hz = float(grf_cutoff_hz)
    if filt_hz > 0.0 and fs > 0.0:
        grf = _butter_lowpass_filtfilt(grf, filt_hz, fs)
        cop_mm = _filter_cop_mm_stance_spans(cop_mm, grf[:, 2], filt_hz, fs)
        fm = _butter_lowpass_filtfilt(fm.reshape(-1, 1), filt_hz, fs).ravel()
        fm_cop = _filter_with_nan_support(
            fm_cop.reshape(-1, 1), filt_hz, fs).ravel()
    else:
        filt_hz = 0.0
    cop = cop_mm * 1.0e-3
    if out_path is None:
        base = os.path.splitext(os.path.basename(c3d_path))[0]
        out_path = os.path.join(os.path.dirname(
            os.path.abspath(c3d_path)), base + "_grf_export.npz")
    out_path = os.path.abspath(out_path)
    _od = os.path.dirname(out_path)
    if _od:
        os.makedirs(_od, exist_ok=True)
    np.savez(
        out_path,
        grf_N=grf,
        cop_lab_m=cop,
        cop_lab_mm=cop_mm,
        time=time_s,
        free_moment_Nm=fm,
        free_moment_about_cop_Nm=fm_cop,
        sampling_rate_hz=np.array([fs], dtype=float),
        n_frames=np.array([n], dtype=np.int64),
        source_c3d=np.array(c3d_path, dtype=object),
        export_version=np.array([2], dtype=np.int32),
        fp_grf_filtered_hz=np.array([filt_hz], dtype=float),
    )
    print(
        "Exported GRF/COP NPZ (filtered @ {:.1f} Hz unless cutoff 0): {}".format(
            filt_hz, out_path
        )
    )
    return out_path


def _stance_windows(fz, threshold_n: float | None = None):
    """Return list of (start_idx, end_exclusive) for contiguous stance (Fz > threshold)."""
    if threshold_n is None:
        threshold_n = DEFAULT_STANCE_FZ_THRESHOLD_N
    above = np.asarray(fz, dtype=float).reshape(-1) > float(threshold_n)
    if not np.any(above):
        return []
    return [(s, e + 1) for s, e in _contiguous_true_windows(above)]


def plot_fz_with_stance(result, threshold_n=DEFAULT_STANCE_FZ_THRESHOLD_N, save_path=None, show=True):
    """Plot Fz (N) vs time with stance windows shaded."""
    if not HAVE_MPL:
        print("matplotlib not available; skip Fz plot.")
        return
    fz = result["grf"][:, 2]
    rate = result["sampling_rate_hz"]
    n_frames = result["n_frames"]
    time_s = _uniform_time_seconds(n_frames, rate)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    windows = _stance_windows(fz, threshold_n)
    for start, end in windows:
        ax.axvspan(time_s[start], time_s[end - 1], alpha=0.25, color="green")
    ax.plot(time_s, fz, color="C0", linewidth=1, label="Fz (N)")
    if windows:
        ax.axvspan(np.nan, np.nan, alpha=0.25, color="green",
                   label="Stance (Fz > {} N)".format(threshold_n))
    ax.legend(loc="upper right")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fz (N)")
    ax.set_title(
        "Vertical GRF with stance windows (Fz > {} N)".format(threshold_n))
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="k", linewidth=0.5)
    fig.tight_layout()
    _mpl_finalize_figure(fig, save_path, show)


def plot_cop_xy_stance(result, threshold_n=DEFAULT_STANCE_FZ_THRESHOLD_N, save_path=None, show=True):
    """Scatter plot of COP X vs Y during stance (Fz > threshold), colored by GRF z. Lab frame, meters."""
    if not HAVE_MPL:
        print("matplotlib not available; skip COP XY plot.")
        return
    fz = result["grf"][:, 2]
    cop = result["cop"]
    stance = fz > threshold_n
    cop_x = cop[stance, 0]
    cop_y = cop[stance, 1]
    fz_stance = fz[stance]
    valid = np.isfinite(cop_x) & np.isfinite(cop_y)
    cop_x, cop_y = cop_x[valid], cop_y[valid]
    fz_stance = fz_stance[valid]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sc = ax.scatter(cop_x, cop_y, s=8, alpha=0.7, c=fz_stance,
                    cmap="viridis", edgecolors="none")
    cbar = fig.colorbar(sc, ax=ax, label="Fz (N)")
    ax.set_xlabel("COP X (m)")
    ax.set_ylabel("COP Y (m)")
    ax.set_title(
        "Center of pressure during stance (Fz > {} N)".format(threshold_n))
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _mpl_finalize_figure(fig, save_path, show)


# Right foot marker names to plot during stance
RIGHT_FOOT_MARKERS = ["R_Calc", "R_Ank_Lat", "R_Midfoot_Sup", "R_Midfoot_Lat"]
# When C3D has generic labels ('*0', '*1', ...), use these indices for the 4 markers above (order: R_Calc, R_Ank_Lat, R_Midfoot_Sup, R_Midfoot_Lat)
RIGHT_FOOT_MARKER_INDICES = [0, 1, 2, 3]


def _marker_index(labels, name):
    """Return index of marker whose label equals or ends with name (e.g. 'Subject:R_Calc' -> R_Calc)."""
    name = name.strip()
    for i, lab in enumerate(labels):
        if lab is None:
            continue
        lab = lab.strip() if isinstance(lab, str) else str(lab)
        if lab == name or lab.endswith(":" + name) or lab.split(":")[-1].strip() == name:
            return i
    return None


def _labels_are_generic(labels):
    """True if all labels look like *0, *1, etc. (no descriptive names)."""
    if not labels:
        return True
    pattern = re.compile(r"^\*\d+$")
    return all(pattern.match(str(lab).strip()) for lab in labels)


def _point_unit_scale_to_m(point_unit_raw: str, xyz_sample: np.ndarray | None = None) -> tuple[float, str]:
    """
    Return multiplier to convert marker coordinates to meters and normalized unit label.
    Falls back to a simple magnitude heuristic if unit text is unknown.
    """
    u = (point_unit_raw or "").strip().lower()
    if u in ("mm", "millimeter", "millimeters"):
        return 0.001, "mm"
    if u in ("m", "meter", "meters"):
        return 1.0, "m"
    if u in ("cm", "centimeter", "centimeters"):
        return 0.01, "cm"
    # Heuristic fallback: if magnitudes look like hundreds/thousands -> mm.
    if xyz_sample is not None:
        finite = xyz_sample[np.isfinite(xyz_sample)]
        if finite.size > 0:
            med_abs = float(np.median(np.abs(finite)))
            if med_abs > 10.0:
                return 0.001, "mm?"
    return 1.0, (point_unit_raw or "unknown")


def plot_right_foot_markers_stance(c3d_path, result, marker_names=None, marker_indices=None,
                                   threshold_n=DEFAULT_STANCE_FZ_THRESHOLD_N, point_units=None, save_path=None, auto_open=True):
    """Interactive Plotly 3D plot of right foot markers + COP during stance.
    If C3D has generic labels ('*0', '*1', ...), pass marker_indices (list of 4 indices)
    or set RIGHT_FOOT_MARKER_INDICES to match your channel order."""
    if not HAVE_EZC3D:
        print("ezc3d not available; skip right foot markers plot.")
        return
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:
        print("plotly not available; install with: pip install plotly")
        return
    if marker_names is None:
        marker_names = RIGHT_FOOT_MARKERS
    if marker_indices is None:
        marker_indices = RIGHT_FOOT_MARKER_INDICES

    c3d = ezc3d.c3d(c3d_path)
    labels = list(c3d["parameters"]["POINT"]["LABELS"]["value"])
    pts = c3d["data"]["points"]  # (4, n_markers, n_frames)
    xyz = pts[:3, :, :]  # (3, n_markers, n_frames), typically mm
    n_markers = xyz.shape[1]
    if point_units is None:
        point_units = _point_units_str_from_parameters(c3d["parameters"])

    n_marker_frames = xyz.shape[2]
    fz = result["grf"][: n_marker_frames, 2]
    stance = fz > threshold_n

    to_m, unit_detected = _point_unit_scale_to_m(str(point_units), xyz)
    unit_label = "m"
    print("Marker units detected: {} -> plotted in meters (x{:.6f})".format(unit_detected, to_m))

    # Resolve (name, index) for each marker: by label first, then by marker_indices if labels are generic
    name_to_idx = []
    for k, name in enumerate(marker_names):
        idx = _marker_index(labels, name)
        if idx is not None:
            name_to_idx.append((name, idx))
        elif _labels_are_generic(labels) and k < len(marker_indices):
            idx = marker_indices[k]
            if 0 <= idx < n_markers:
                name_to_idx.append((name, idx))
            else:
                print("  Marker '{}' index {} out of range [0, {}).".format(
                    name, idx, n_markers))
        else:
            if not _labels_are_generic(labels):
                print("  Marker '{}' not found in C3D (labels: {}).".format(
                    name, labels[:5]))

    if not name_to_idx:
        print("  No right foot markers to plot.")
        return

    cop = result["cop"]  # (n_frames, 3) in m
    cop_stance = cop[: n_marker_frames][stance]  # same frames as markers
    fz_stance = fz[stance]  # GRF z (N) at each stance frame

    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for k, (name, idx) in enumerate(name_to_idx):
        x = xyz[0, idx, stance] * to_m
        y = xyz[1, idx, stance] * to_m
        z = xyz[2, idx, stance] * to_m
        valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if not np.any(valid):
            continue
        x, y, z = x[valid], y[valid], z[valid]
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode="markers",
            marker=dict(size=3, color=colors[k % len(colors)], opacity=0.8),
            name=name
        ))
    # Overlay COP during stance (already in m)
    cx = cop_stance[:, 0]
    cy = cop_stance[:, 1]
    cz = cop_stance[:, 2]
    cop_valid = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(cz)
    if np.any(cop_valid):
        fig.add_trace(go.Scatter3d(
            x=cx[cop_valid], y=cy[cop_valid], z=cz[cop_valid],
            mode="markers",
            marker=dict(size=4, color="red", opacity=0.95),
            name="COP"
        ))
    # GRF z as vertical arrows at COP (subsample for clarity)
    step = max(1, np.sum(cop_valid) // 25)
    idx_arrow = np.where(cop_valid)[0][::step]
    if len(idx_arrow) > 0:
        x0 = cx[idx_arrow]
        y0 = cy[idx_arrow]
        z0 = cz[idx_arrow]
        fz_arrow = fz_stance[idx_arrow]
        # Scale arrow length: ~0.2 m per 1000 N so arrows are visible in scene
        scale_fz = 0.0002  # m/N
        for xa, ya, za, fza in zip(x0, y0, z0, fz_arrow):
            z1 = za + fza * scale_fz
            fig.add_trace(go.Scatter3d(
                x=[xa, xa], y=[ya, ya], z=[za, z1],
                mode="lines",
                line=dict(color="green", width=4),
                name="Fz (N)",
                showlegend=False
            ))
        # Add one legend item for Fz.
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="lines",
            line=dict(color="green", width=4),
            name="Fz (N)"
        ))

    fig.update_layout(
        title="Right foot markers and COP during stance (Fz > {} N)".format(
            threshold_n),
        scene=dict(
            xaxis_title="X ({})".format(unit_label),
            yaxis_title="Y ({})".format(unit_label),
            zaxis_title="Z ({})".format(unit_label),
            aspectmode="data",
        ),
        legend=dict(x=0.01, y=0.99),
        margin=dict(l=10, r=10, b=10, t=40),
    )

    if save_path is None:
        base = os.path.splitext(os.path.basename(c3d_path))[0]
        save_path = os.path.join(os.path.dirname(
            c3d_path), "{}_right_foot_cop_stance.html".format(base))
    pio.write_html(fig, file=save_path, auto_open=auto_open)
    print("Saved interactive plot:", os.path.abspath(save_path))


def main():
    path = C3D_PATH
    if not os.path.isfile(path):
        path = os.path.join(
            os.path.dirname(__file__),
            "..", "..",
            "c3d", "subject 02", "grf", "Walk_R04.c3d",
        )
        path = os.path.abspath(path)
    result = read_grf_com_free_moment(path)
    print("C3D:", result["c3d_path"])
    print("Frames:", result["n_frames"])
    print("Sampling rate: {:.0f} Hz (3D/GRF frame rate); analog raw rate: {:.0f} Hz".format(
        result["sampling_rate_hz"], result["analog_rate"]))
    ts = result["time_sync"]
    print("Time sync (force plate / mocap): {} (aligned: {})".format(
        ts["time_base"], ts["aligned"]))
    print("  ", ts["description"])
    print("Units: point/COP = {}, force = {}, moment = {}".format(
        result["units"]["point"], result["units"]["force"], result["units"]["moment"]))
    print("Plate coordinate axes:",
          result["plate_axes"]["description"] or "(not available)")
    print("GRF shape:", result["grf"].shape, "COP shape:",
          result["cop"].shape, "Free moment shape:", result["free_moment"].shape)
    print("Analog labels:", result["analog_labels"])
    export_grf_to_npz(path)
    print("GRF sample (frame 100):", result["grf"][100])
    print("Free moment sample (frame 100):", result["free_moment"][100])
    # Stance sample (frame where |Fz| is large)
    fz = result["grf"][:, 2]
    stance_idx = np.where(np.abs(fz) > DEFAULT_STANCE_FZ_THRESHOLD_N)[0]
    if len(stance_idx) > 0:
        i = stance_idx[len(stance_idx) // 2]
        print("Stance sample (frame {}): GRF={}, COP={}, Mz={:.4f}".format(
            i, result["grf"][i], result["cop"][i], result["free_moment"][i]))
        # Free moment verification: reported Mz vs moment about COP
        mz_reported = result["free_moment"][i]
        mz_about_cop = result["free_moment_about_cop"][i]
        if np.isfinite(mz_about_cop):
            diff = abs(mz_reported - mz_about_cop)
            print("  Free moment: reported Mz = {:.4f} Nm, recomputed Mz about COP = {:.4f} Nm (diff = {:.6f})".format(
                mz_reported, mz_about_cop, diff))
            if diff < 1e-6:
                print("  => C3D Mz is moment about COP (free moment).")
            else:
                print(
                    "  => C3D Mz is moment about plate origin; free moment about COP = recomputed value.")

    # Plot Fz over time with stance windows shaded
    plot_fz_with_stance(result, save_path=None, show=True)
    # COP X–Y scatter during stance
    plot_cop_xy_stance(result, save_path=None, show=True)
    # Right foot markers during stance
    plot_right_foot_markers_stance(result["c3d_path"], result, marker_names=RIGHT_FOOT_MARKERS,
                                   save_path=None, auto_open=True)


if __name__ == "__main__":
    main()
