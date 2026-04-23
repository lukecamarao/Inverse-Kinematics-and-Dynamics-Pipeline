"""
Pelvis anatomical coordinate system (ACS) per ISB recommendations using SymPy.

Inputs are 3D points for LASI, RASI, LPSI, RPSI. The ACS is defined as:
- Origin at ASIS midpoint (midpoint of LASI and RASI)
- X axis pointing to subject's right (from LASI to RASI)
- Y axis pointing anterior
- Z axis pointing superior

Implementation outline:
1) Compute origin O = midpoint(LASI, RASI)
2) Compute X_right = normalize(RASI - LASI)
3) Compute PSIS midpoint P = midpoint(LPSI, RPSI)
4) Compute anterior direction seed: Y_seed = normalize(O - P)
5) Compute Z_up = normalize(X_right × Y_seed)
6) Recompute Y_anterior = Z_up × X_right (ensures orthonormal right-handed frame)

Returns rotation matrix R such that its columns are the pelvis axes
expressed in the lab frame: R = [X_right, Y_anterior, Z_up].
"""

from __future__ import annotations

import os
from typing import Iterable, Tuple

from sympy import Matrix
import numpy as np
import ezc3d
import plotly.graph_objects as go
import plotly.io as pio


def _to_vector(point: Iterable[float]) -> Matrix:
    """Convert a 3D iterable into a 3x1 SymPy Matrix."""
    p = Matrix(point)
    if p.shape == (3, 1):
        return p
    if p.shape == (1, 3):
        return p.T
    if p.shape == (3, 3) and p.cols == 1:
        return p
    if p.shape[0] == 3 and p.cols == 1:
        return p
    # Fallback: flatten then reshape
    flat = Matrix(list(p)).reshape(len(p), 1)
    if flat.shape != (3, 1):
        raise ValueError("Point must have exactly 3 elements")
    return flat


def _normalize(v: Matrix) -> Matrix:
    """Return unit vector in the direction of v; raises if zero length."""
    v = Matrix(v)
    norm_val = v.norm()
    if norm_val == 0:
        raise ValueError("Cannot normalize zero-length vector")
    return v / norm_val


def compute_pelvis_acs(
    lasi: Iterable[float],
    rasi: Iterable[float],
    lpsi: Iterable[float],
    rpsi: Iterable[float],
) -> Tuple[Matrix, Matrix, Matrix, Matrix, Matrix]:
    """
    Compute pelvis ACS from ASIS/PSIS landmarks per ISB.

    Parameters
    - lasi, rasi, lpsi, rpsi: iterables of 3 floats (x, y, z) in lab frame.

    Returns
    - origin: 3x1 Matrix, pelvis origin at ASIS midpoint
    - R: 3x3 Matrix, rotation with columns [X_right, Y_anterior, Z_up]
    - X_right: 3x1 unit vector pointing to subject's right
    - Y_anterior: 3x1 unit vector pointing anterior
    - Z_up: 3x1 unit vector pointing superior
    """
    lasi_v = _to_vector(lasi)
    rasi_v = _to_vector(rasi)
    lpsi_v = _to_vector(lpsi)
    rpsi_v = _to_vector(rpsi)

    origin = (lasi_v + rasi_v) / 2
    psis_mid = (lpsi_v + rpsi_v) / 2

    x_right = _normalize(rasi_v - lasi_v)
    y_seed = _normalize(origin - psis_mid)  # roughly anterior
    # Enforce Z to be superior (up) while keeping X to the right and right-handedness
    up = Matrix([0.0, 0.0, 1.0])
    z_pre = x_right.cross(y_seed)
    try:
        z_pre_dot_up = float(z_pre.dot(up))
    except Exception:
        z_pre_dot_up = float(z_pre.dot(up).evalf())
    if z_pre_dot_up < 0.0:
        # Flip anterior seed so that X × Y points upward
        y_seed = -y_seed
    z_up = _normalize(x_right.cross(y_seed))
    # Recompute Y to ensure orthonormal right-handed basis and anterior direction
    y_anterior = _normalize(z_up.cross(x_right))

    R = Matrix.hstack(x_right, y_anterior, z_up)
    return origin, R, x_right, y_anterior, z_up


def is_rotation_matrix(R: Matrix, tol: float = 1e-8) -> bool:
    """Check if R is orthonormal with det +1 within tolerance."""
    if R.shape != (3, 3):
        return False
    should_be_identity = (R.T * R).evalf()
    identity = Matrix.eye(3)
    if (should_be_identity - identity).norm() > tol:
        return False
    det_val = R.det().evalf()
    return abs(det_val - 1.0) <= tol


__all__ = [
    "compute_pelvis_acs",
    "is_rotation_matrix",
    "harrington_hip_centers",
    "femur_acs_from_landmarks",
    "tibia_acs_from_landmarks",
    "ankle_acs_from_foot_markers",
    "hip_angles_isb",
    "knee_angles_grood_suntay",
]


# ---------- C3D IO / LABELS ----------
def _norm_label(s: str) -> str:
    s = s.lower().replace("\x00", "").strip()
    return s.replace("_", "").replace("-", "").replace(" ", "")


def get_labels_xyz(c3d_path: str):
    c3d = ezc3d.c3d(c3d_path)
    pt = c3d["parameters"]["POINT"]
    labels = list(pt["LABELS"]["value"])
    if "LABELS2" in pt:
        labels += list(pt["LABELS2"]["value"])
    clean = []
    for s in labels:
        if isinstance(s, bytes):
            s = s.decode("utf-8", errors="ignore")
        clean.append(s.replace("\x00", "").strip())
    pts = c3d["data"]["points"]  # (4, N, F)
    xyz = np.transpose(pts[:3, :, :], (2, 1, 0))  # (F, N, 3)
    return clean, xyz


def pick(label: str, labels, xyz, frame_idx: int):
    if label is None:
        return None
    if label in labels:
        return xyz[frame_idx, labels.index(label), :].astype(float)
    lower_map = {lab.lower(): i for i, lab in enumerate(labels)}
    if label.lower() in lower_map:
        return xyz[frame_idx, lower_map[label.lower()], :].astype(float)
    norm_map = {_norm_label(lab): i for i, lab in enumerate(labels)}
    key = _norm_label(label)
    if key in norm_map:
        return xyz[frame_idx, norm_map[key], :].astype(float)
    return None


def axes_segments(
    label_prefix: str,
    origin_np: np.ndarray,
    R_np: np.ndarray,
    length_mm: float = 80.0,
    *,
    x_left: bool = False,
    y_suffix: str | None = None,
    z_suffix: str | None = None,
    x_suffix: str | None = None,
):
    ex, ey, ez = R_np[:, 0], R_np[:, 1], R_np[:, 2]
    # For visualization only: allow drawing X to the left while keeping math right-handed
    draw_x = -ex if x_left else ex
    x_name = f"{label_prefix} X_left" if x_left else f"{label_prefix} X"
    if x_suffix:
        x_name = f"{x_name} {x_suffix}"
    y_name = f"{label_prefix} Y" if y_suffix is None else f"{
        label_prefix} Y {y_suffix}"
    z_name = f"{label_prefix} Z" if z_suffix is None else f"{
        label_prefix} Z {z_suffix}"
    return [
        (x_name, "#e41a1c", origin_np, origin_np + length_mm * draw_x),
        (y_name, "#377eb8", origin_np, origin_np + length_mm * ey),
        (z_name, "#4daf4a", origin_np, origin_np + length_mm * ez),
    ]


def _enforce_right_handed(R: np.ndarray) -> np.ndarray:
    """Orthonormalize and ensure det=+1 using SVD."""
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def build_tcs_from_cluster(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build static Technical CS (TCS) from ≥3 non-collinear cluster markers.

    Returns (origin_T, R_T_world, C_local):
      - origin_T: centroid of points (3,)
      - R_T_world: 3x3 right-handed orthonormal basis of cluster in world
      - C_local: template points expressed in TCS (N,3), i.e., local canonical set
    """
    pts = np.asarray(points, float)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 3:
        raise ValueError("points must be (N,3) with N>=3")
    origin_T = pts.mean(axis=0)
    Q = pts - origin_T  # (N,3)
    # SVD on 3xN for principal axes in world
    U, _, _ = np.linalg.svd(Q.T, full_matrices=False)  # U: (3,3)
    R_T = _enforce_right_handed(U)
    # Local canonical template coordinates
    C_local = (R_T.T @ Q.T).T  # (N,3)
    return origin_T, R_T, C_local


def harrington_hip_centers(
    origin_pelvis: np.ndarray,
    R_pelvis: np.ndarray,
    LASIS: np.ndarray,
    RASIS: np.ndarray,
    LPSIS: np.ndarray,
    RPSIS: np.ndarray,
    anterior_coeff: float = 0.30,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute RHJC and LHJC using Harrington 2007 regression (mm) in pelvis CS.

    Pelvis ACS used here has axes: X right+, Y anterior+, Z superior+.

    Offsets (magnitudes):
        lateral:  0.24*PW + 0.0099*PD - 3.91
        anterior: 0.30*PW + 10.9  (use ``anterior_coeff`` to adjust if needed)
        superior: 0.33*PW + 7.3

    Signs in this ACS:
        right HJC:  x = +lateral, y = -anterior, z = -superior
        left  HJC:  x = -lateral, y = -anterior, z = -superior
    """
    PW = float(np.linalg.norm(LASIS - RASIS))
    mid_ASIS = 0.5 * (LASIS + RASIS)
    mid_PSIS = 0.5 * (LPSIS + RPSIS)
    PD = float(np.linalg.norm(mid_ASIS - mid_PSIS))

    off_lat = 0.24 * PW + 0.0099 * PD - 3.91
    off_ant_mag = anterior_coeff * PW + 10.9
    off_sup_mag = 0.33 * PW + 7.3

    RHJC_p = np.array([+off_lat, -off_ant_mag, -off_sup_mag], dtype=float)
    LHJC_p = np.array([-off_lat, -off_ant_mag, -off_sup_mag], dtype=float)

    RHJC = origin_pelvis + R_pelvis @ RHJC_p
    LHJC = origin_pelvis + R_pelvis @ LHJC_p

    # Guard: ensure LHJC ends up on the LASIS side in pelvis CS
    LASIS_p = R_pelvis.T @ (LASIS - origin_pelvis)
    RHJC_p2 = R_pelvis.T @ (RHJC - origin_pelvis)
    LHJC_p2 = R_pelvis.T @ (LHJC - origin_pelvis)
    if LHJC_p2[0] * LASIS_p[0] < 0 and RHJC_p2[0] * LASIS_p[0] > 0:
        RHJC, LHJC = LHJC, RHJC

    dims = dict(PW_mm=PW, PD_mm=PD)
    return RHJC, LHJC, dims


def femur_acs_from_landmarks(
    knee_medial: np.ndarray,
    knee_lateral: np.ndarray,
    hip_center: np.ndarray,
    origin_at: str = "hip",
    anterior_hint: np.ndarray | None = None,
    thigh_cluster_pts: np.ndarray | None = None,
    x_alignment_hint: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Femur ACS with Z superior (knee→hip), Y anterior, X right.

    Axes:
        - Z (superior): from knee joint center to hip joint center (KJC→HJC)
        - Y (anterior): pelvis anterior hint projected orthogonal to Z (fallbacks if hint missing)
        - X (right): X = Y × Z (right-handed)

    Origin:
        - "hip": at the hip joint center (proximal positioning)
        - "knee": at the knee joint center

    x_alignment_hint:
        Optional vector (e.g. lateral−medial at the knee) projected orthogonal to Z;
        if provided, X is flipped when misaligned with this direction, then Y is re-orthogonalized.
    """
    k_med = np.asarray(knee_medial, float)
    k_lat = np.asarray(knee_lateral, float)
    hjc = np.asarray(hip_center, float)

    kjc = 0.5 * (k_med + k_lat)
    # Long axis (superior): knee -> hip as Z
    z = hjc - kjc
    z_norm = np.linalg.norm(z)
    if z_norm == 0:
        raise ValueError(
            "Hip and knee centers coincide; cannot define femur Z axis")
    z = z / z_norm

    # Anterior axis: project hint onto plane orthogonal to Z (fallbacks if hint missing)
    if anterior_hint is not None:
        ah = np.asarray(anterior_hint, float)
        y_raw = ah - z * float(np.dot(ah, z))
    else:
        # Fallbacks: derive a forward direction from cluster or epicondyle line
        y_raw = None
        if thigh_cluster_pts is not None:
            try:
                pts = np.asarray(thigh_cluster_pts, float)
                if pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] >= 3:
                    _, R_seed_l, _ = build_tcs_from_cluster(pts)
                    y_raw = R_seed_l[:, 1] - z * \
                        float(np.dot(R_seed_l[:, 1], z))
            except Exception:
                y_raw = None
        if y_raw is None:
            # Use medial->lateral and cross with Z to get an anterior-like direction
            x_ml = k_lat - k_med
            x_ml = x_ml - z * float(np.dot(x_ml, z))
            if np.linalg.norm(x_ml) > 0:
                y_raw = np.cross(z, x_ml)
            else:
                y_raw = np.array([0.0, 1.0, 0.0])
    y_norm = np.linalg.norm(y_raw)
    if y_norm == 0:
        raise ValueError(
            "Cannot define femur Y axis (anterior) from provided inputs")
    y = y_raw / y_norm

    # X completes right-handed basis and should point to subject's right
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-12)

    if x_alignment_hint is not None:
        hint = np.asarray(x_alignment_hint, dtype=float).reshape(3)
        hint = hint - z * float(np.dot(hint, z))
        hn = np.linalg.norm(hint)
        if hn > 1e-12:
            hint = hint / hn
            if float(np.dot(x, hint)) < 0.0:
                x = -x
                y = np.cross(z, x)
                y = y / (np.linalg.norm(y) + 1e-12)
                x = np.cross(y, z)
                x = x / (np.linalg.norm(x) + 1e-12)

    R = np.column_stack([x, y, z])
    origin = hjc if origin_at.lower() == "hip" else kjc
    return origin, R


def tibia_acs_from_landmarks(
    knee_center: np.ndarray,
    ankle_medial: np.ndarray,
    ankle_lateral: np.ndarray,
    origin_at: str = "ankle",
    anterior_hint: np.ndarray | None = None,
    knee_medial: np.ndarray | None = None,
    knee_lateral: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tibia ACS using AJC for origin option and condyles to seed anterior if needed.

    Axes (consistent with updated femur/pelvis convention):
        - Z (superior/proximal): ankle center → knee center
        - Y (anterior): pelvis anterior hint projected orthogonal to Z (fallback: use condyle line)
        - X (right/lateral): X = Y × Z (right-handed)

    Returns (origin, R, ankle_center). Origin placed per ``origin_at`` ("ankle" or "knee").
    """
    k = np.asarray(knee_center, float)
    a_med = np.asarray(ankle_medial, float)
    a_lat = np.asarray(ankle_lateral, float)
    ajc = 0.5 * (a_med + a_lat)

    # Z up (proximal): ankle -> knee
    z = k - ajc
    z_norm = np.linalg.norm(z)
    if z_norm == 0:
        raise ValueError(
            "Knee and ankle centers coincide; cannot define tibia Z axis")
    z = z / z_norm

    # Y anterior: from hint projected orthogonal to Z; fallback to malleoli-derived
    if anterior_hint is not None:
        ah = np.asarray(anterior_hint, float)
        y_raw = ah - z * float(np.dot(ah, z))
    else:
        y_raw = None
    if y_raw is None or np.linalg.norm(y_raw) == 0:
        # Prefer using condyle line to seed X, then derive Y = Z × X (approx anterior)
        x_seed = None
        if (knee_medial is not None) and (knee_lateral is not None):
            km = np.asarray(knee_medial, float)
            kl = np.asarray(knee_lateral, float)
            kc_line = kl - km  # medial->lateral at knee
            x_seed = kc_line - z * float(np.dot(kc_line, z))
        if (x_seed is None) or (np.linalg.norm(x_seed) == 0):
            # fallback: use malleoli line if condyles unavailable
            lm = a_lat - a_med  # medial->lateral across malleoli
            x_seed = lm - z * float(np.dot(lm, z))
        if np.linalg.norm(x_seed) == 0:
            # generic fallback
            x_seed = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(x_seed, z)) > 0.9:
                x_seed = np.array([0.0, 1.0, 0.0])
            x_seed = x_seed - z * float(np.dot(x_seed, z))
        x_seed = x_seed / (np.linalg.norm(x_seed) + 1e-12)
        y_raw = np.cross(z, x_seed)
    y = y_raw / (np.linalg.norm(y_raw) + 1e-12)

    # X completes right-hand
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-12)

    R = np.column_stack([x, y, z])
    origin = ajc if origin_at.lower() == "ankle" else k
    return origin, R, ajc


def ankle_acs_from_foot_markers(
    ankle_medial: np.ndarray,
    ankle_lateral: np.ndarray,
    calcaneus: np.ndarray,
    met1: np.ndarray,
    met5: np.ndarray,
    met2: np.ndarray | None = None,
    tibia_z_hint: np.ndarray | None = None,
    pelvis_x_hint: np.ndarray | None = None,
    global_up: np.ndarray | tuple[float, float, float] = (0.0, 0.0, 1.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ankle/foot ACS at the ankle joint center (AJC).

    Origin:
        - AJC = midpoint of medial and lateral malleoli

    Axes:
        Build from hinge first:
        - X (hinge/right): medial→lateral across malleoli, sign enforced once via pelvis_x_hint (subject right)
        - Z (superior): parallel to tibia_z_hint but orthogonalized to X; fallback to plantar plane normal, then orthogonalize to X; flipped upward
        - Y (anterior): Y = Z × X (right-handed)
    """
    a_med = np.asarray(ankle_medial, float)
    a_lat = np.asarray(ankle_lateral, float)
    calc = np.asarray(calcaneus, float)
    mh1 = np.asarray(met1, float)
    mh5 = np.asarray(met5, float)
    mh2 = None if met2 is None else np.asarray(met2, float)

    ajc = 0.5 * (a_med + a_lat)

    up = np.asarray(global_up, float)
    # 1) Z up (proximal): prefer tibia_z_hint; else plantar-plane normal, flipped upward
    if tibia_z_hint is not None:
        z = np.asarray(tibia_z_hint, float)
    else:
        v1 = mh1 - calc
        v5 = mh5 - calc
        z = np.cross(v5, v1)
        if np.linalg.norm(z) == 0:
            z = up
    z = z / (np.linalg.norm(z) + 1e-12)
    if np.dot(z, up) < 0:
        z = -z

    # 2) Y anterior: from calc->toe (or midpoint toes), projected orthogonal to Z
    if met2 is not None and np.isfinite(mh2).all():
        y_raw = mh2 - calc
    else:
        y_raw = mh1 - calc
        if np.linalg.norm(y_raw) == 0:
            y_raw = 0.5 * (mh1 + mh5) - calc
    y_raw = y_raw - z * float(np.dot(y_raw, z))
    if np.linalg.norm(y_raw) == 0:
        # Fallback using pelvis X to derive an anterior-like direction
        if pelvis_x_hint is not None:
            px = np.asarray(pelvis_x_hint, float)
            y_raw = np.cross(z, px)
        else:
            y_raw = np.array([0.0, 1.0, 0.0])
            if abs(np.dot(y_raw, z)) > 0.9:
                y_raw = np.array([1.0, 0.0, 0.0])
                y_raw = y_raw - z * float(np.dot(y_raw, z))
    y = y_raw / (np.linalg.norm(y_raw) + 1e-12)

    # 3) X right: X = Y × Z; enforce alignment with pelvis X (subject right)
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-12)
    if pelvis_x_hint is not None:
        px = np.asarray(pelvis_x_hint, float)
        if np.dot(x, px) < 0:
            # Flip Y to flip X while preserving Z and right-handedness
            y = -y
            x = np.cross(y, z)
            x = x / (np.linalg.norm(x) + 1e-12)

    # Final orthonormalization for numerical stability
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-12)
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-12)

    R = np.column_stack([x, y, z])
    return ajc, R, ajc


def _segment_tcs_R_from_proximal_z(
    z_proximal: np.ndarray,
    R_pelvis: np.ndarray,
    cluster_pts: np.ndarray | None,
) -> np.ndarray | None:
    """Right-handed TCS with columns [X right, Y anterior, Z proximal] from long-axis Z and pelvis anterior."""
    z = np.asarray(z_proximal, dtype=float).reshape(3)
    zn = np.linalg.norm(z)
    if zn < 1e-12:
        return None
    z = z / zn
    anterior = np.asarray(R_pelvis, dtype=float)[:, 1]
    y_raw = anterior - z * float(np.dot(anterior, z))
    if np.linalg.norm(y_raw) < 1e-12 and cluster_pts is not None:
        pts = np.asarray(cluster_pts, dtype=float)
        if pts.ndim == 2 and pts.shape[1] == 3 and pts.shape[0] >= 3:
            _, R_seed, _ = build_tcs_from_cluster(pts)
            y_raw = R_seed[:, 1] - z * float(np.dot(R_seed[:, 1], z))
    yn = np.linalg.norm(y_raw)
    if yn < 1e-12:
        return None
    y = y_raw / yn
    if float(np.dot(y, anterior)) < 0.0:
        y = -y
    x = np.cross(y, z)
    xn = np.linalg.norm(x)
    if xn < 1e-12:
        return None
    x = x / xn
    y = np.cross(z, x)
    y = y / (np.linalg.norm(y) + 1e-12)
    return np.column_stack([x, y, z])


def _euler_xyz_from_R(R: np.ndarray) -> tuple[float, float, float]:
    """XYZ Cardan angles in degrees (same convention as prior nested helper)."""
    R = np.asarray(R, dtype=float)
    sy = float(np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    if sy > 1e-8:
        x = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        y = np.degrees(np.arctan2(-R[2, 0], sy))
        z = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        x = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        y = np.degrees(np.arctan2(-R[2, 0], sy))
        z = 0.0
    return float(x), float(y), float(z)


def _signed_angle_about_axis(u: np.ndarray, v: np.ndarray, axis: np.ndarray) -> float:
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


def hip_angles_isb(R_pelvis: np.ndarray, R_femur: np.ndarray) -> tuple[float, float, float]:
    """
    ISB hip JCS: pelvis X (FE), femur Z (axial), floating e2 = e3 × e1.
    Returns (FE, AbAd, IE) in degrees.
    """
    if R_pelvis is None or R_femur is None or (np.isnan(R_pelvis).any() or np.isnan(R_femur).any()):
        return float("nan"), float("nan"), float("nan")
    Rp = _enforce_right_handed(np.asarray(R_pelvis, float))
    Rf = _enforce_right_handed(np.asarray(R_femur, float))
    pX, pY, pZ = Rp[:, 0], Rp[:, 1], Rp[:, 2]
    fX, fY, fZ = Rf[:, 0], Rf[:, 1], Rf[:, 2]
    if float(np.dot(fZ, pZ)) < 0.0:
        Rf = Rf.copy()
        Rf[:, [0, 2]] *= -1.0
        fX, fY, fZ = Rf[:, 0], Rf[:, 1], Rf[:, 2]
    e1 = pX / (np.linalg.norm(pX) + 1e-12)
    e3 = fZ / (np.linalg.norm(fZ) + 1e-12)
    e2 = np.cross(e3, e1)
    if np.linalg.norm(e2) < 1e-12:
        e2 = np.cross(e3, pZ)
    e2 = e2 / (np.linalg.norm(e2) + 1e-12)
    FE = _signed_angle_about_axis(pZ, fZ, e1)
    AbAd = _signed_angle_about_axis(pZ, e3, e2)
    pX_perp = pX - e3 * float(np.dot(pX, e3))
    fX_perp = fX - e3 * float(np.dot(fX, e3))
    IE = _signed_angle_about_axis(pX_perp, fX_perp, e3)
    return float(FE), float(AbAd), float(IE)


def knee_angles_grood_suntay(
    R_femur_in: np.ndarray, R_tibia_in: np.ndarray, side: str = "left"
) -> tuple[float, float, float]:
    """Grood–Suntay knee angles (degrees); FE, Var/Val, IE."""
    R_femur = R_femur_in

    def compute_angles(fX_seed: np.ndarray, Rtib: np.ndarray) -> tuple[float, float, float]:
        femur_Z = R_femur[:, 2]
        femur_Y = R_femur[:, 1]
        tibia_X = Rtib[:, 0]
        tibia_Z = Rtib[:, 2]
        fX = fX_seed - femur_Z * float(np.dot(fX_seed, femur_Z))
        if np.linalg.norm(fX) < 1e-12:
            fX = np.cross(femur_Z, femur_Y)
        fX = fX / (np.linalg.norm(fX) + 1e-12)
        e3 = tibia_Z / (np.linalg.norm(tibia_Z) + 1e-12)
        e2 = np.cross(e3, fX)
        if np.linalg.norm(e2) < 1e-12:
            e2 = np.cross(e3, tibia_X)
        e2 = e2 / (np.linalg.norm(e2) + 1e-12)
        FE = _signed_angle_about_axis(R_femur[:, 2], tibia_Z, fX)
        AbAd = _signed_angle_about_axis(R_femur[:, 2], e3, e2)
        IE = _signed_angle_about_axis(fX, tibia_X, e3)
        return FE, AbAd, IE

    def best_with_tibia(Rtib: np.ndarray) -> tuple[float, float, float]:
        FE1, AbAd1, IE1 = compute_angles(R_femur[:, 0], Rtib)
        FE2, AbAd2, IE2 = compute_angles(-R_femur[:, 0], Rtib)
        cand = [(FE1, AbAd1, IE1), (FE2, AbAd2, IE2)]
        cand.sort(key=lambda t: (abs(t[2]), abs(t[1])))
        return cand[0]

    Rt_flip = R_tibia_in.copy()
    Rt_flip[:, [0, 2]] *= -1.0
    c1 = best_with_tibia(R_tibia_in)
    c2 = best_with_tibia(Rt_flip)
    FE, AbAd, IE = min([c1, c2], key=lambda t: (abs(t[2]), abs(t[1])))
    if side.lower().startswith("l"):
        FE = -FE
    return FE, AbAd, IE


def _ankle_raw_R_for_euler(
    labels,
    xyz,
    frame_idx: int,
    side: str,
    R_ankle_fallback: np.ndarray,
) -> np.ndarray:
    """Foot ACS from markers only (no tibia/pelvis hints) for ankle angle neutralization."""
    pref = "L_" if side.lower().startswith("l") else "R_"
    am = pick(f"{pref}Ank_Med", labels, xyz, frame_idx)
    al = pick(f"{pref}Ank_Lat", labels, xyz, frame_idx)
    calc = pick(f"{pref}Calc", labels, xyz, frame_idx)
    tm = pick(f"{pref}Toe_Med", labels, xyz, frame_idx)
    tl = pick(f"{pref}Toe_Lat", labels, xyz, frame_idx)
    tt = pick(f"{pref}Toe_Tip", labels, xyz, frame_idx)
    if (am is not None) and (al is not None) and (calc is not None) and (tm is not None) and (tl is not None):
        _, R_raw, _ = ankle_acs_from_foot_markers(
            ankle_medial=am,
            ankle_lateral=al,
            calcaneus=calc,
            met1=tm,
            met5=tl,
            met2=tt,
            tibia_z_hint=None,
            pelvis_x_hint=None,
        )
        return _enforce_right_handed(R_raw)
    return _enforce_right_handed(R_ankle_fallback)


def compute_and_plot_static_calibration(
    c3d_path: str,
    frame_idx: int = 0,
    out_html: str | None = None,
    axis_len_mm: float = 80.0,
) -> None:
    labels, xyz = get_labels_xyz(c3d_path)
    # Prepare per-trial output directory in scripts dir (or alongside provided out_html)
    try:
        base = os.path.splitext(os.path.basename(c3d_path))[0]
        subject = os.path.basename(os.path.dirname(c3d_path))
        folder_name = f"{subject} - {base}"
        out_root_dir = os.path.dirname(
            out_html) if out_html else os.path.dirname(__file__)
        out_dir_trial = os.path.join(out_root_dir, folder_name)
        os.makedirs(out_dir_trial, exist_ok=True)
    except Exception:
        out_dir_trial = "."

    # Collect joint centers (lab coords, mm) for CSV export
    joint_centers_dict = {}

    PELVIS = dict(RASIS="R_ASIS", LASIS="L_ASIS",
                  RPSIS="R_PSIS", LPSIS="L_PSIS")

    RASIS = pick(PELVIS["RASIS"], labels, xyz, frame_idx)
    LASIS = pick(PELVIS["LASIS"], labels, xyz, frame_idx)
    RPSIS = pick(PELVIS["RPSIS"], labels, xyz, frame_idx)
    LPSIS = pick(PELVIS["LPSIS"], labels, xyz, frame_idx)

    pelvis_vals = {"R_ASIS": RASIS, "L_ASIS": LASIS,
                   "R_PSIS": RPSIS, "L_PSIS": LPSIS}
    missing = [k for k, v in pelvis_vals.items() if v is None]
    if missing:
        raise ValueError(f"Pelvis markers missing: {', '.join(missing)}")

    # Units guard
    PW_raw = float(np.linalg.norm(LASIS - RASIS))
    if PW_raw < 1.0:
        xyz[:] *= 1000.0
        RASIS = pick(PELVIS["RASIS"], labels, xyz, frame_idx)
        LASIS = pick(PELVIS["LASIS"], labels, xyz, frame_idx)
        RPSIS = pick(PELVIS["RPSIS"], labels, xyz, frame_idx)
        LPSIS = pick(PELVIS["LPSIS"], labels, xyz, frame_idx)

    origin_sym, R_sym, *_ = compute_pelvis_acs(LASIS, RASIS, LPSIS, RPSIS)

    origin_np = np.array(origin_sym, dtype=float).reshape(3)
    R_np = np.array(R_sym, dtype=float)

    fig = go.Figure()
    for name, P in pelvis_vals.items():
        fig.add_trace(go.Scatter3d(
            x=[P[0]], y=[P[1]], z=[P[2]],
            mode="markers+text",
            marker=dict(size=6, color="#1f77b4"),
            text=[name], textposition="top center",
            name=name
        ))

    for nm, color, p0, p1 in axes_segments("Pelvis", origin_np, R_np, axis_len_mm, y_suffix="(anterior)", z_suffix="(superior)"):
        fig.add_trace(go.Scatter3d(
            x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
            mode="lines", line=dict(color=color, width=6), name=nm
        ))

    # Pelvis TCS from cluster markers, intent-aligned: Z superior, Y anterior, X lateral (right)
    pelvis_cluster = [P for P in (RASIS, LASIS, RPSIS, LPSIS) if P is not None]
    if len(pelvis_cluster) >= 3:
        pts_p = np.vstack(pelvis_cluster)
        O_T = pts_p.mean(axis=0)
        # X right from LASIS -> RASIS
        X_right = None
        if (RASIS is not None) and (LASIS is not None):
            xr = RASIS - LASIS
            if np.linalg.norm(xr) > 0:
                X_right = xr / np.linalg.norm(xr)
        # Y anterior from PSIS_mid -> ASIS_mid
        Y_ant = None
        if (RASIS is not None) and (LASIS is not None) and (RPSIS is not None) and (LPSIS is not None):
            asis_mid = 0.5 * (RASIS + LASIS)
            psis_mid = 0.5 * (RPSIS + LPSIS)
            yraw = asis_mid - psis_mid
            if X_right is not None:
                yraw = yraw - X_right * float(np.dot(yraw, X_right))
            if np.linalg.norm(yraw) > 0:
                Y_ant = yraw / np.linalg.norm(yraw)
        # Fallback to SVD axes if needed (single cluster decomposition)
        R_seed_p = None
        if (X_right is None) or (Y_ant is None):
            _, R_seed_p, _ = build_tcs_from_cluster(pts_p)
            if X_right is None:
                X_right = R_seed_p[:, 0]
            if Y_ant is None:
                Y_ant = R_seed_p[:, 1]
        # Z superior completes right-hand
        Z_sup = np.cross(X_right, Y_ant)
        if np.linalg.norm(Z_sup) == 0:
            if R_seed_p is None:
                _, R_seed_p, _ = build_tcs_from_cluster(pts_p)
            Z_sup = R_seed_p[:, 2]
        Z_sup = Z_sup / np.linalg.norm(Z_sup)
        # Ensure Z is superior
        if np.dot(Z_sup, np.array([0.0, 0.0, 1.0])) < 0:
            Y_ant = -Y_ant
            Z_sup = -Z_sup
        # Re-orthogonalize X to ensure perfect basis
        X_right = np.cross(Y_ant, Z_sup)
        X_right = X_right / np.linalg.norm(X_right)
        R_T = np.column_stack([X_right, Y_ant, Z_sup])
        C_local = (R_T.T @ (pts_p - O_T).T).T
        for nm, color, p0, p1 in axes_segments("Pelvis TCS", O_T, R_T, axis_len_mm * 0.8, y_suffix="(anterior)", z_suffix="(superior)"):
            fig.add_trace(go.Scatter3d(
                x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
                mode="lines", line=dict(color=color, width=4), name=nm
            ))
        # Save TCS template for dynamics (Kabsch) + static offsets (how A sits in T)
        try:
            base = os.path.splitext(os.path.basename(c3d_path))[0]
            out_dir = out_dir_trial
            tplt_path = os.path.join(
                out_dir, f"{base}_pelvis_tcs_template.npz")
            # Static transforms between frames
            R_TA = R_T.T @ R_np
            t_TA = R_T.T @ (origin_np - O_T)
            R_AT = R_np.T @ R_T
            t_AT = R_np.T @ (O_T - origin_np)
            np.savez(
                tplt_path,
                marker_labels=np.array([n for n, P in {
                    "R_ASIS": RASIS, "L_ASIS": LASIS, "R_PSIS": RPSIS, "L_PSIS": LPSIS
                }.items() if P is not None], dtype=object),
                # Tracking frame (static)
                origin_T=O_T,
                R_T_world=R_T,
                C_local=C_local,
                # Anatomical frame (static)
                O_A_static=origin_np,
                R_A_static=R_np,
                # Offsets: anatomy expressed in tracking
                R_TA=R_TA,
                t_TA=t_TA,
                # Convenience inverse
                R_AT=R_AT,
                t_AT=t_AT,
            )
            print(f"Saved pelvis TCS template: {os.path.abspath(tplt_path)}")
            print("Pelvis static A-in-T: R_TA=\n", R_TA, "\n t_TA=", t_TA)
        except Exception as e:
            print(f"Warning: could not save pelvis TCS template: {e}")

    # Harrington hip centers (default anterior_coeff in harrington_hip_centers)
    RHJC, LHJC, dims = harrington_hip_centers(
        origin_np, R_np, LASIS, RASIS, LPSIS, RPSIS)
    joint_centers_dict["RHJC"] = np.asarray(
        RHJC, dtype=float).reshape(3).copy()
    joint_centers_dict["LHJC"] = np.asarray(
        LHJC, dtype=float).reshape(3).copy()
    fig.add_trace(go.Scatter3d(
        x=[RHJC[0]], y=[RHJC[1]], z=[RHJC[2]],
        mode="markers+text",
        marker=dict(size=7, color="#d62728", symbol="diamond"),
        text=["Right HJC"], textposition="bottom right",
        name="Right HJC"
    ))
    fig.add_trace(go.Scatter3d(
        x=[LHJC[0]], y=[LHJC[1]], z=[LHJC[2]],
        mode="markers+text",
        marker=dict(size=7, color="#2ca02c", symbol="diamond"),
        text=["Left HJC"], textposition="bottom right",
        name="Left HJC"
    ))

    # Femur TCS (left) from thigh cluster markers if available
    L_thigh_AS = pick("L_Thigh_AS", labels, xyz, frame_idx)
    L_thigh_PS = pick("L_Thigh_PS", labels, xyz, frame_idx)
    L_thigh_AI = pick("L_Thigh_AI", labels, xyz, frame_idx)
    L_thigh_PI = pick("L_Thigh_PI", labels, xyz, frame_idx)
    thigh_cluster = [P for P in (
        L_thigh_AS, L_thigh_PS, L_thigh_AI, L_thigh_PI) if P is not None]
    R_T_femur = None
    C_local_femur = None
    if len(thigh_cluster) >= 3:
        pts_l = np.vstack(thigh_cluster)
        O_T_femur = pts_l.mean(axis=0)
        L_knee_lat_tcs = pick("L_Knee_Lat", labels, xyz, frame_idx)
        L_knee_med_tcs = pick("L_Knee_Med", labels, xyz, frame_idx)
        if (L_knee_lat_tcs is not None) and (L_knee_med_tcs is not None):
            L_KJC_tcs = 0.5 * (L_knee_med_tcs + L_knee_lat_tcs)
            R_T_femur = _segment_tcs_R_from_proximal_z(
                LHJC - L_KJC_tcs, R_np, pts_l)
            if R_T_femur is not None:
                C_local_femur = (R_T_femur.T @ (pts_l - O_T_femur).T).T
        if R_T_femur is not None:
            try:
                print("det left thigh TCS:", float(np.linalg.det(R_T_femur)))
            except Exception:
                pass
            for nm, color, p0, p1 in axes_segments("Femur TCS", O_T_femur, R_T_femur, axis_len_mm * 0.8, y_suffix="(anterior)", z_suffix="(superior)"):
                fig.add_trace(go.Scatter3d(
                    x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
                    mode="lines", line=dict(color=color, width=4), name=nm
                ))

    # Tibia (shank) TCS (left) from shank cluster markers if available
    L_shank_AS = pick("L_Shank_AS", labels, xyz, frame_idx)
    L_shank_PS = pick("L_Shank_PS", labels, xyz, frame_idx)
    L_shank_AI = pick("L_Shank_AI", labels, xyz, frame_idx)
    L_shank_PI = pick("L_Shank_PI", labels, xyz, frame_idx)
    shank_cluster = [P for P in (
        L_shank_AS, L_shank_PS, L_shank_AI, L_shank_PI) if P is not None]
    # Plot left shank markers (teal)
    left_shank_points = [("L_Shank_AS", L_shank_AS), ("L_Shank_PS", L_shank_PS),
                         ("L_Shank_AI", L_shank_AI), ("L_Shank_PI", L_shank_PI)]
    for name, P in left_shank_points:
        if P is not None:
            fig.add_trace(go.Scatter3d(
                x=[P[0]], y=[P[1]], z=[P[2]],
                mode="markers+text",
                marker=dict(size=5, color="#17becf"),
                text=[name], textposition="top center",
                name=name,
                showlegend=False
            ))
    if len(shank_cluster) >= 3:
        pts_s = np.vstack(shank_cluster)
        O_T_tibia = pts_s.mean(axis=0)
        # Ensure axes by intent: Y anterior, Z superior (proximal), X right
        L_knee_lat_tcs = pick("L_Knee_Lat", labels, xyz, frame_idx)
        L_knee_med_tcs = pick("L_Knee_Med", labels, xyz, frame_idx)
        L_ank_lat_tcs = pick("L_Ank_Lat", labels, xyz, frame_idx)
        L_ank_med_tcs = pick("L_Ank_Med", labels, xyz, frame_idx)
        R_T_tibia = None
        C_local_tibia = None
        if (L_knee_lat_tcs is not None) and (L_knee_med_tcs is not None) and (L_ank_lat_tcs is not None) and (L_ank_med_tcs is not None):
            L_KJC_tcs = 0.5 * (L_knee_med_tcs + L_knee_lat_tcs)
            L_AJC_tcs = 0.5 * (L_ank_med_tcs + L_ank_lat_tcs)
            joint_centers_dict.setdefault(
                "L_KJC", np.asarray(L_KJC_tcs, dtype=float).copy())
            joint_centers_dict.setdefault(
                "L_AJC", np.asarray(L_AJC_tcs, dtype=float).copy())
            R_T_tibia = _segment_tcs_R_from_proximal_z(
                L_KJC_tcs - L_AJC_tcs, R_np, pts_s)
            if R_T_tibia is not None:
                C_local_tibia = (R_T_tibia.T @ (pts_s - O_T_tibia).T).T
        # Fallback to SVD TCS if markers insufficient or computation failed
        if R_T_tibia is None:
            _, R_seed_s, _ = build_tcs_from_cluster(pts_s)
            R_T_tibia = R_seed_s
            C_local_tibia = (R_T_tibia.T @ (pts_s - O_T_tibia).T).T
        # Determinant check to ensure right-handedness
        try:
            print("det left shank TCS:", float(np.linalg.det(R_T_tibia)))
        except Exception:
            pass
        for nm, color, p0, p1 in axes_segments("Tibia TCS", O_T_tibia, R_T_tibia, axis_len_mm * 0.8, y_suffix="(anterior)", z_suffix="(superior)"):
            fig.add_trace(go.Scatter3d(
                x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
                mode="lines", line=dict(color=color, width=4), name=nm
            ))
        # (Defer saving tibia template until ACS is computed to include static offsets)

    # Foot TCS (left) from available foot markers (need ≥3)
    L_calc = pick("L_Calc", labels, xyz, frame_idx)
    L_mid_sup = pick("L_Midfoot_Sup", labels, xyz, frame_idx)
    L_mid_lat = pick("L_Midfoot_Lat", labels, xyz, frame_idx)
    L_toe_med = pick("L_Toe_Med", labels, xyz, frame_idx)
    L_toe_lat = pick("L_Toe_Lat", labels, xyz, frame_idx)
    L_toe_tip = pick("L_Toe_Tip", labels, xyz, frame_idx)
    foot_cluster = [P for P in (
        L_calc, L_mid_sup, L_mid_lat, L_toe_med, L_toe_lat, L_toe_tip) if P is not None]
    if len(foot_cluster) >= 3:
        pts_f = np.vstack(foot_cluster)
        O_T_foot = pts_f.mean(axis=0)
        # Ensure axes by intent: Y anterior, Z superior, X medial (left side => toward right)
        # Z from plantar plane using calc + toe med/lat or midfoot sup/lat fallback
        Z_sup_l = None
        if (L_calc is not None) and (L_toe_med is not None) and (L_toe_lat is not None):
            v1 = L_toe_med - L_calc
            v5 = L_toe_lat - L_calc
            Z_sup_l = np.cross(v5, v1)
        elif (L_calc is not None) and (L_mid_sup is not None) and (L_mid_lat is not None):
            v1 = L_mid_sup - L_calc
            v5 = L_mid_lat - L_calc
            Z_sup_l = np.cross(v5, v1)
        if Z_sup_l is None or np.linalg.norm(Z_sup_l) == 0:
            # Fallback to cluster normal
            _, R_seed_f, _ = build_tcs_from_cluster(pts_f)
            Z_sup_l = R_seed_f[:, 2]
        Z_sup_l = Z_sup_l / np.linalg.norm(Z_sup_l)
        if np.dot(Z_sup_l, np.array([0.0, 0.0, 1.0])) < 0:
            Z_sup_l = -Z_sup_l
        # Y anterior from calc to toe tip or midpoint toe med/lat, projected orthogonal to Z
        if L_toe_tip is not None and L_calc is not None:
            y_raw = L_toe_tip - L_calc
        elif (L_toe_med is not None) and (L_toe_lat is not None) and (L_calc is not None):
            y_raw = 0.5 * (L_toe_med + L_toe_lat) - L_calc
        else:
            # Fallback: use cluster Y seed projected
            _, R_seed_f, _ = build_tcs_from_cluster(pts_f)
            y_raw = R_seed_f[:, 1]
        Y_ant_l = y_raw - Z_sup_l * float(np.dot(y_raw, Z_sup_l))
        if np.linalg.norm(Y_ant_l) == 0:
            Y_ant_l = np.array([0.0, 1.0, 0.0])
        Y_ant_l = Y_ant_l / np.linalg.norm(Y_ant_l)
        # Align with pelvis anterior
        if np.dot(Y_ant_l, R_np[:, 1]) < 0:
            Y_ant_l = -Y_ant_l
        # X medial (left): for left limb, medial points to subject right => X = Y × Z
        X_med_l = np.cross(Y_ant_l, Z_sup_l)
        if np.linalg.norm(X_med_l) > 0:
            X_med_l = X_med_l / np.linalg.norm(X_med_l)
        R_T_foot = np.column_stack([X_med_l, Y_ant_l, Z_sup_l])
        C_local_foot = (R_T_foot.T @ (pts_f - O_T_foot).T).T
        for nm, color, p0, p1 in axes_segments("Foot TCS", O_T_foot, R_T_foot, axis_len_mm * 0.8):
            fig.add_trace(go.Scatter3d(
                x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
                mode="lines", line=dict(color=color, width=4), name=nm
            ))
        # (Defer saving foot template until ankle ACS is computed to include static offsets)

    # Left side only
    pref = "L_"

    # Left knee markers
    knee_lat = pick(pref + "Knee_Lat", labels, xyz, frame_idx)
    knee_med = pick(pref + "Knee_Med", labels, xyz, frame_idx)
    if (knee_lat is None) or (knee_med is None):
        print(f"Knee markers not found: {pref}Knee_Lat or {pref}Knee_Med")
    else:
        # Plot knee markers
        fig.add_trace(go.Scatter3d(
            x=[knee_lat[0]], y=[knee_lat[1]], z=[knee_lat[2]],
            mode="markers+text",
            marker=dict(size=6, color="#ff7f0e"),
            text=[pref + "Knee_Lat"], textposition="top center",
            name=pref + "Knee_Lat"
        ))
        fig.add_trace(go.Scatter3d(
            x=[knee_med[0]], y=[knee_med[1]], z=[knee_med[2]],
            mode="markers+text",
            marker=dict(size=6, color="#ff7f0e"),
            text=[pref + "Knee_Med"], textposition="top center",
            name=pref + "Knee_Med"
        ))
        # Epicondyle line
        fig.add_trace(go.Scatter3d(
            x=[knee_med[0], knee_lat[0]], y=[knee_med[1],
                                             knee_lat[1]], z=[knee_med[2], knee_lat[2]],
            mode="lines", line=dict(color="#ff7f0e", width=3), name="left Knee epicondyle"
        ))

        # Femur ACS (left) via function: Z up (KJC→HJC), Y anterior, X right (right-handed)
        O_femur, R_femur = femur_acs_from_landmarks(
            knee_medial=knee_med,
            knee_lateral=knee_lat,
            hip_center=LHJC,
            origin_at="hip",
            anterior_hint=R_np[:, 1],
            thigh_cluster_pts=(pts_l if 'pts_l' in locals() else None),
        )
        try:
            print("det left femur ACS:", float(np.linalg.det(R_femur)))
        except Exception:
            pass
        for nm, color, p0, p1 in axes_segments("Femur (left)", O_femur, R_femur, axis_len_mm):
            fig.add_trace(go.Scatter3d(
                x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
                mode="lines", line=dict(color=color, width=5), name=nm
            ))

        # Save femur TCS template with static offsets (if TCS was built above)
        if (
            "O_T_femur" in locals()
            and locals().get("R_T_femur") is not None
            and locals().get("C_local_femur") is not None
        ):
            try:
                base = os.path.splitext(os.path.basename(c3d_path))[0]
                out_dir = out_dir_trial
                tplt_path = os.path.join(
                    out_dir, f"{base}_femur_tcs_template.npz")
                used = [("L_Thigh_AS", L_thigh_AS), ("L_Thigh_PS", L_thigh_PS),
                        ("L_Thigh_AI", L_thigh_AI), ("L_Thigh_PI", L_thigh_PI)]
                R_TA = R_T_femur.T @ R_femur
                t_TA = R_T_femur.T @ (O_femur - O_T_femur)
                R_AT = R_femur.T @ R_T_femur
                t_AT = R_femur.T @ (O_T_femur - O_femur)
                np.savez(
                    tplt_path,
                    marker_labels=np.array(
                        [name for name, P in used if P is not None], dtype=object),
                    origin_T=O_T_femur,
                    R_T_world=R_T_femur,
                    C_local=C_local_femur,
                    O_A_static=O_femur,
                    R_A_static=R_femur,
                    R_TA=R_TA,
                    t_TA=t_TA,
                    R_AT=R_AT,
                    t_AT=t_AT,
                )
                print(f"Saved femur TCS template: {
                      os.path.abspath(tplt_path)}")
                print("Femur (L) static A-in-T: R_TA=\n", R_TA, "\n t_TA=", t_TA)
            except Exception as e:
                print(f"Warning: could not save femur TCS template: {e}")

        # Right femur ACS: X right (lateral), Y anterior, Z up (proximal) — same convention as left
        R_knee_lat = pick("R_Knee_Lat", labels, xyz, frame_idx)
        R_knee_med = pick("R_Knee_Med", labels, xyz, frame_idx)
        if (R_knee_lat is not None) and (R_knee_med is not None):
            # Plot right knee markers and epicondyle line
            fig.add_trace(go.Scatter3d(
                x=[R_knee_lat[0]], y=[R_knee_lat[1]], z=[R_knee_lat[2]],
                mode="markers+text",
                marker=dict(size=6, color="#ff7f0e"),
                text=["R_Knee_Lat"], textposition="top center",
                name="R_Knee_Lat"
            ))
            fig.add_trace(go.Scatter3d(
                x=[R_knee_med[0]], y=[R_knee_med[1]], z=[R_knee_med[2]],
                mode="markers+text",
                marker=dict(size=6, color="#ff7f0e"),
                text=["R_Knee_Med"], textposition="top center",
                name="R_Knee_Med"
            ))
            fig.add_trace(go.Scatter3d(
                x=[R_knee_med[0], R_knee_lat[0]], y=[R_knee_med[1],
                                                     R_knee_lat[1]], z=[R_knee_med[2], R_knee_lat[2]],
                mode="lines", line=dict(color="#ff7f0e", width=3), name="right Knee epicondyle"
            ))
            R_KJC = 0.5 * (R_knee_med + R_knee_lat)
            joint_centers_dict.setdefault(
                "R_KJC", np.asarray(R_KJC, dtype=float).copy())
            O_femur_right, R_femur_right = femur_acs_from_landmarks(
                knee_medial=R_knee_med,
                knee_lateral=R_knee_lat,
                hip_center=RHJC,
                origin_at="hip",
                anterior_hint=R_np[:, 1],
                x_alignment_hint=R_knee_lat - R_knee_med,
            )
            try:
                print("det right femur ACS:", float(
                    np.linalg.det(R_femur_right)))
            except Exception:
                pass
            for nm, color, p0, p1 in axes_segments("Femur (right)", O_femur_right, R_femur_right, axis_len_mm):
                fig.add_trace(go.Scatter3d(
                    x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
                    mode="lines", line=dict(color=color, width=5), name=nm
                ))

            # Right thigh TCS: X right, Y anterior, Z up (proximal) — match left convention
            R_thigh_AS = pick("R_Thigh_AS", labels, xyz, frame_idx)
            R_thigh_PS = pick("R_Thigh_PS", labels, xyz, frame_idx)
            R_thigh_AI = pick("R_Thigh_AI", labels, xyz, frame_idx)
            R_thigh_PI = pick("R_Thigh_PI", labels, xyz, frame_idx)
            thigh_cluster_r = [P for P in (
                R_thigh_AS, R_thigh_PS, R_thigh_AI, R_thigh_PI) if P is not None]
            if (R_knee_lat is not None) and (R_knee_med is not None) and len(thigh_cluster_r) >= 3:
                pts_r = np.vstack(thigh_cluster_r)
                O_T_femur_r = pts_r.mean(axis=0)
                R_KJC = 0.5 * (R_knee_med + R_knee_lat)
                R_T_femur_r = _segment_tcs_R_from_proximal_z(
                    RHJC - R_KJC, R_np, pts_r)
                if R_T_femur_r is not None:
                    C_local_femur_r = (
                        R_T_femur_r.T @ (pts_r - O_T_femur_r).T).T
                    try:
                        print("det right thigh TCS:", float(
                            np.linalg.det(R_T_femur_r)))
                    except Exception:
                        pass
                    for nm, color, p0, p1 in axes_segments("Femur TCS (right)", O_T_femur_r, R_T_femur_r, axis_len_mm * 0.8):
                        fig.add_trace(go.Scatter3d(
                            x=[p0[0], p1[0]], y=[
                                p0[1], p1[1]], z=[p0[2], p1[2]],
                            mode="lines", line=dict(color=color, width=4), name=nm
                        ))
                    # Save right femur TCS template
                    try:
                        base = os.path.splitext(os.path.basename(c3d_path))[0]
                        out_dir = out_dir_trial
                        tplt_path = os.path.join(
                            out_dir, f"{base}_femur_right_tcs_template.npz")
                        used = [("R_Thigh_AS", R_thigh_AS), ("R_Thigh_PS", R_thigh_PS),
                                ("R_Thigh_AI", R_thigh_AI), ("R_Thigh_PI", R_thigh_PI)]
                        np.savez(
                            tplt_path,
                            marker_labels=np.array(
                                [name for name, P in used if P is not None], dtype=object),
                            origin_T=O_T_femur_r,
                            R_T_world=R_T_femur_r,
                            C_local=C_local_femur_r,
                            O_A_static=O_femur_right,
                            R_A_static=R_femur_right,
                        )
                        print(f"Saved right femur TCS template: {
                              os.path.abspath(tplt_path)}")
                    except Exception as e:
                        print(
                            f"Warning: could not save right femur TCS template: {e}")

            # Right tibia ACS: Y anterior, Z superior (proximal), X right
            R_ank_lat = pick("R_Ank_Lat", labels, xyz, frame_idx)
            R_ank_med = pick("R_Ank_Med", labels, xyz, frame_idx)
            if (R_ank_lat is not None) and (R_ank_med is not None):
                # Plot right ankle markers and line
                fig.add_trace(go.Scatter3d(
                    x=[R_ank_lat[0]], y=[R_ank_lat[1]], z=[R_ank_lat[2]],
                    mode="markers+text",
                    marker=dict(size=6, color="#8c564b"),
                    text=["R_Ank_Lat"], textposition="top center",
                    name="R_Ank_Lat"
                ))
                fig.add_trace(go.Scatter3d(
                    x=[R_ank_med[0]], y=[R_ank_med[1]], z=[R_ank_med[2]],
                    mode="markers+text",
                    marker=dict(size=6, color="#8c564b"),
                    text=["R_Ank_Med"], textposition="top center",
                    name="R_Ank_Med"
                ))
                fig.add_trace(go.Scatter3d(
                    x=[R_ank_med[0], R_ank_lat[0]], y=[R_ank_med[1],
                                                       R_ank_lat[1]], z=[R_ank_med[2], R_ank_lat[2]],
                    mode="lines", line=dict(color="#8c564b", width=3), name="right Ankle malleoli"
                ))

                R_AJC = 0.5 * (R_ank_med + R_ank_lat)
                joint_centers_dict.setdefault(
                    "R_AJC", np.asarray(R_AJC, dtype=float).copy())
                R_tibia_right = None
                O_tibia_right = None
                # Ensure we have right knee center; use knee markers above
                if (R_knee_lat is not None) and (R_knee_med is not None):
                    R_KJC = 0.5 * (R_knee_med + R_knee_lat)
                    try:
                        O_tibia_right, R_tibia_right, _ = tibia_acs_from_landmarks(
                            knee_center=R_KJC,
                            ankle_medial=R_ank_med,
                            ankle_lateral=R_ank_lat,
                            origin_at="knee",
                            anterior_hint=R_np[:, 1],
                            knee_medial=R_knee_med,
                            knee_lateral=R_knee_lat,
                        )
                    except ValueError:
                        R_tibia_right = None
                        O_tibia_right = None
                    if R_tibia_right is not None:
                        try:
                            if "R_femur_right" in locals():
                                print("right dot(Z_femur, Z_tibia) =", float(
                                    np.dot(R_femur_right[:, 2], R_tibia_right[:, 2])))
                        except Exception:
                            pass
                        for nm, color, p0, p1 in axes_segments("Tibia (right)", O_tibia_right, R_tibia_right, axis_len_mm):
                            fig.add_trace(go.Scatter3d(
                                x=[p0[0], p1[0]], y=[
                                    p0[1], p1[1]], z=[p0[2], p1[2]],
                                mode="lines", line=dict(color=color, width=5), name=nm
                            ))

                # Right ankle ACS: mirror left logic using the same routine and plotting
                R_calc = pick("R_Calc", labels, xyz, frame_idx)
                R_toe_med = pick("R_Toe_Med", labels, xyz, frame_idx)
                R_toe_lat = pick("R_Toe_Lat", labels, xyz, frame_idx)
                R_toe_tip = pick("R_Toe_Tip", labels, xyz, frame_idx)
                R_mid_sup = pick("R_Midfoot_Sup", labels, xyz, frame_idx)
                R_mid_lat = pick("R_Midfoot_Lat", labels, xyz, frame_idx)
                if (R_calc is not None) and (R_toe_med is not None) and (R_toe_lat is not None):
                    O_ankle_right, R_ankle_right, R_AJC = ankle_acs_from_foot_markers(
                        ankle_medial=R_ank_med,
                        ankle_lateral=R_ank_lat,
                        calcaneus=R_calc,
                        met1=R_toe_med,
                        met5=R_toe_lat,
                        met2=R_toe_tip,
                        tibia_z_hint=(
                            R_tibia_right[:, 2] if 'R_tibia_right' in locals() else None),
                        pelvis_x_hint=R_np[:, 0],
                    )
                    # Plot right foot markers the same way as left
                    foot_markers_r = {
                        "R_Calc": R_calc,
                        "R_Toe_Med": R_toe_med,
                        "R_Toe_Lat": R_toe_lat,
                    }
                    if R_toe_tip is not None:
                        foot_markers_r["R_Toe_Tip"] = R_toe_tip
                    if R_mid_sup is not None:
                        foot_markers_r["R_Midfoot_Sup"] = R_mid_sup
                    if R_mid_lat is not None:
                        foot_markers_r["R_Midfoot_Lat"] = R_mid_lat
                    for nm_pt, P in foot_markers_r.items():
                        fig.add_trace(go.Scatter3d(
                            x=[P[0]], y=[P[1]], z=[P[2]],
                            mode="markers+text",
                            marker=dict(size=6, color="#bcbd22"),
                            text=[nm_pt], textposition="top center",
                            name=nm_pt
                        ))
                    # Plot ankle triad at AJC
                    for nm, color, p0, p1 in axes_segments("ankle ISB JCS", O_ankle_right, R_ankle_right, axis_len_mm):
                        fig.add_trace(go.Scatter3d(
                            x=[p0[0], p1[0]], y=[
                                p0[1], p1[1]], z=[p0[2], p1[2]],
                            mode="lines", line=dict(color=color, width=5), name=nm
                        ))

                # Right shank TCS
                R_shank_AS = pick("R_Shank_AS", labels, xyz, frame_idx)
                R_shank_PS = pick("R_Shank_PS", labels, xyz, frame_idx)
                R_shank_AI = pick("R_Shank_AI", labels, xyz, frame_idx)
                R_shank_PI = pick("R_Shank_PI", labels, xyz, frame_idx)
                shank_cluster_r = [P for P in (
                    R_shank_AS, R_shank_PS, R_shank_AI, R_shank_PI) if P is not None]
                right_shank_points = [("R_Shank_AS", R_shank_AS), ("R_Shank_PS", R_shank_PS), (
                    "R_Shank_AI", R_shank_AI), ("R_Shank_PI", R_shank_PI)]
                for name, P in right_shank_points:
                    if P is not None:
                        fig.add_trace(go.Scatter3d(
                            x=[P[0]], y=[P[1]], z=[P[2]],
                            mode="markers+text",
                            marker=dict(size=5, color="#17becf"),
                            text=[name], textposition="top center",
                            name=name,
                            showlegend=False
                        ))
                if len(shank_cluster_r) >= 3:
                    pts_sr = np.vstack(shank_cluster_r)
                    O_T_shank_r = pts_sr.mean(axis=0)
                    if (R_knee_lat is not None) and (R_knee_med is not None):
                        R_KJC = 0.5 * (R_knee_med + R_knee_lat)
                        R_T_shank_r = _segment_tcs_R_from_proximal_z(
                            R_KJC - R_AJC, R_np, pts_sr)
                        if R_T_shank_r is not None:
                            try:
                                print("det right shank TCS:", float(
                                    np.linalg.det(R_T_shank_r)))
                            except Exception:
                                pass
                            C_local_shank_r = (
                                R_T_shank_r.T @ (pts_sr - O_T_shank_r).T).T
                            for nm, color, p0, p1 in axes_segments("Tibia TCS (right)", O_T_shank_r, R_T_shank_r, axis_len_mm * 0.8):
                                fig.add_trace(go.Scatter3d(
                                    x=[p0[0], p1[0]], y=[
                                        p0[1], p1[1]], z=[p0[2], p1[2]],
                                    mode="lines", line=dict(color=color, width=4), name=nm
                                ))
                            if R_tibia_right is not None:
                                try:
                                    base = os.path.splitext(
                                        os.path.basename(c3d_path))[0]
                                    out_dir = out_dir_trial
                                    tplt_path = os.path.join(
                                        out_dir, f"{base}_tibia_right_tcs_template.npz")
                                    used = [("R_Shank_AS", R_shank_AS), ("R_Shank_PS", R_shank_PS), (
                                        "R_Shank_AI", R_shank_AI), ("R_Shank_PI", R_shank_PI)]
                                    np.savez(
                                        tplt_path,
                                        marker_labels=np.array(
                                            [name for name, P in used if P is not None], dtype=object),
                                        origin_T=O_T_shank_r,
                                        R_T_world=R_T_shank_r,
                                        C_local=C_local_shank_r,
                                        O_A_static=O_tibia_right,
                                        R_A_static=R_tibia_right,
                                    )
                                    print(f"Saved right tibia TCS template: {
                                          os.path.abspath(tplt_path)}")
                                except Exception as e:
                                    print(
                                        f"Warning: could not save right tibia TCS template: {e}")

                # Right foot TCS: Z superior, Y anterior, X lateral
                R_calc = pick("R_Calc", labels, xyz, frame_idx)
                R_mid_sup = pick("R_Midfoot_Sup", labels, xyz, frame_idx)
                R_mid_lat = pick("R_Midfoot_Lat", labels, xyz, frame_idx)
                R_toe_med = pick("R_Toe_Med", labels, xyz, frame_idx)
                R_toe_lat = pick("R_Toe_Lat", labels, xyz, frame_idx)
                R_toe_tip = pick("R_Toe_Tip", labels, xyz, frame_idx)
                foot_cluster_r = [P for P in (
                    R_calc, R_mid_sup, R_mid_lat, R_toe_med, R_toe_lat, R_toe_tip) if P is not None]
                if len(foot_cluster_r) >= 3:
                    pts_fr = np.vstack(foot_cluster_r)
                    O_T_foot_r = pts_fr.mean(axis=0)
                    # Z from plantar plane; prefer calc + toe med/lat, else calc + midfoot sup/lat
                    Z_sup_r = None
                    if (R_calc is not None) and (R_toe_med is not None) and (R_toe_lat is not None):
                        v1 = R_toe_med - R_calc
                        v5 = R_toe_lat - R_calc
                        Z_sup_r = np.cross(v5, v1)
                    elif (R_calc is not None) and (R_mid_sup is not None) and (R_mid_lat is not None):
                        v1 = R_mid_sup - R_calc
                        v5 = R_mid_lat - R_calc
                        Z_sup_r = np.cross(v5, v1)
                    if Z_sup_r is None or np.linalg.norm(Z_sup_r) == 0:
                        _, R_seed_fr, _ = build_tcs_from_cluster(pts_fr)
                        Z_sup_r = R_seed_fr[:, 2]
                    Z_sup_r = Z_sup_r / np.linalg.norm(Z_sup_r)
                    if np.dot(Z_sup_r, np.array([0.0, 0.0, 1.0])) < 0:
                        Z_sup_r = -Z_sup_r
                    # Y anterior: calc -> toe tip or midpoint toes, projected off Z
                    if (R_calc is not None) and (R_toe_tip is not None):
                        yraw = R_toe_tip - R_calc
                    elif (R_calc is not None) and (R_toe_med is not None) and (R_toe_lat is not None):
                        yraw = 0.5 * (R_toe_med + R_toe_lat) - R_calc
                    else:
                        _, R_seed_fr2, _ = build_tcs_from_cluster(pts_fr)
                        yraw = R_seed_fr2[:, 1]
                    Y_ant_r = yraw - Z_sup_r * float(np.dot(yraw, Z_sup_r))
                    if np.linalg.norm(Y_ant_r) == 0:
                        Y_ant_r = np.array([0.0, 1.0, 0.0])
                    Y_ant_r = Y_ant_r / np.linalg.norm(Y_ant_r)
                    if np.dot(Y_ant_r, R_np[:, 1]) < 0:
                        Y_ant_r = -Y_ant_r
                    # X lateral on right: X = Y × Z
                    X_lat_r = np.cross(Y_ant_r, Z_sup_r)
                    if np.linalg.norm(X_lat_r) > 0:
                        X_lat_r = X_lat_r / np.linalg.norm(X_lat_r)
                        R_T_foot_r = np.column_stack(
                            [X_lat_r, Y_ant_r, Z_sup_r])
                        C_local_foot_r = (
                            R_T_foot_r.T @ (pts_fr - O_T_foot_r).T).T
                        for nm, color, p0, p1 in axes_segments("Foot TCS (right)", O_T_foot_r, R_T_foot_r, axis_len_mm * 0.8):
                            fig.add_trace(go.Scatter3d(
                                x=[p0[0], p1[0]], y=[
                                    p0[1], p1[1]], z=[p0[2], p1[2]],
                                mode="lines", line=dict(color=color, width=4), name=nm
                            ))
                        # Save right foot TCS template
                        try:
                            base = os.path.splitext(
                                os.path.basename(c3d_path))[0]
                            out_dir = out_dir_trial
                            tplt_path = os.path.join(
                                out_dir, f"{base}_foot_right_tcs_template.npz")
                            used = [("R_Calc", R_calc), ("R_Midfoot_Sup", R_mid_sup), ("R_Midfoot_Lat", R_mid_lat), (
                                "R_Toe_Med", R_toe_med), ("R_Toe_Lat", R_toe_lat), ("R_Toe_Tip", R_toe_tip)]
                            np.savez(
                                tplt_path,
                                marker_labels=np.array(
                                    [name for name, P in used if P is not None], dtype=object),
                                origin_T=O_T_foot_r,
                                R_T_world=R_T_foot_r,
                                C_local=C_local_foot_r,
                                O_A_static=O_ankle_right,
                                R_A_static=R_ankle_right,
                            )
                            print(f"Saved right foot TCS template: {
                                  os.path.abspath(tplt_path)}")
                        except Exception as e:
                            print(
                                f"Warning: could not save right foot TCS template: {e}")
        # Tibia ACS at ankle using both malleoli if present
        ank_lat = pick(pref + "Ank_Lat", labels, xyz, frame_idx)
        ank_med = pick(pref + "Ank_Med", labels, xyz, frame_idx)
        if (ank_lat is None) or (ank_med is None):
            print(f"Ankle markers not found: {pref}Ank_Lat or {pref}Ank_Med")
        else:
            # Plot ankle markers and line
            fig.add_trace(go.Scatter3d(
                x=[ank_lat[0]], y=[ank_lat[1]], z=[ank_lat[2]],
                mode="markers+text",
                marker=dict(size=6, color="#8c564b"),
                text=[pref + "Ank_Lat"], textposition="top center",
                name=pref + "Ank_Lat"
            ))
            fig.add_trace(go.Scatter3d(
                x=[ank_med[0]], y=[ank_med[1]], z=[ank_med[2]],
                mode="markers+text",
                marker=dict(size=6, color="#8c564b"),
                text=[pref + "Ank_Med"], textposition="top center",
                name=pref + "Ank_Med"
            ))
            fig.add_trace(go.Scatter3d(
                x=[ank_med[0], ank_lat[0]], y=[ank_med[1],
                                               ank_lat[1]], z=[ank_med[2], ank_lat[2]],
                mode="lines", line=dict(color="#8c564b", width=3), name="left Ankle malleoli"
            ))

            # Knee center from femur landmarks
            KJC = 0.5 * (knee_med + knee_lat)
            joint_centers_dict.setdefault(
                "L_KJC", np.asarray(KJC, dtype=float).copy())
            O_tibia, R_tibia, L_AJC = tibia_acs_from_landmarks(
                knee_center=KJC,
                ankle_medial=ank_med,
                ankle_lateral=ank_lat,
                origin_at="knee",
                anterior_hint=R_np[:, 1],
                knee_medial=knee_med,
                knee_lateral=knee_lat,
            )
            for nm, color, p0, p1 in axes_segments("Tibia (left)", O_tibia, R_tibia, axis_len_mm):
                fig.add_trace(go.Scatter3d(
                    x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
                    mode="lines", line=dict(color=color, width=5), name=nm
                ))

            # Check left tibia vs femur Z alignment in static
            try:
                zf = R_femur[:, 2]
                zt = R_tibia[:, 2]
                print("left dot(Z_femur, Z_tibia) =", float(np.dot(zf, zt)))
            except Exception:
                pass

            # Foot/Ankle ACS (requires calcaneus and toe markers)
            calc = pick(pref + "Calc", labels, xyz, frame_idx)
            toe_lat = pick(pref + "Toe_Lat", labels, xyz, frame_idx)
            toe_med = pick(pref + "Toe_Med", labels, xyz, frame_idx)
            toe_tip = pick(pref + "Toe_Tip", labels, xyz, frame_idx)
            mid_sup = pick(pref + "Midfoot_Sup", labels, xyz, frame_idx)
            mid_lat = pick(pref + "Midfoot_Lat", labels, xyz, frame_idx)
            # Use toe medial/lateral as proxies for 1st/5th met heads; toe tip as 2nd met proxy
            if (calc is not None) and (toe_med is not None) and (toe_lat is not None):
                O_ankle, R_ankle, L_AJC = ankle_acs_from_foot_markers(
                    ankle_medial=ank_med,
                    ankle_lateral=ank_lat,
                    calcaneus=calc,
                    met1=toe_med,
                    met5=toe_lat,
                    met2=toe_tip,
                    tibia_z_hint=R_tibia[:, 2],
                    pelvis_x_hint=R_np[:, 0],
                )
                joint_centers_dict.setdefault(
                    "L_AJC", np.asarray(L_AJC, dtype=float).copy())
                # Plot foot markers
                foot_markers = {
                    pref + "Calc": calc,
                    pref + "Toe_Med": toe_med,
                    pref + "Toe_Lat": toe_lat,
                }
                if toe_tip is not None:
                    foot_markers[pref + "Toe_Tip"] = toe_tip
                if mid_sup is not None:
                    foot_markers[pref + "Midfoot_Sup"] = mid_sup
                if mid_lat is not None:
                    foot_markers[pref + "Midfoot_Lat"] = mid_lat
                for nm_pt, P in foot_markers.items():
                    fig.add_trace(go.Scatter3d(
                        x=[P[0]], y=[P[1]], z=[P[2]],
                        mode="markers+text",
                        marker=dict(size=6, color="#bcbd22"),
                        text=[nm_pt], textposition="top center",
                        name=nm_pt
                    ))
                # Plot ankle triad at AJC
                for nm, color, p0, p1 in axes_segments("ankle ISB JCS", O_ankle, R_ankle, axis_len_mm):
                    fig.add_trace(go.Scatter3d(
                        x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]],
                        mode="lines", line=dict(color=color, width=5), name=nm
                    ))

                # Build tibia->foot neutral relative from this static frame
                try:
                    R_ti_static = _enforce_right_handed(R_tibia)
                    R_fo_static = _enforce_right_handed(R_ankle)
                    # Align X axes (paired flip on foot X&Z) for consistent polarity
                    if float(np.dot(R_ti_static[:, 0], R_fo_static[:, 0])) < 0.0:
                        R_fo_static[:, [0, 2]] *= -1.0
                    R_rel_neutral_raw = R_ti_static.T @ R_fo_static
                    R_neutral_left = R_rel_neutral_raw
                    # Sanity checks
                    try:
                        print("||R_neutral_left - I||_F =",
                              float(np.linalg.norm(R_neutral_left - np.eye(3))))
                        print("neutral (L) Euler XYZ:",
                              _euler_xyz_from_R(R_neutral_left))
                    except Exception:
                        pass
                except Exception:
                    pass

                # Save tibia TCS template with static offsets (if TCS was built above)
                if (
                    "O_T_tibia" in locals()
                    and locals().get("R_T_tibia") is not None
                    and locals().get("C_local_tibia") is not None
                ):
                    try:
                        base = os.path.splitext(os.path.basename(c3d_path))[0]
                        out_dir = out_dir_trial
                        tplt_path = os.path.join(
                            out_dir, f"{base}_tibia_tcs_template.npz")
                        used = [("L_Shank_AS", L_shank_AS), ("L_Shank_PS", L_shank_PS),
                                ("L_Shank_AI", L_shank_AI), ("L_Shank_PI", L_shank_PI)]
                        R_TA = R_T_tibia.T @ R_tibia
                        t_TA = R_T_tibia.T @ (O_tibia - O_T_tibia)
                        R_AT = R_tibia.T @ R_T_tibia
                        t_AT = R_tibia.T @ (O_T_tibia - O_tibia)
                        np.savez(
                            tplt_path,
                            marker_labels=np.array(
                                [name for name, P in used if P is not None], dtype=object),
                            origin_T=O_T_tibia,
                            R_T_world=R_T_tibia,
                            C_local=C_local_tibia,
                            O_A_static=O_tibia,
                            R_A_static=R_tibia,
                            R_TA=R_TA,
                            t_TA=t_TA,
                            R_AT=R_AT,
                            t_AT=t_AT,
                        )
                        print(f"Saved tibia TCS template: {
                              os.path.abspath(tplt_path)}")
                        print("Tibia (L) static A-in-T: R_TA=\n",
                              R_TA, "\n t_TA=", t_TA)
                    except Exception as e:
                        print(
                            f"Warning: could not save tibia TCS template: {e}")

                # Save foot TCS template with static offsets (if TCS was built above)
                if (
                    "O_T_foot" in locals()
                    and locals().get("R_T_foot") is not None
                    and locals().get("C_local_foot") is not None
                ):
                    try:
                        base = os.path.splitext(os.path.basename(c3d_path))[0]
                        out_dir = out_dir_trial
                        tplt_path = os.path.join(
                            out_dir, f"{base}_foot_tcs_template.npz")
                        used = [
                            ("L_Calc", L_calc), ("L_Midfoot_Sup",
                                                 L_mid_sup), ("L_Midfoot_Lat", L_mid_lat),
                            ("L_Toe_Med", toe_med), ("L_Toe_Lat",
                                                     toe_lat), ("L_Toe_Tip", toe_tip)
                        ]
                        R_TA = R_T_foot.T @ R_ankle
                        t_TA = R_T_foot.T @ (O_ankle - O_T_foot)
                        R_AT = R_ankle.T @ R_T_foot
                        t_AT = R_ankle.T @ (O_T_foot - O_ankle)
                        # Constant TCS->ACS rotation per static: C = R_A @ R_T^T
                        C_TCS_to_ACS = R_ankle @ R_T_foot.T
                        np.savez(
                            tplt_path,
                            marker_labels=np.array(
                                [name for name, P in used if P is not None], dtype=object),
                            origin_T=O_T_foot,
                            R_T_world=R_T_foot,
                            C_local=C_local_foot,
                            O_A_static=O_ankle,
                            R_A_static=R_ankle,
                            R_TA=R_TA,
                            t_TA=t_TA,
                            R_AT=R_AT,
                            t_AT=t_AT,
                            C_TCS_to_ACS=C_TCS_to_ACS,
                        )
                        print(f"Saved foot TCS template: {
                              os.path.abspath(tplt_path)}")
                        print("Foot/Ankle (L) static A-in-T: R_TA=\n",
                              R_TA, "\n t_TA=", t_TA)
                        try:
                            print("Foot (L) C_TCS_to_ACS:\n", C_TCS_to_ACS)
                        except Exception:
                            pass
                    except Exception as e:
                        print(
                            f"Warning: could not save foot TCS template: {e}")

    # --- Static joint angles (ISB-aligned XYZ Cardan; printed in degrees) ---
    angle_lines = []
    # Left-side proximal mirror for X-axis (lateral) per ISB left conventions
    S_L = np.diag([-1.0, 1.0, 1.0])

    # Debug: determinants and proximal-distal X alignment
    try:
        print("det pelvis:", float(np.linalg.det(R_np)))
    except Exception:
        pass
    if 'R_femur' in locals():
        try:
            print("det femur:", float(np.linalg.det(R_femur)))
            dot_x = float(np.dot((R_np @ S_L)[:, 0], R_femur[:, 0]))
            print("dot(X_p*, X_f)=", dot_x)
        except Exception:
            pass

    # Hip (left): ISB JCS using ACS (matches angles_only2.py)
    if 'R_femur' in locals():
        FE, AbAd, IE = hip_angles_isb(
            np.array(R_np, float), np.array(R_femur, float))
        print(
            f"Hip (L) ISB JCS [FE, Ab/Ad, IE]: ({FE:.1f}, {AbAd:.1f}, {IE:.1f}) deg")
        angle_lines.append(f"Hip (L) ISB JCS: {FE:.1f}, {
                           AbAd:.1f}, {IE:.1f} deg")
        # Save static hip offsets for dynamics zeroing (per-channel mean subtraction)
        try:
            base = os.path.splitext(os.path.basename(c3d_path))[0]
            out_dir = out_dir_trial
            offsets_path = os.path.join(out_dir, f"{base}_hip_offsets.npz")
            np.savez(offsets_path, left=np.array([FE, AbAd, IE], float))
            print(f"Saved hip static offsets (left) to: {
                  os.path.abspath(offsets_path)}")
        except Exception as e:
            print(f"Warning: could not save hip offsets: {e}")

    # Knee: Grood–Suntay JCS (match angles_only2.py logic)
    if 'R_femur' in locals() and 'R_tibia' in locals():
        knee_FE, knee_VarVal, knee_IE = knee_angles_grood_suntay(
            R_femur, R_tibia, side="left")
        print(f"Knee (L) Grood–Suntay [FE, Var/Val, IE]: ({
              knee_FE:.1f}, {knee_VarVal:.1f}, {knee_IE:.1f}) deg")
        angle_lines.append(f"Knee (L) GS: {knee_FE:.1f}, {
                           knee_VarVal:.1f}, {knee_IE:.1f} deg")
        # Save static GS offsets for dynamics zeroing (per-channel mean subtraction)
        try:
            base = os.path.splitext(os.path.basename(c3d_path))[0]
            out_dir = out_dir_trial
            offsets_path = os.path.join(out_dir, f"{base}_knee_gs_offsets.npz")
            # Store left (and optionally right later) in one file
            np.savez(offsets_path, left=np.array(
                [knee_FE, knee_VarVal, knee_IE], float))
            print(f"Saved knee GS static offsets (left) to: {
                  os.path.abspath(offsets_path)}")
        except Exception as e:
            print(f"Warning: could not save knee GS offsets: {e}")

    # --- Left ankle angles with neutral-comp and clean signs ---
    if "R_tibia" in locals() and "R_ankle" in locals():
        R_ti = _enforce_right_handed(R_tibia)
        R_fo = _ankle_raw_R_for_euler(labels, xyz, frame_idx, "left", R_ankle)

        # 1) Align X axes: if opposite, pair-flip foot X&Z (keeps det=+1)
        if float(np.dot(R_ti[:, 0], R_fo[:, 0])) < 0.0:
            R_fo[:, [0, 2]] *= -1.0

        # Debug: dot products of aligned axes
        try:
            print("dot X:", float(np.dot(R_ti[:, 0], R_fo[:, 0])))
            print("dot Y:", float(np.dot(R_ti[:, 1], R_fo[:, 1])))
            print("dot Z:", float(np.dot(R_ti[:, 2], R_fo[:, 2])))
        except Exception:
            pass

        R_rel_raw = R_ti.T @ R_fo
        # Axis-angle of raw relative rotation
        try:
            tr = float(np.trace(R_rel_raw))
            angle = float(np.degrees(
                np.arccos(max(-1.0, min(1.0, 0.5 * (tr - 1.0))))))
            axis = np.array([
                R_rel_raw[2, 1] - R_rel_raw[1, 2],
                R_rel_raw[0, 2] - R_rel_raw[2, 0],
                R_rel_raw[1, 0] - R_rel_raw[0, 1],
            ], dtype=float)
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            print("axis-angle:", axis, angle)
        except Exception:
            pass
        R_rel = R_rel_raw

        # 3) Euler XYZ + polarity + unwrap
        # X=dorsi/plantar, Y=inv/ev, Z=abd/add
        ax, ay, az = _euler_xyz_from_R(R_rel)

        # Small unwrapping so values live near 0 rather than ±180
        if ax < -90.0:
            ax += 180.0
        elif ax > 90.0:
            ax -= 180.0
        # Convention tweak: flip left ankle Z sign
        az = -az

        print(
            f"Ankle (L) XYZ [X=Dorsi/Plantar, Y=Inv/Ev, Z=Abd/Add]: ({ax:.1f}, {ay:.1f}, {az:.1f}) deg")
        angle_lines.append(f"Ankle (L) XYZ: {ax:.1f}, {ay:.1f}, {az:.1f} deg")

    # Right side angles (if available)
    if 'R_femur_right' in locals():
        FE_r, AbAd_r, IE_r = hip_angles_isb(
            np.array(R_np, float), np.array(R_femur_right, float))
        print(
            f"Hip (R) ISB JCS [FE, Ab/Ad, IE]: ({FE_r:.1f}, {AbAd_r:.1f}, {IE_r:.1f}) deg")
        angle_lines.append(f"Hip (R) ISB JCS: {FE_r:.1f}, {
                           AbAd_r:.1f}, {IE_r:.1f} deg")

    # Right knee: Grood–Suntay JCS (mirror left-side approach)
    if ("R_femur_right" in locals()) and (locals().get("R_tibia_right") is not None):
        knee_FE_r, knee_VarVal_r, knee_IE_r = knee_angles_grood_suntay(
            R_femur_right, R_tibia_right, side="right")
        # Optional: match left display polarity if desired (angles_only2.py flips FE, VarVal)
        knee_FE_r = -knee_FE_r
        knee_VarVal_r = -knee_VarVal_r
        print(f"Knee (R) Grood–Suntay [FE, Var/Val, IE]: ({
              knee_FE_r:.1f}, {knee_VarVal_r:.1f}, {knee_IE_r:.1f}) deg")
        angle_lines.append(f"Knee (R) GS: {knee_FE_r:.1f}, {
                           knee_VarVal_r:.1f}, {knee_IE_r:.1f} deg")

    # --- Right ankle angles with neutral-comp and clean signs (mirror left logic) ---
    if "R_tibia_right" in locals() and "R_ankle_right" in locals():
        R_ti_r = _enforce_right_handed(R_tibia_right)
        R_fo_r = _ankle_raw_R_for_euler(
            labels, xyz, frame_idx, "right", R_ankle_right)

        # Align X axes: if opposite, pair-flip foot X&Z (keeps det=+1)
        if float(np.dot(R_ti_r[:, 0], R_fo_r[:, 0])) < 0.0:
            R_fo_r[:, [0, 2]] *= -1.0

        # Relative rotation tibia->foot (right)
        R_rel_raw_r = R_ti_r.T @ R_fo_r

        # Euler XYZ + polarity + unwrap (same as left)
        # X=dorsi/plantar, Y=inv/ev, Z=abd/add
        ax_r, ay_r, az_r = _euler_xyz_from_R(R_rel_raw_r)
        if ax_r < -90.0:
            ax_r += 180.0
        elif ax_r > 90.0:
            ax_r -= 180.0
        # Convention enforcement (both sides):
        # - Dorsiflexion (X) positive on both sides  -> keep ax_r as-is
        # - Inversion (Y) positive on both sides     -> flip sign on right side
        # - Abduction/Adduction (Z) not specified    -> leave as-is
        ay_r = -ay_r

        print(f"Ankle (R) XYZ [X=Dorsi/Plantar, Y=Inv/Ev, Z=Abd/Add]: ({
              ax_r:.1f}, {ay_r:.1f}, {az_r:.1f}) deg")
        angle_lines.append(f"Ankle (R) XYZ: {ax_r:.1f}, {
                           ay_r:.1f}, {az_r:.1f} deg")

    # Overlay the angles as a text box in the HTML
    if angle_lines:
        fig.add_annotation(
            xref="paper", yref="paper",
            x=1.0, y=1.0,
            xanchor="right", yanchor="top",
            showarrow=False,
            align="left",
            bordercolor="rgba(150,150,150,0.6)",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.85)",
            font=dict(size=12),
            text="<br>".join(angle_lines),
        )

    # --- Plot all lower-body markers (left/right) with corresponding colors ---
    try:
        group_to_labels_L = {
            "thigh": ["L_Thigh_AS", "L_Thigh_PS", "L_Thigh_AI", "L_Thigh_PI"],
            "shank": ["L_Shank_AS", "L_Shank_PS", "L_Shank_AI", "L_Shank_PI"],
            "knee": ["L_Knee_Lat", "L_Knee_Med"],
            "ankle": ["L_Ank_Lat", "L_Ank_Med"],
            "foot": ["L_Calc", "L_Midfoot_Sup", "L_Midfoot_Lat", "L_Toe_Med", "L_Toe_Lat", "L_Toe_Tip"],
        }
        group_to_labels_R = {
            "thigh": ["R_Thigh_AS", "R_Thigh_PS", "R_Thigh_AI", "R_Thigh_PI"],
            "shank": ["R_Shank_AS", "R_Shank_PS", "R_Shank_AI", "R_Shank_PI"],
            "knee": ["R_Knee_Lat", "R_Knee_Med"],
            "ankle": ["R_Ank_Lat", "R_Ank_Med"],
            "foot": ["R_Calc", "R_Midfoot_Sup", "R_Midfoot_Lat", "R_Toe_Med", "R_Toe_Lat", "R_Toe_Tip"],
        }
        group_color = {
            "thigh": "#9467bd",  # purple
            "shank": "#17becf",  # teal
            "knee": "#ff7f0e",   # orange
            "ankle": "#8c564b",  # brown
            "foot": "#bcbd22",   # olive
        }
        # Helper to add markers for a side

        def _plot_marker_group(side_labels: dict[str, list[str]]):
            for grp, names in side_labels.items():
                color = group_color.get(grp, "#7f7f7f")
                for nm_pt in names:
                    P = pick(nm_pt, labels, xyz, frame_idx)
                    if P is None:
                        continue
                    fig.add_trace(go.Scatter3d(
                        x=[P[0]], y=[P[1]], z=[P[2]],
                        mode="markers+text",
                        marker=dict(size=5, color=color),
                        text=[nm_pt], textposition="top center",
                        name=nm_pt,
                        showlegend=False,
                    ))
        _plot_marker_group(group_to_labels_L)
        _plot_marker_group(group_to_labels_R)
    except Exception:
        pass

    fig.update_scenes(
        aspectmode="data",
        xaxis_title="X (mm)", yaxis_title="Y (mm)", zaxis_title="Z (mm)",
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.2))
    )
    try:
        _base = os.path.splitext(os.path.basename(c3d_path))[0]
        _subject = os.path.basename(os.path.dirname(c3d_path))
        _title_txt = f"{_subject} - {_base}"
    except Exception:
        _title_txt = "Static calibration"
    fig.update_layout(title=_title_txt, legend=dict(itemsizing="constant"))

    if out_html is None:
        base = os.path.splitext(os.path.basename(c3d_path))[0]
        # Write HTML into the per-trial folder
        out_html = os.path.join(
            out_dir_trial, f"{base}_left_static_calibration.html")

    # Export joint centers to CSV (lab coordinates, mm; static frame)
    try:
        base = os.path.splitext(os.path.basename(c3d_path))[0]
        jc_path = os.path.join(out_dir_trial, f"{base}_joint_centers.csv")
        order = ["RHJC", "LHJC", "L_KJC", "L_AJC", "R_KJC", "R_AJC"]
        with open(jc_path, "w", newline="", encoding="utf-8") as f:
            f.write("name,x_mm,y_mm,z_mm\n")
            for name in order:
                if name in joint_centers_dict:
                    p = joint_centers_dict[name]
                    f.write(f"{name},{p[0]:.6f},{p[1]:.6f},{p[2]:.6f}\n")
        print(f"Wrote joint centers: {os.path.abspath(jc_path)}")
    except Exception as e:
        print(f"Warning: could not save joint centers CSV: {e}")

    try:
        print("Harrington PW_mm, PD_mm:", float(
            dims.get("PW_mm", float("nan"))), float(dims.get("PD_mm", float("nan"))))
    except Exception:
        pass
    pio.write_html(fig, file=out_html, include_plotlyjs="cdn", auto_open=True)
    print(f"Wrote {out_html}")


if __name__ == "__main__":
    # For direct module execution, fall back to CLI wrapper
    try:
        from static_calibration_cli import main as _cli_main
        _cli_main()
    except Exception:
        # Minimal fallback: use a default path
        c3d_path = r"C:/Users/lmcam/Documents/Grad project/c3d/subject 02/S_Cal02.c3d"
