# -*- coding: utf-8 -*-
"""
inertial_segments

Load joint centers exported from static_calibration.py (CSV: name, x_mm, y_mm, z_mm)
and compute segment lengths + Winter inertial parameters from static trial data.

  - Thigh/femur length = hip joint center → knee joint center
  - Shank/tibia length = knee joint center → ankle joint center
  - Foot length = heel (calcaneus) → toe marker (or ankle → toe projection)

This script:
  1. Computes segment mass (Winter fractions; mass_kg per segment).
  2. Defines COM location in lab and in segment ACS (origin at proximal joint, Z along segment).
  3. Defines inertia tensor in segment ACS (diagonal, at COM; I_zz = m*k² longitudinal).
  4. Confirms segment origins are at joint centers (hip, knee, ankle where applicable).

@author: lmcam
"""

from __future__ import annotations

import os
import numpy as np

try:
    import ezc3d
    HAVE_EZC3D = True
except ImportError:
    HAVE_EZC3D = False


def load_joint_centers(csv_path: str) -> dict[str, np.ndarray]:
    """
    Load joint centers from the static calibration export CSV.

    Parameters
    ----------
    csv_path : str
        Path to the *_joint_centers.csv file.

    Returns
    -------
    dict[str, np.ndarray]
        Maps joint name (e.g. "RHJC", "LHJC", "L_KJC", "L_AJC", "R_KJC", "R_AJC")
        to position in lab frame, shape (3,), units mm.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Joint centers CSV not found: {csv_path}")

    out = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        return out
    header = lines[0].lower()
    if "name" not in header or "x_mm" not in header:
        raise ValueError(
            f"Expected header 'name,x_mm,y_mm,z_mm' in {
                csv_path}; got: {lines[0]}"
        )
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 4:
            continue
        name = parts[0]
        try:
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        except ValueError:
            continue
        out[name] = np.array([x, y, z], dtype=float)
    return out


def segment_endpoints(joint_centers: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Return proximal and distal endpoints for each lower-body segment from joint centers.

    Parameters
    ----------
    joint_centers : dict[str, np.ndarray]
        From load_joint_centers(); keys e.g. RHJC, LHJC, L_KJC, L_AJC, R_KJC, R_AJC.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        Segment name -> (proximal_pt_mm, distal_pt_mm) in lab frame.
        Keys: "pelvis" (mid-hip to mid-hip), "L_thigh", "L_shank", "R_thigh", "R_shank".
    """
    segments = {}
    # Pelvis: left HJC to right HJC (proximal = LHJC, distal = RHJC for a vector; or use mid-point as reference)
    if "LHJC" in joint_centers and "RHJC" in joint_centers:
        segments["pelvis"] = (joint_centers["LHJC"].copy(),
                              joint_centers["RHJC"].copy())
    # Left thigh: hip to knee
    if "LHJC" in joint_centers and "L_KJC" in joint_centers:
        segments["L_thigh"] = (joint_centers["LHJC"].copy(),
                               joint_centers["L_KJC"].copy())
    # Left shank: knee to ankle
    if "L_KJC" in joint_centers and "L_AJC" in joint_centers:
        segments["L_shank"] = (joint_centers["L_KJC"].copy(),
                               joint_centers["L_AJC"].copy())
    # Right thigh: hip to knee
    if "RHJC" in joint_centers and "R_KJC" in joint_centers:
        segments["R_thigh"] = (joint_centers["RHJC"].copy(),
                               joint_centers["R_KJC"].copy())
    # Right shank: knee to ankle
    if "R_KJC" in joint_centers and "R_AJC" in joint_centers:
        segments["R_shank"] = (joint_centers["R_KJC"].copy(),
                               joint_centers["R_AJC"].copy())
    return segments


def segment_lengths(
    joint_centers: dict[str, np.ndarray],
) -> dict[str, float]:
    """
    Compute segment lengths (mm) from joint centers.

    - Thigh = hip JC → knee JC
    - Shank = knee JC → ankle JC
    - Pelvis width = distance between hip JCs

    Parameters
    ----------
    joint_centers : dict[str, np.ndarray]
        From load_joint_centers(); keys RHJC, LHJC, L_KJC, L_AJC, R_KJC, R_AJC.

    Returns
    -------
    dict[str, float]
        Keys: L_thigh_mm, R_thigh_mm, L_shank_mm, R_shank_mm, pelvis_width_mm.
    """
    out: dict[str, float] = {}
    for name, (prox, dist) in segment_endpoints(joint_centers).items():
        if name == "pelvis":
            out["pelvis_width_mm"] = float(np.linalg.norm(dist - prox))
        else:
            out[f"{name}_mm"] = float(np.linalg.norm(dist - prox))
    return out


# ---------------------------------------------------------------------------
# Winter (Biomechanics and Motor Control of Human Movement) anthropometry
# ---------------------------------------------------------------------------
# Reference: Winter, D.A. Biomechanics and Motor Control of Human Movement,
# 4th ed. Wiley, 2009. (Typical values from anthropometric tables; approximate.)
#
# Thigh segment (both legs use same fractions):
#   - Mass fraction of total body mass: 0.100  (10%)
#   - Center of mass from proximal (hip) end: 0.433 of segment length
#   - Radius of gyration (about COM, longitudinal): 0.323 of segment length
#
# Shank/tibia segment (both legs use same fractions):
#   - Mass fraction of total body mass: 0.0465  (~4.65%)
#   - Center of mass from proximal (knee) end: 0.433 of segment length
#   - Radius of gyration (about COM, longitudinal): 0.302 of segment length
#
WINTER_THIGH_MASS_FRACTION = 0.100   # m_thigh / M_body
WINTER_THIGH_COM_FRACTION_PROXIMAL = 0.433   # COM from hip along segment
# k / L (radius of gyration about COM, as fraction of L)
WINTER_THIGH_K_FRACTION = 0.323

WINTER_SHANK_MASS_FRACTION = 0.0465   # m_shank / M_body
WINTER_SHANK_COM_FRACTION_PROXIMAL = 0.433   # COM from knee along segment
# k / L (radius of gyration about COM, as fraction of L)
WINTER_SHANK_K_FRACTION = 0.302

# Foot segment (heel → toe; both legs use same fractions):
#   - Mass fraction of total body mass: 0.0145  (~1.45%)
#   - Center of mass from proximal (heel) end: 0.44 of segment length
#   - Radius of gyration (about COM): 0.475 of segment length
#
WINTER_FOOT_MASS_FRACTION = 0.0145   # m_foot / M_body
# COM from heel along segment (heel → toe)
WINTER_FOOT_COM_FRACTION_PROXIMAL = 0.44
# k / L (radius of gyration about COM, as fraction of L)
WINTER_FOOT_K_FRACTION = 0.475


def _winter_proximal_distal_inertial(
    proximal_mm: np.ndarray,
    distal_mm: np.ndarray,
    mass_fraction: float,
    com_fraction_proximal: float,
    k_fraction: float,
    body_mass_kg: float,
    zero_length_message: str,
) -> dict[str, float | np.ndarray]:
    """
    Winter-style segment: length along proximal→distal, COM fraction from proximal,
    mass = fraction * body mass, radius of gyration = k_fraction * L (all mm except mass kg).
    """
    p = np.asarray(proximal_mm, dtype=float).reshape(3)
    d = np.asarray(distal_mm, dtype=float).reshape(3)
    vec = d - p
    L_mm = float(np.linalg.norm(vec))
    if L_mm < 1e-6:
        raise ValueError(zero_length_message)
    com_mm = p + com_fraction_proximal * vec
    mass_kg = mass_fraction * body_mass_kg
    radius_gyration_mm = k_fraction * L_mm
    return {
        "length_mm": L_mm,
        "mass_kg": mass_kg,
        "com_mm": np.asarray(com_mm, dtype=float),
        "radius_gyration_mm": radius_gyration_mm,
    }


def thigh_inertial_winter(
    joint_centers: dict[str, np.ndarray],
    body_mass_kg: float,
    side: str = "L",
) -> dict[str, float | np.ndarray]:
    """
    Thigh inertial parameters using Winter anthropometric fractions (both thighs).

    Reference: Winter, D.A. Biomechanics and Motor Control of Human Movement.

    Math steps (same for left and right; substitute LHJC/L_KJC or RHJC/R_KJC):

      1) Segment length (mm)
         L = ||KJC - HJC||
         (proximal = hip joint center, distal = knee joint center)

      2) Center of mass position (lab frame, mm)
         COM = HJC + (COM_fraction_proximal) * (KJC - HJC)
             = HJC + 0.433 * (KJC - HJC)
         So the COM lies 43.3% of the way from hip toward knee.

      3) Segment mass (kg)
         m_thigh = (mass_fraction) * M_body = 0.100 * M_body

      4) Radius of gyration (mm), about COM, longitudinal axis
         k = (k_fraction) * L = 0.323 * L
         (Used with m to get moment of inertia I = m * k^2 if needed.)

    Parameters
    ----------
    joint_centers : dict[str, np.ndarray]
        From load_joint_centers(); need LHJC, L_KJC for left, RHJC, R_KJC for right.
    body_mass_kg : float
        Total body mass (kg).
    side : str
        "L" or "R" for left/right thigh.

    Returns
    -------
    dict
        length_mm, mass_kg, com_mm (3,), radius_gyration_mm.
    """
    side = side.upper()
    if side == "L":
        HJC, KJC = joint_centers["LHJC"], joint_centers["L_KJC"]
    else:
        HJC, KJC = joint_centers["RHJC"], joint_centers["R_KJC"]
    return _winter_proximal_distal_inertial(
        HJC,
        KJC,
        WINTER_THIGH_MASS_FRACTION,
        WINTER_THIGH_COM_FRACTION_PROXIMAL,
        WINTER_THIGH_K_FRACTION,
        body_mass_kg,
        f"Thigh segment length near zero for side={side}",
    )


def _try_bilateral_winter_thigh_shank(
    winter_fn,
    joint_centers: dict[str, np.ndarray],
    body_mass_kg: float,
    left_key: str,
    right_key: str,
) -> dict[str, dict | None]:
    """Call ``winter_fn(..., side='L'|'R')`` for both legs; missing data → None for that side."""
    out: dict[str, dict | None] = {}
    for side, key in (("L", left_key), ("R", right_key)):
        try:
            out[key] = winter_fn(joint_centers, body_mass_kg, side=side)
        except (KeyError, ValueError):
            out[key] = None
    return out


def thigh_anthropometry_both_winter(
    joint_centers: dict[str, np.ndarray],
    body_mass_kg: float,
) -> dict[str, dict]:
    """
    Apply Winter anthropometric fractions to both thighs; returns all parameters and math summary.

    For each thigh we compute (see thigh_inertial_winter for formulas):
      - length_mm   = ||KJC - HJC||
      - com_mm      = HJC + 0.433 * (KJC - HJC)
      - mass_kg     = 0.100 * body_mass_kg
      - radius_gyration_mm = 0.323 * length_mm

    Parameters
    ----------
    joint_centers : dict[str, np.ndarray]
        From load_joint_centers().
    body_mass_kg : float
        Total body mass (kg).

    Returns
    -------
    dict
        "L_thigh" and "R_thigh" each with length_mm, mass_kg, com_mm, radius_gyration_mm;
        "math_steps" (str) summarizing the Winter formulas used.
    """
    math_steps = (
        "Winter (Biomechanics and Motor Control of Human Movement) thigh anthropometry:\n"
        "  L = ||KJC - HJC||  (segment length)\n"
        "  COM = HJC + 0.433 * (KJC - HJC)  (43.3% from hip toward knee)\n"
        "  m = 0.100 * M_body  (10% of body mass)\n"
        "  k = 0.323 * L  (radius of gyration about COM, as fraction of L)"
    )
    out: dict = {"math_steps": math_steps}
    out.update(_try_bilateral_winter_thigh_shank(
        thigh_inertial_winter, joint_centers, body_mass_kg, "L_thigh", "R_thigh"))
    return out


def shank_inertial_winter(
    joint_centers: dict[str, np.ndarray],
    body_mass_kg: float,
    side: str = "L",
) -> dict[str, float | np.ndarray]:
    """
    Shank (tibia) inertial parameters using Winter anthropometric fractions (both shanks).

    Reference: Winter, D.A. Biomechanics and Motor Control of Human Movement.

    Math steps (same for left and right; substitute L_KJC/L_AJC or R_KJC/R_AJC):

      1) Segment length (mm)
         L = ||AJC - KJC||
         (proximal = knee joint center, distal = ankle joint center)

      2) Center of mass position (lab frame, mm)
         COM = KJC + (COM_fraction_proximal) * (AJC - KJC)
             = KJC + 0.433 * (AJC - KJC)
         So the COM lies 43.3% of the way from knee toward ankle.

      3) Segment mass (kg)
         m_shank = (mass_fraction) * M_body = 0.0465 * M_body

      4) Radius of gyration (mm), about COM, longitudinal axis
         k = (k_fraction) * L = 0.302 * L

    Parameters
    ----------
    joint_centers : dict[str, np.ndarray]
        From load_joint_centers(); need L_KJC, L_AJC for left, R_KJC, R_AJC for right.
    body_mass_kg : float
        Total body mass (kg).
    side : str
        "L" or "R" for left/right shank.

    Returns
    -------
    dict
        length_mm, mass_kg, com_mm (3,), radius_gyration_mm.
    """
    side = side.upper()
    if side == "L":
        KJC, AJC = joint_centers["L_KJC"], joint_centers["L_AJC"]
    else:
        KJC, AJC = joint_centers["R_KJC"], joint_centers["R_AJC"]
    return _winter_proximal_distal_inertial(
        KJC,
        AJC,
        WINTER_SHANK_MASS_FRACTION,
        WINTER_SHANK_COM_FRACTION_PROXIMAL,
        WINTER_SHANK_K_FRACTION,
        body_mass_kg,
        f"Shank segment length near zero for side={side}",
    )


def shank_anthropometry_both_winter(
    joint_centers: dict[str, np.ndarray],
    body_mass_kg: float,
) -> dict[str, dict]:
    """
    Apply Winter anthropometric fractions to both shanks; returns all parameters and math summary.

    For each shank we compute (see shank_inertial_winter for formulas):
      - length_mm   = ||AJC - KJC||
      - com_mm      = KJC + 0.433 * (AJC - KJC)
      - mass_kg     = 0.0465 * body_mass_kg
      - radius_gyration_mm = 0.302 * length_mm

    Parameters
    ----------
    joint_centers : dict[str, np.ndarray]
        From load_joint_centers().
    body_mass_kg : float
        Total body mass (kg).

    Returns
    -------
    dict
        "L_shank" and "R_shank" each with length_mm, mass_kg, com_mm, radius_gyration_mm;
        "math_steps" (str) summarizing the Winter formulas used.
    """
    math_steps = (
        "Winter (Biomechanics and Motor Control of Human Movement) shank anthropometry:\n"
        "  L = ||AJC - KJC||  (segment length, knee to ankle)\n"
        "  COM = KJC + 0.433 * (AJC - KJC)  (43.3% from knee toward ankle)\n"
        "  m = 0.0465 * M_body  (~4.65% of body mass)\n"
        "  k = 0.302 * L  (radius of gyration about COM, as fraction of L)"
    )
    out: dict = {"math_steps": math_steps}
    out.update(_try_bilateral_winter_thigh_shank(
        shank_inertial_winter, joint_centers, body_mass_kg, "L_shank", "R_shank"))
    return out


def foot_inertial_winter(
    body_mass_kg: float,
    heel_mm: np.ndarray,
    toe_mm: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """
    Foot inertial parameters using Winter anthropometric fractions.

    Reference: Winter, D.A. Biomechanics and Motor Control of Human Movement.

    Segment length L = heel (calcaneus) → toe. Math steps:

      1) Segment length (mm)
         L = ||toe - heel||

      2) Center of mass position (lab frame, mm)
         COM = heel + (COM_fraction_proximal) * (toe - heel)
             = heel + 0.44 * (toe - heel)
         So the COM lies 44% of the way from heel toward toe.

      3) Segment mass (kg)
         m_foot = (mass_fraction) * M_body = 0.0145 * M_body

      4) Radius of gyration (mm), about COM
         k = (k_fraction) * L = 0.475 * L

    Parameters
    ----------
    body_mass_kg : float
        Total body mass (kg).
    heel_mm : np.ndarray
        Heel (calcaneus) position in lab frame, shape (3,), mm.
    toe_mm : np.ndarray
        Toe position in lab frame, shape (3,), mm (Toe_Tip or mid Toe_Med/Toe_Lat).

    Returns
    -------
    dict
        length_mm, mass_kg, com_mm (3,), radius_gyration_mm.
    """
    return _winter_proximal_distal_inertial(
        heel_mm,
        toe_mm,
        WINTER_FOOT_MASS_FRACTION,
        WINTER_FOOT_COM_FRACTION_PROXIMAL,
        WINTER_FOOT_K_FRACTION,
        body_mass_kg,
        "Foot segment length (heel to toe) near zero",
    )


def _load_c3d_labels_xyz_frame(
    c3d_path: str,
    frame_idx: int,
) -> tuple[list, np.ndarray] | None:
    """POINT labels and (n_markers, 3) positions (mm) for one frame; None if unavailable."""
    if not HAVE_EZC3D or not os.path.isfile(c3d_path):
        return None
    c3d = ezc3d.c3d(c3d_path)
    labels = list(c3d["parameters"]["POINT"]["LABELS"]["value"])
    xyz = c3d["data"]["points"][:3, :, frame_idx].T
    return labels, xyz


def _heel_toe_from_c3d(
    c3d_path: str,
    frame_idx: int,
    side: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (heel_mm, toe_mm) for one foot from C3D; (None, None) if unavailable."""
    ld = _load_c3d_labels_xyz_frame(c3d_path, frame_idx)
    if ld is None:
        return None, None
    labels, xyz = ld
    return _heel_toe_from_labels_xyz(labels, xyz, side)


def _foot_markers_both_from_c3d(
    c3d_path: str,
    frame_idx: int,
) -> dict[str, tuple[np.ndarray, np.ndarray] | None]:
    """Load C3D once and return heel/toe for both feet. Keys 'L' and 'R'; value (heel, toe) or None."""
    out: dict[str, tuple[np.ndarray, np.ndarray]
              | None] = {"L": None, "R": None}
    ld = _load_c3d_labels_xyz_frame(c3d_path, frame_idx)
    if ld is None:
        return out
    labels, xyz = ld
    for side in ("L", "R"):
        ht = _heel_toe_from_labels_xyz(labels, xyz, side)
        if ht[0] is not None and ht[1] is not None:
            out[side] = ht
    return out


def _resolve_foot_markers_both(
    foot_markers_both: dict[str, tuple[np.ndarray, np.ndarray] | None] | None,
    c3d_path: str | None,
    frame_idx: int,
) -> dict[str, tuple[np.ndarray, np.ndarray] | None] | None:
    """Use provided heel/toe dict or load from ``c3d_path`` when needed."""
    if foot_markers_both is not None:
        return foot_markers_both
    if c3d_path and os.path.isfile(c3d_path):
        return _foot_markers_both_from_c3d(c3d_path, frame_idx)
    return None


def foot_anthropometry_both_winter(
    body_mass_kg: float,
    c3d_path: str | None = None,
    frame_idx: int = 0,
    foot_markers_both: dict[str, tuple[np.ndarray,
                                       np.ndarray] | None] | None = None,
) -> dict[str, dict]:
    """
    Apply Winter anthropometric fractions to both feet; returns all parameters and math summary.

    Requires heel and toe markers from C3D (Calc; Toe_Tip or Toe_Med/Toe_Lat). For each foot:
      - length_mm   = ||toe - heel||
      - com_mm      = heel + 0.44 * (toe - heel)
      - mass_kg     = 0.0145 * body_mass_kg
      - radius_gyration_mm = 0.475 * length_mm

    Parameters
    ----------
    body_mass_kg : float
        Total body mass (kg).
    c3d_path : str, optional
        Path to static trial C3D; if None and foot_markers_both is None, L_foot and R_foot will be None.
    frame_idx : int
        Frame index in C3D for foot markers (default 0).
    foot_markers_both : dict, optional
        Pre-loaded heel/toe from _foot_markers_both_from_c3d(); if provided, used instead of loading c3d_path.

    Returns
    -------
    dict
        "L_foot" and "R_foot" each with length_mm, mass_kg, com_mm, radius_gyration_mm (or None);
        "math_steps" (str) summarizing the Winter formulas used.
    """
    math_steps = (
        "Winter (Biomechanics and Motor Control of Human Movement) foot anthropometry:\n"
        "  L = ||toe - heel||  (segment length, heel to toe)\n"
        "  COM = heel + 0.44 * (toe - heel)  (44% from heel toward toe)\n"
        "  m = 0.0145 * M_body  (~1.45% of body mass)\n"
        "  k = 0.475 * L  (radius of gyration about COM, as fraction of L)"
    )
    out = {"math_steps": math_steps, "L_foot": None, "R_foot": None}
    if foot_markers_both is not None:
        markers = foot_markers_both
    elif c3d_path and os.path.isfile(c3d_path):
        markers = _foot_markers_both_from_c3d(c3d_path, frame_idx)
    else:
        return out
    for side, key in (("L", "L_foot"), ("R", "R_foot")):
        if markers.get(side) is not None:
            heel, toe = markers[side]
            try:
                out[key] = foot_inertial_winter(body_mass_kg, heel, toe)
            except ValueError:
                out[key] = None
    return out


# ---------------------------------------------------------------------------
# Inertial export per segment (mass, r_com_seg, I_com_seg) using ACS from static calibration
# ---------------------------------------------------------------------------
# Segment ID -> npz filename (without path). static_calibration saves O_A_static, R_A_static in these.
SEGMENT_ACS_TEMPLATE_MAP = {
    "L_thigh": "femur_tcs_template.npz",
    "R_thigh": "femur_right_tcs_template.npz",
    "L_shank": "tibia_tcs_template.npz",
    "R_shank": "tibia_right_tcs_template.npz",
    "L_foot": "foot_tcs_template.npz",
    "R_foot": "foot_right_tcs_template.npz",
}


def _winter_inertial_for_segment_id(
    segment_id: str,
    joint_centers: dict[str, np.ndarray],
    body_mass_kg: float,
    foot_markers_both: dict[str, tuple[np.ndarray, np.ndarray] | None] | None,
) -> dict[str, float | np.ndarray] | None:
    """
    Winter dict (length_mm, mass_kg, com_mm, radius_gyration_mm) for a supported
    ``segment_id`` in ``SEGMENT_ACS_TEMPLATE_MAP``, or None if data missing / invalid.
    """
    if segment_id not in SEGMENT_ACS_TEMPLATE_MAP:
        return None
    side = "L" if segment_id.startswith("L_") else "R"
    try:
        if segment_id.endswith("_thigh"):
            return thigh_inertial_winter(joint_centers, body_mass_kg, side=side)
        if segment_id.endswith("_shank"):
            return shank_inertial_winter(joint_centers, body_mass_kg, side=side)
        if segment_id.endswith("_foot"):
            if foot_markers_both is None or foot_markers_both.get(side) is None:
                return None
            heel, toe = foot_markers_both[side]
            return foot_inertial_winter(body_mass_kg, heel, toe)
    except (KeyError, ValueError):
        return None
    return None


def load_segment_acs_from_static(
    template_dir: str,
    base: str,
    segment_id: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load segment ACS (origin and rotation) from static calibration npz.
    Uses O_A_static (mm, lab), R_A_static (3x3, segment axes in lab as columns).
    v_lab = O_A_static + R_A_static @ v_seg  =>  v_seg = R_A_static.T @ (v_lab - O_A_static).

    Returns
    -------
    (O_A_mm, R_A) or (None, None) if file/key missing.
    """
    if segment_id not in SEGMENT_ACS_TEMPLATE_MAP:
        return None, None
    fname = SEGMENT_ACS_TEMPLATE_MAP[segment_id]
    # static_calibration saves e.g. S_Cal02_femur_tcs_template.npz in template_dir
    path = os.path.join(template_dir, f"{base}_{fname}")
    if not os.path.isfile(path):
        return None, None
    try:
        data = np.load(path, allow_pickle=True)
        if "O_A_static" not in data or "R_A_static" not in data:
            return None, None
        O_A = np.asarray(data["O_A_static"], dtype=float).reshape(3)
        R_A = np.asarray(data["R_A_static"], dtype=float).reshape(3, 3)
        return O_A, R_A
    except Exception:
        return None, None


def _inertia_tensor_diagonal_segment(
    mass_kg: float,
    length_m: float,
    radius_gyration_longitudinal_m: float,
) -> np.ndarray:
    """
    Build 3x3 inertia tensor (kg·m²) about COM in segment frame (Z = longitudinal).
    I_zz = m * k_long^2. I_xx = I_yy from thin-rod approximation (1/12)*m*L^2.
    """
    I_zz = mass_kg * (radius_gyration_longitudinal_m ** 2)
    I_xx = I_yy = (1.0 / 12.0) * mass_kg * (length_m ** 2)
    return np.diag([I_xx, I_yy, I_zz])


def build_inertial_export_segment(
    segment_id: str,
    body_mass_kg: float,
    joint_centers: dict[str, np.ndarray],
    foot_markers_both: dict[str, tuple[np.ndarray, np.ndarray] | None] | None,
    template_dir: str,
    base: str,
) -> dict | None:
    """
    Build per-segment export: mass (kg), r_com_seg (3x1 m), I_com_seg (3x3 kg·m²)
    using Winter inertial params and ACS from static calibration npz.

    Returns dict with keys mass_kg, r_com_seg (3,), I_com_seg (3,3), or None if segment/ACS unavailable.
    """
    O_A, R_A = load_segment_acs_from_static(template_dir, base, segment_id)
    if O_A is None or R_A is None:
        return None
    out_w = _winter_inertial_for_segment_id(
        segment_id, joint_centers, body_mass_kg, foot_markers_both
    )
    if out_w is None:
        return None
    com_mm = out_w["com_mm"]
    mass_kg = out_w["mass_kg"]
    length_mm = out_w["length_mm"]
    k_mm = out_w["radius_gyration_mm"]
    # r_com_seg: from segment origin to COM, in segment frame (m)
    r_com_lab_mm = np.asarray(com_mm, dtype=float).reshape(3) - O_A
    r_com_seg_m = (R_A.T @ r_com_lab_mm) / 1000.0
    # I_com_seg (3x3, kg·m²) in segment frame (Z longitudinal)
    length_m = length_mm / 1000.0
    k_m = k_mm / 1000.0
    I_com_seg = _inertia_tensor_diagonal_segment(mass_kg, length_m, k_m)
    return {
        "mass_kg": mass_kg,
        "r_com_seg": np.asarray(r_com_seg_m, dtype=float).reshape(3),
        "I_com_seg": I_com_seg,
    }


def export_inertial_segments(
    body_mass_kg: float,
    joint_centers: dict[str, np.ndarray],
    foot_markers_both: dict[str, tuple[np.ndarray, np.ndarray] | None] | None,
    template_dir: str,
    base: str,
    out_path: str | None = None,
) -> dict[str, dict]:
    """
    Build and optionally save per-segment inertial export (mass, r_com_seg, I_com_seg)
    using ACS from static calibration. Each segment: mass_kg, r_com_seg (3, m), I_com_seg (3x3, kg·m²).

    If out_path is provided, saves to npz (arrays) and to a sibling .json for readability (rounded).
    """
    results = {}
    for seg_id in SEGMENT_ACS_TEMPLATE_MAP:
        data = build_inertial_export_segment(
            seg_id, body_mass_kg, joint_centers, foot_markers_both, template_dir, base
        )
        if data is not None:
            results[seg_id] = data
    if out_path:
        npz_path = out_path if out_path.endswith(
            ".npz") else (out_path + ".npz")
        to_save = {}
        for seg_id, d in results.items():
            to_save[f"{seg_id}_mass_kg"] = np.array(d["mass_kg"], dtype=float)
            to_save[f"{seg_id}_r_com_seg"] = d["r_com_seg"]
            to_save[f"{seg_id}_I_com_seg"] = d["I_com_seg"]
        np.savez(npz_path, **to_save)
        print(f"Saved inertial export: {os.path.abspath(npz_path)}")
        try:
            import json
            json_path = npz_path.replace(".npz", "_inertial_export.json")
            json_obj = {}
            for seg_id, d in results.items():
                json_obj[seg_id] = {
                    "mass_kg": round(float(d["mass_kg"]), 6),
                    "r_com_seg_m": d["r_com_seg"].tolist(),
                    "I_com_seg_kg_m2": [row.tolist() for row in d["I_com_seg"]],
                }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_obj, f, indent=2)
            print(f"Saved inertial export (readable): {
                  os.path.abspath(json_path)}")
        except Exception as e:
            print(f"Could not write JSON: {e}")
    return results


# ACS check tolerances
ACS_R_ORTHONORMAL_TOL = 1e-4   # max ||R.T@R - I||_F and |det(R)-1|
ACS_COM_CONSISTENCY_MM = 5.0   # max ||reconstructed COM - Winter COM|| in mm


def check_acs_loaded_and_used(
    template_dir: str,
    base: str,
    body_mass_kg: float,
    joint_centers: dict[str, np.ndarray],
    foot_markers_both: dict[str, tuple[np.ndarray, np.ndarray] | None] | None,
    inertial_export: dict[str, dict],
    verbose: bool = True,
) -> dict:
    """
    Ensure ACS from static calibration were loaded and used properly:
    1) Each exported segment has valid ACS (R orthonormal, det=1).
    2) Reconstructed COM in lab (O_A + R_A @ r_com_seg) matches Winter COM within tolerance.

    Returns
    -------
    dict with all_ok (bool), results (list of per-segment checks), issues (list of str).
    """
    issues = []
    results = []
    for seg_id in list(inertial_export.keys()):
        O_A, R_A = load_segment_acs_from_static(template_dir, base, seg_id)
        d = inertial_export[seg_id]
        r_com_seg = d["r_com_seg"]  # m
        # 1) ACS loaded
        if O_A is None or R_A is None:
            issues.append(
                "{}: ACS not loaded (missing npz or O_A_static/R_A_static)".format(seg_id))
            results.append({"segment": seg_id, "acs_loaded": False,
                           "R_orthonormal": False, "com_consistent": False})
            continue
        # 2) R is valid rotation (orthonormal, right-handed)
        R_A = np.asarray(R_A, dtype=float).reshape(3, 3)
        I_err = np.linalg.norm(R_A.T @ R_A - np.eye(3), ord="fro")
        det_err = abs(np.linalg.det(R_A) - 1.0)
        R_ok = I_err <= ACS_R_ORTHONORMAL_TOL and det_err <= ACS_R_ORTHONORMAL_TOL
        if not R_ok:
            issues.append(
                "{}: R_A_static not orthonormal (||R'R-I||={:.2e}, |det-1|={:.2e})".format(seg_id, I_err, det_err))
        # 3) Reconstructed COM = O_A + R_A @ r_com_seg (r in m -> mm) should match Winter COM
        r_com_mm = r_com_seg * 1000.0
        com_reconstructed = O_A + R_A @ r_com_mm
        win = _winter_inertial_for_segment_id(
            seg_id, joint_centers, body_mass_kg, foot_markers_both)
        com_winter = win["com_mm"] if win is not None else None
        com_ok = True
        if com_winter is not None:
            com_winter = np.asarray(com_winter, dtype=float).reshape(3)
            diff_mm = np.linalg.norm(com_reconstructed - com_winter)
            com_ok = diff_mm <= ACS_COM_CONSISTENCY_MM
            if not com_ok:
                issues.append("{}: COM mismatch (reconstructed vs Winter) = {:.2f} mm".format(
                    seg_id, diff_mm))
        else:
            com_ok = True  # skip if Winter COM not available
        results.append({
            "segment": seg_id,
            "acs_loaded": True,
            "R_orthonormal": R_ok,
            "com_consistent": com_ok,
            "com_diff_mm": float(np.linalg.norm(com_reconstructed - com_winter)) if com_winter is not None else None,
        })
    all_ok = len(issues) == 0
    out = {"all_ok": all_ok, "results": results, "issues": issues}
    if verbose:
        print("\nCheck: ACS loaded and used correctly")
        print("  (R orthonormal; reconstructed COM = O_A + R_A @ r_com_seg matches Winter COM.)")
        for r in results:
            status = "PASS" if (
                r["acs_loaded"] and r["R_orthonormal"] and r["com_consistent"]) else "FAIL"
            msg = "{}: loaded={}, R_ok={}, COM_ok={}".format(
                r["segment"], r["acs_loaded"], r["R_orthonormal"], r["com_consistent"])
            if r.get("com_diff_mm") is not None:
                msg += " (diff={:.2f} mm)".format(r["com_diff_mm"])
            print("  {} {}".format(status, msg))
        if all_ok:
            print("  Overall: PASS")
        else:
            print("  Overall: FAIL")
            for s in issues:
                print("    - {}".format(s))
    return out


def _pick_marker(labels: list, xyz_frame: np.ndarray, name: str) -> np.ndarray | None:
    """Return marker position (3,) for label matching name (exact or suffix)."""
    for i, lab in enumerate(labels):
        if lab is None:
            continue
        s = str(lab).strip()
        if s == name or s.endswith(":" + name) or s.split(":")[-1].strip() == name:
            return xyz_frame[i, :].astype(float)
    lower = name.lower()
    for i, lab in enumerate(labels):
        if lab is None:
            continue
        s = str(lab).strip().lower()
        if s == lower or s.endswith(":" + lower):
            return xyz_frame[i, :].astype(float)
    return None


def _heel_toe_from_labels_xyz(
    labels: list,
    xyz_frame: np.ndarray,
    side: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (heel_mm, toe_mm) for one foot from labels and one frame of xyz; (None, None) if unavailable."""
    pref = "L_" if side.upper() == "L" else "R_"
    heel = _pick_marker(labels, xyz_frame, pref + "Calc")
    toe_tip = _pick_marker(labels, xyz_frame, pref + "Toe_Tip")
    toe_med = _pick_marker(labels, xyz_frame, pref + "Toe_Med")
    toe_lat = _pick_marker(labels, xyz_frame, pref + "Toe_Lat")
    if heel is None:
        return None, None
    if toe_tip is not None:
        toe = toe_tip
    elif toe_med is not None and toe_lat is not None:
        toe = 0.5 * (toe_med + toe_lat)
    else:
        return None, None
    return heel, toe


def foot_length_from_markers(
    heel_mm: np.ndarray,
    toe_mm: np.ndarray,
) -> float:
    """
    Foot length = heel (calcaneus) → toe marker, in mm.
    """
    return float(np.linalg.norm(np.asarray(toe_mm, dtype=float) - np.asarray(heel_mm, dtype=float)))


def foot_length_from_c3d(
    c3d_path: str,
    frame_idx: int,
    side: str = "L",
) -> float | None:
    """
    Load one frame from C3D and compute foot length (heel → toe) in mm.

    Heel = Calc; toe = Toe_Tip if available, else midpoint of Toe_Med and Toe_Lat.

    Parameters
    ----------
    c3d_path : str
        Path to static trial C3D.
    frame_idx : int
        Frame index (e.g. 0).
    side : str
        "L" or "R" for left/right foot.

    Returns
    -------
    float or None
        Foot length in mm, or None if markers missing.
    """
    heel, toe = _heel_toe_from_c3d(c3d_path, frame_idx, side)
    if heel is None or toe is None:
        return None
    return foot_length_from_markers(heel, toe)


# Mass total check: sum of segment masses as fraction of body mass
# One leg (thigh + shank + foot): ~15–20%; both legs: ~30–40%. ~70% suggests an error (e.g. double-counting).
MASS_FRACTION_ONE_LEG_MIN = 0.15
MASS_FRACTION_ONE_LEG_MAX = 0.20
MASS_FRACTION_BOTH_LEGS_MIN = 0.30
MASS_FRACTION_BOTH_LEGS_MAX = 0.40
MASS_FRACTION_SUSPICIOUS_HIGH = 0.50  # above this (e.g. 70%) → something wrong


def check_mass_totals(
    body_mass_kg: float,
    joint_centers: dict[str, np.ndarray],
    foot_markers_both: dict[str, tuple[np.ndarray,
                                       np.ndarray] | None] | None = None,
    c3d_path: str | None = None,
    frame_idx: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Check 1: Mass totals. Sum of modeled segment masses should be
    ~15–20% of body mass (one leg) or ~30–40% (both legs). ~70% → something wrong.

    Parameters
    ----------
    body_mass_kg : float
        Total body mass (kg).
    joint_centers : dict
        From load_joint_centers().
    foot_markers_both : dict, optional
        Pre-loaded heel/toe from _foot_markers_both_from_c3d(); used for foot mass.
    c3d_path : str, optional
        If foot_markers_both is None, used to load foot markers.
    frame_idx : int
        Frame index for C3D.
    verbose : bool
        If True, print summary and warn if out of range.

    Returns
    -------
    dict
        total_kg, total_fraction, n_legs (1 or 2), in_range (bool), message (str).
    """
    total_kg = 0.0
    n_legs = 0
    for side in ("L", "R"):
        try:
            t = thigh_inertial_winter(joint_centers, body_mass_kg, side=side)
            total_kg += t["mass_kg"]
            n_legs += 1
        except (KeyError, ValueError):
            pass
        try:
            s = shank_inertial_winter(joint_centers, body_mass_kg, side=side)
            total_kg += s["mass_kg"]
        except (KeyError, ValueError):
            pass
    foot_markers = _resolve_foot_markers_both(
        foot_markers_both, c3d_path, frame_idx)
    for side in ("L", "R"):
        if foot_markers and foot_markers.get(side) is not None:
            try:
                f = foot_inertial_winter(
                    body_mass_kg, foot_markers[side][0], foot_markers[side][1])
                total_kg += f["mass_kg"]
            except ValueError:
                pass

    if body_mass_kg <= 0:
        total_fraction = 0.0
        in_range = False
        message = "body_mass_kg must be > 0"
    else:
        total_fraction = total_kg / body_mass_kg
        if n_legs <= 1:
            lo, hi = MASS_FRACTION_ONE_LEG_MIN, MASS_FRACTION_ONE_LEG_MAX
            in_range = lo <= total_fraction <= hi
            message = "one leg: expected {:.0%}–{:.0%}".format(lo, hi)
        else:
            lo, hi = MASS_FRACTION_BOTH_LEGS_MIN, MASS_FRACTION_BOTH_LEGS_MAX
            in_range = lo <= total_fraction <= hi
            message = "both legs: expected {:.0%}–{:.0%}".format(lo, hi)
        if total_fraction >= MASS_FRACTION_SUSPICIOUS_HIGH:
            in_range = False
            message += "; {:.1%} is suspiciously high (e.g. double-counting?)".format(
                total_fraction)

    out = {
        "total_kg": total_kg,
        "total_fraction": total_fraction,
        "n_legs": n_legs,
        "in_range": in_range,
        "message": message,
    }
    if verbose:
        print("\nCheck 1: Mass totals")
        print("  Sum of segment masses: {:.3f} kg ({:.1%} of body mass)".format(
            total_kg, total_fraction))
        print("  {}".format(message))
        if in_range:
            print("  PASS")
        else:
            print("  FAIL or SUSPICIOUS")
        if total_fraction >= MASS_FRACTION_SUSPICIOUS_HIGH:
            print("  If ~70%: check for double-counting or wrong body mass.")
    return out


# Check 2: COM location. COM must lie within segment bounds and (for thigh/shank) be closer to proximal end.
# COM should lie on segment line at expected Winter fraction (thigh/shank 0.433, foot 0.44).
COM_BOUNDS_TOLERANCE = 0.02   # allow t in [-0.02, 1.02] as "within bounds"
# max distance (mm) of COM from segment line (numerical tolerance)
COM_MAX_OFF_LINE_MM = 2.0
# |t - expected| <= this to confirm expected fraction
COM_EXPECTED_FRACTION_TOLERANCE = 0.02


def _segment_com_bounds_check(
    proximal_mm: np.ndarray,
    distal_mm: np.ndarray,
    com_mm: np.ndarray,
    segment_name: str,
    require_closer_to_proximal: bool = False,
    expected_fraction: float | None = None,
) -> dict:
    """
    Check that COM lies within segment bounds (between proximal and distal),
    optionally closer to proximal (for thigh/shank), and on the segment line at expected fraction.

    t = dot(COM - proximal, distal - proximal) / L^2  →  fraction from proximal (0.433 shank, 0.44 foot).
    """
    p = np.asarray(proximal_mm, dtype=float).reshape(3)
    d = np.asarray(distal_mm, dtype=float).reshape(3)
    c = np.asarray(com_mm, dtype=float).reshape(3)
    vec = d - p
    L_sq = float(np.dot(vec, vec))
    if L_sq < 1e-12:
        return {
            "segment": segment_name,
            "in_bounds": False,
            "closer_to_proximal_ok": False,
            "t_fraction": float("nan"),
            "dist_to_line_mm": float("nan"),
            "expected_fraction": expected_fraction,
            "fraction_ok": False,
            "message": "segment length near zero",
        }
    t = float(np.dot(c - p, vec) / L_sq)
    closest = p + t * vec
    dist_to_line_mm = float(np.linalg.norm(c - closest))
    in_bounds = (COM_BOUNDS_TOLERANCE * -1 <= t <= 1.0 + COM_BOUNDS_TOLERANCE and
                 dist_to_line_mm <= COM_MAX_OFF_LINE_MM)
    closer_ok = (not require_closer_to_proximal) or (t <= 0.5)
    if expected_fraction is not None:
        fraction_ok = abs(
            t - expected_fraction) <= COM_EXPECTED_FRACTION_TOLERANCE
    else:
        fraction_ok = True
    if not in_bounds:
        msg = "COM outside segment (t={:.3f}, dist_to_line={:.2f} mm)".format(
            t, dist_to_line_mm)
    elif not closer_ok:
        msg = "COM not closer to proximal end (t={:.3f} > 0.5)".format(t)
    elif expected_fraction is not None and not fraction_ok:
        msg = "COM t={:.3f} (expected {:.2f}), {:.2f} mm from line".format(
            t, expected_fraction, dist_to_line_mm)
    else:
        exp_str = " (expected {:.2f})".format(
            expected_fraction) if expected_fraction is not None else ""
        msg = "OK: on segment line at t={:.3f}{}, {:.2f} mm from line".format(
            t, exp_str, dist_to_line_mm)
    return {
        "segment": segment_name,
        "in_bounds": in_bounds,
        "closer_to_proximal_ok": closer_ok,
        "t_fraction": t,
        "dist_to_line_mm": dist_to_line_mm,
        "expected_fraction": expected_fraction,
        "fraction_ok": fraction_ok,
        "message": msg,
    }


def check_com_locations(
    body_mass_kg: float,
    joint_centers: dict[str, np.ndarray],
    foot_markers_both: dict[str, tuple[np.ndarray,
                                       np.ndarray] | None] | None = None,
    c3d_path: str | None = None,
    frame_idx: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Check 2: COM location. For each segment, COM must lie within segment bounds
    and (for thigh/shank) be closer to the proximal end. If COM is outside the segment → error.

    Parameters
    ----------
    body_mass_kg : float
        Total body mass (kg).
    joint_centers : dict
        From load_joint_centers().
    foot_markers_both : dict, optional
        Pre-loaded heel/toe for foot segments.
    c3d_path : str, optional
        Used to load foot markers if foot_markers_both is None.
    frame_idx : int
        Frame index for C3D.
    verbose : bool
        If True, print per-segment results and any errors.

    Returns
    -------
    dict
        all_ok (bool), results (list of per-segment dicts from _segment_com_bounds_check).
    """
    results = []
    for side in ("L", "R"):
        # Thigh: proximal = HJC, distal = KJC
        try:
            t = thigh_inertial_winter(joint_centers, body_mass_kg, side=side)
            hjc = joint_centers["LHJC"] if side == "L" else joint_centers["RHJC"]
            kjc = joint_centers["L_KJC"] if side == "L" else joint_centers["R_KJC"]
            r = _segment_com_bounds_check(
                hjc, kjc, t["com_mm"], "{}_thigh".format(side),
                require_closer_to_proximal=True,
                expected_fraction=WINTER_THIGH_COM_FRACTION_PROXIMAL,
            )
            results.append(r)
        except (KeyError, ValueError):
            pass
        # Shank: proximal = KJC, distal = AJC (expected 0.433)
        try:
            s = shank_inertial_winter(joint_centers, body_mass_kg, side=side)
            kjc = joint_centers["L_KJC"] if side == "L" else joint_centers["R_KJC"]
            ajc = joint_centers["L_AJC"] if side == "L" else joint_centers["R_AJC"]
            r = _segment_com_bounds_check(
                kjc, ajc, s["com_mm"], "{}_shank".format(side),
                require_closer_to_proximal=True,
                expected_fraction=WINTER_SHANK_COM_FRACTION_PROXIMAL,
            )
            results.append(r)
        except (KeyError, ValueError):
            pass
    foot_markers = _resolve_foot_markers_both(
        foot_markers_both, c3d_path, frame_idx)
    for side in ("L", "R"):
        if foot_markers and foot_markers.get(side) is not None:
            try:
                f = foot_inertial_winter(
                    body_mass_kg, foot_markers[side][0], foot_markers[side][1])
                heel, toe = foot_markers[side][0], foot_markers[side][1]
                r = _segment_com_bounds_check(
                    heel, toe, f["com_mm"], "{}_foot".format(side),
                    require_closer_to_proximal=False,
                    expected_fraction=WINTER_FOOT_COM_FRACTION_PROXIMAL,
                )
                results.append(r)
            except ValueError:
                pass
    all_ok = all(
        r["in_bounds"] and r["closer_to_proximal_ok"] and r.get(
            "fraction_ok", True)
        for r in results
    )
    out = {"all_ok": all_ok, "results": results}
    if verbose:
        print("\nCheck 2: COM location")
        print("  COM must lie on segment line within bounds; thigh/shank 0.433, foot 0.44 from proximal.")
        for r in results:
            ok = r["in_bounds"] and r["closer_to_proximal_ok"] and r.get(
                "fraction_ok", True)
            status = "PASS" if ok else "FAIL"
            print("  {} {}: {}".format(r["segment"], status, r["message"]))
        if all_ok:
            print("  Overall: PASS (COM on correct segment line at expected fraction)")
        else:
            print(
                "  Overall: FAIL (COM outside segment, wrong fraction, or not closer to proximal)")
    return out


# Check 3: Inertia magnitude. I should scale with size; longer segments → ~quadratic in L. Smaller subject with larger I → check units.
# I = m * k² (about COM). I/L² = m * (k/L)² should be in a plausible range (kg). Left vs right: similar segment type should have I_L/I_R ≈ (L_L/L_R)².
INERTIA_IL2_THIGH_KG_MIN = 0.05   # I/L² (kg) plausible min for thigh
INERTIA_IL2_THIGH_KG_MAX = 2.0
INERTIA_IL2_SHANK_KG_MIN = 0.02
INERTIA_IL2_SHANK_KG_MAX = 1.5
INERTIA_IL2_FOOT_KG_MIN = 0.01
INERTIA_IL2_FOOT_KG_MAX = 1.0
# I_left/I_right or I_right/I_left should be >= this (avoid 0)
INERTIA_LR_RATIO_MIN = 0.3
# if ratio outside [0.3, 3.5] with similar L → suspect units
INERTIA_LR_RATIO_MAX = 3.5
# if shorter segment has I > longer_segment_I * this → flag (units?)
INERTIA_SMALLER_SEGMENT_MAX_RATIO = 2.0


def _inertia_magnitude_row(seg_type: str, side: str, w: dict) -> dict:
    """One row for ``check_inertia_magnitude``: I = m*k² in kg·mm², I/L² in kg, with plausibility bounds."""
    L = w["length_mm"]
    I_kg_mm2 = w["mass_kg"] * (w["radius_gyration_mm"] ** 2)
    I_over_L2 = I_kg_mm2 / (L * L) if L > 0 else float("nan")
    il2_min, il2_max = {
        "thigh": (INERTIA_IL2_THIGH_KG_MIN, INERTIA_IL2_THIGH_KG_MAX),
        "shank": (INERTIA_IL2_SHANK_KG_MIN, INERTIA_IL2_SHANK_KG_MAX),
        "foot": (INERTIA_IL2_FOOT_KG_MIN, INERTIA_IL2_FOOT_KG_MAX),
    }[seg_type]
    return {
        "segment": seg_type,
        "side": side,
        "mass_kg": w["mass_kg"],
        "length_mm": L,
        "k_mm": w["radius_gyration_mm"],
        "I_kg_mm2": I_kg_mm2,
        "I_over_L2_kg": I_over_L2,
        "il2_min": il2_min,
        "il2_max": il2_max,
    }


def check_inertia_magnitude(
    body_mass_kg: float,
    joint_centers: dict[str, np.ndarray],
    foot_markers_both: dict[str, tuple[np.ndarray,
                                       np.ndarray] | None] | None = None,
    c3d_path: str | None = None,
    frame_idx: int = 0,
    verbose: bool = True,
) -> dict:
    """
    Check 3: Inertia magnitude. Moment of inertia should scale roughly with:
    larger subjects → larger inertia; longer segments → quadratic increase.
    If a smaller subject (or shorter segment) has larger inertia → check units.

    Uses I = m * k² (about COM, longitudinal). I/L² = m*(k/L)² in kg; left vs right compared.
    """
    segments: list[dict] = []
    for side in ("L", "R"):
        try:
            segments.append(_inertia_magnitude_row(
                "thigh", side, thigh_inertial_winter(joint_centers, body_mass_kg, side=side)))
        except (KeyError, ValueError):
            pass
        try:
            segments.append(_inertia_magnitude_row(
                "shank", side, shank_inertial_winter(joint_centers, body_mass_kg, side=side)))
        except (KeyError, ValueError):
            pass
    foot_markers = _resolve_foot_markers_both(
        foot_markers_both, c3d_path, frame_idx)
    for side in ("L", "R"):
        if foot_markers and foot_markers.get(side) is not None:
            try:
                f = foot_inertial_winter(
                    body_mass_kg, foot_markers[side][0], foot_markers[side][1])
                segments.append(_inertia_magnitude_row("foot", side, f))
            except ValueError:
                pass

    issues = []
    for seg in segments:
        il2 = seg["I_over_L2_kg"]
        if not (seg["il2_min"] <= il2 <= seg["il2_max"]):
            issues.append("{} {} I/L² = {:.4f} kg outside [{}, {}]".format(
                seg["segment"], seg["side"], il2, seg["il2_min"], seg["il2_max"]))
    # Left vs right: shorter segment should not have much larger I (suggests units error)
    for seg_type in ("thigh", "shank", "foot"):
        pair = [s for s in segments if s["segment"] == seg_type]
        if len(pair) != 2:
            continue
        # Ensure order L then R
        pair = sorted(pair, key=lambda s: (0 if s["side"] == "L" else 1))
        L_L, I_L = pair[0]["length_mm"], pair[0]["I_kg_mm2"]
        L_R, I_R = pair[1]["length_mm"], pair[1]["I_kg_mm2"]
        if L_L <= 0 or L_R <= 0:
            continue
        # If shorter has larger I than longer * factor → flag (check units)
        if L_L < L_R and I_L > I_R * INERTIA_SMALLER_SEGMENT_MAX_RATIO:
            issues.append("{}: shorter segment (L, L={:.0f} mm) has I > R (L={:.0f} mm) by factor >{} → check units".format(
                seg_type, L_L, L_R, INERTIA_SMALLER_SEGMENT_MAX_RATIO))
        elif L_R < L_L and I_R > I_L * INERTIA_SMALLER_SEGMENT_MAX_RATIO:
            issues.append("{}: shorter segment (R, L={:.0f} mm) has I > L (L={:.0f} mm) by factor >{} → check units".format(
                seg_type, L_R, L_L, INERTIA_SMALLER_SEGMENT_MAX_RATIO))
        # L-R ratio sanity: I_L/I_R should be in [0.3, 3.5] (similar segments)
        ratio = I_L / I_R if I_R > 0 else float("nan")
        if not (INERTIA_LR_RATIO_MIN <= ratio <= INERTIA_LR_RATIO_MAX):
            issues.append("{}: I_L/I_R = {:.2f} outside [{}, {}]".format(
                seg_type, ratio, INERTIA_LR_RATIO_MIN, INERTIA_LR_RATIO_MAX))

    all_ok = len(issues) == 0
    out = {"all_ok": all_ok, "segments": segments, "issues": issues}
    if verbose:
        print("\nCheck 3: Inertia magnitude")
        print("  I = m*k² (kg·mm²); I/L² (kg) should be in plausible range. Longer segment → larger I.")
        for seg in segments:
            name = "{}_{}".format(seg["side"], seg["segment"])
            il2_ok = seg["il2_min"] <= seg["I_over_L2_kg"] <= seg["il2_max"]
            print("  {}: I = {:.1e} kg·mm², I/L² = {:.4f} kg {} (L = {:.0f} mm)".format(
                name, seg["I_kg_mm2"], seg["I_over_L2_kg"], "PASS" if il2_ok else "OUT OF RANGE", seg["length_mm"]))
        if issues:
            print("  Issues:")
            for msg in issues:
                print("    - {}".format(msg))
            print("  Overall: FAIL (check units, e.g. kg·mm² vs kg·m²)")
        else:
            print("  Overall: PASS")
    return out


def segment_lengths_from_static(
    joint_centers_csv_path: str,
    c3d_path: str | None = None,
    frame_idx: int = 0,
    foot_markers_both: dict[str, tuple[np.ndarray,
                                       np.ndarray] | None] | None = None,
) -> dict[str, float]:
    """
    Compute all segment lengths from static trial: joint centers CSV + optional C3D for foot.

    Parameters
    ----------
    joint_centers_csv_path : str
        Path to *_joint_centers.csv from static_calibration.py.
    c3d_path : str, optional
        Path to same static trial C3D; if provided (and foot_markers_both is None), foot lengths are computed from markers.
    frame_idx : int
        Frame index in C3D for foot markers (default 0).
    foot_markers_both : dict, optional
        Pre-loaded heel/toe from _foot_markers_both_from_c3d(); if provided, used instead of loading c3d_path.

    Returns
    -------
    dict[str, float]
        L_thigh_mm, R_thigh_mm, L_shank_mm, R_shank_mm, pelvis_width_mm,
        and if c3d_path given: L_foot_mm, R_foot_mm.
    """
    jc = load_joint_centers(joint_centers_csv_path)
    out = segment_lengths(jc)
    foot_markers = _resolve_foot_markers_both(
        foot_markers_both, c3d_path, frame_idx)
    if foot_markers:
        for side, key in (("L", "L_foot_mm"), ("R", "R_foot_mm")):
            if foot_markers.get(side) is not None:
                heel, toe = foot_markers[side]
                out[key] = foot_length_from_markers(heel, toe)
    return out


def main() -> None:
    """Example: load joint centers and print segment lengths from static trial."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    subject, base = "subject 02", "S_Cal02"
    folder = f"{subject} - {base}"
    csv_path = os.path.join(script_dir, folder, f"{base}_joint_centers.csv")
    c3d_path = os.path.join(script_dir, "..", "..",
                            "c3d", subject, f"{base}.c3d")
    c3d_path = os.path.abspath(c3d_path)

    if not os.path.isfile(csv_path):
        print(f"Example CSV not found: {csv_path}")
        print("Run static_calibration.py first to generate the joint centers CSV.")
        return

    # Segment lengths (foot length from one C3D load, single source: foot_length_from_markers)
    foot_markers_both = _foot_markers_both_from_c3d(
        c3d_path, 0) if os.path.isfile(c3d_path) else None
    lengths = segment_lengths_from_static(
        csv_path, c3d_path=None, frame_idx=0, foot_markers_both=foot_markers_both
    )
    print("Segment lengths (mm) from static trial:")
    print("  Thigh  (hip JC → knee JC):  L = {:.2f}  R = {:.2f}".format(
        lengths.get("L_thigh_mm", float("nan")), lengths.get("R_thigh_mm", float("nan"))))
    print("  Shank  (knee JC → ankle JC): L = {:.2f}  R = {:.2f}".format(
        lengths.get("L_shank_mm", float("nan")), lengths.get("R_shank_mm", float("nan"))))
    print("  Pelvis width (LHJC–RHJC):   {:.2f}".format(
        lengths.get("pelvis_width_mm", float("nan"))))
    if "L_foot_mm" in lengths or "R_foot_mm" in lengths:
        print("  Foot   (heel → toe):         L = {:.2f}  R = {:.2f}".format(
            lengths.get("L_foot_mm", float("nan")), lengths.get("R_foot_mm", float("nan"))))
    else:
        print("  Foot   (heel → toe):         pass c3d_path to segment_lengths_from_static() to include.")

    # Winter anthropometry for both thighs and both shanks
    jc = load_joint_centers(csv_path)
    body_mass_kg = 102.1  # example; replace with subject mass
    thigh_both = thigh_anthropometry_both_winter(jc, body_mass_kg)
    print("\n" + thigh_both["math_steps"])
    print("\nThigh inertial parameters (Winter):")
    for key in ("L_thigh", "R_thigh"):
        if thigh_both[key] is not None:
            t = thigh_both[key]
            print("  {}: length = {:.2f} mm, mass = {:.3f} kg, k = {:.2f} mm, COM = {}".format(
                key, t["length_mm"], t["mass_kg"], t["radius_gyration_mm"], t["com_mm"]))
        else:
            print("  {}: (missing joint centers)".format(key))
    shank_both = shank_anthropometry_both_winter(jc, body_mass_kg)
    print("\n" + shank_both["math_steps"])
    print("\nShank inertial parameters (Winter):")
    for key in ("L_shank", "R_shank"):
        if shank_both[key] is not None:
            s = shank_both[key]
            print("  {}: length = {:.2f} mm, mass = {:.3f} kg, k = {:.2f} mm, COM = {}".format(
                key, s["length_mm"], s["mass_kg"], s["radius_gyration_mm"], s["com_mm"]))
        else:
            print("  {}: (missing joint centers)".format(key))
    foot_both = foot_anthropometry_both_winter(
        body_mass_kg, c3d_path=None, frame_idx=0, foot_markers_both=foot_markers_both
    )
    print("\n" + foot_both["math_steps"])
    print("\nFoot inertial parameters (Winter):")
    for key in ("L_foot", "R_foot"):
        if foot_both[key] is not None:
            f = foot_both[key]
            print("  {}: length = {:.2f} mm, mass = {:.3f} kg, k = {:.2f} mm, COM = {}".format(
                key, f["length_mm"], f["mass_kg"], f["radius_gyration_mm"], f["com_mm"]))
        else:
            print("  {}: (pass c3d_path for heel/toe markers)".format(key))

    # Export per-segment inertial data (mass, r_com_seg, I_com_seg) using ACS from static calibration
    template_dir = os.path.join(script_dir, folder)
    inertial_export = export_inertial_segments(
        body_mass_kg, jc, foot_markers_both,
        template_dir=template_dir,
        base=base,
        out_path=os.path.join(template_dir, f"{base}_inertial_segments.npz"),
    )
    if inertial_export:
        print("\nInertial export (per segment, ACS from static calibration):")
        for seg_id, d in inertial_export.items():
            print("  {}: m = {:.4f} kg, r_com_seg = {} m, I_com_seg (diag) = {} kg·m²".format(
                seg_id, d["mass_kg"], d["r_com_seg"].tolist(),
                np.diag(d["I_com_seg"]).tolist()))

    # -------------------------------------------------------------------------
    # Sanity checks (mass totals, COM location, inertia magnitude)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)

    # Check 1: mass totals (~15–20% one leg, ~30–40% both legs)
    check_mass_totals(
        body_mass_kg, jc,
        foot_markers_both=foot_markers_both,
        c3d_path=None, frame_idx=0,
        verbose=True,
    )
    # Check 2: COM within segment bounds, closer to proximal for thigh/shank
    check_com_locations(
        body_mass_kg, jc,
        foot_markers_both=foot_markers_both,
        c3d_path=None, frame_idx=0,
        verbose=True,
    )
    # Check 3: Inertia magnitude (I ∝ L²; smaller segment should not have larger I → check units)
    check_inertia_magnitude(
        body_mass_kg, jc,
        foot_markers_both=foot_markers_both,
        c3d_path=None, frame_idx=0,
        verbose=True,
    )
    # Check 4: ACS loaded and used correctly (R orthonormal; reconstructed COM matches Winter)
    if inertial_export:
        check_acs_loaded_and_used(
            template_dir, base, body_mass_kg, jc, foot_markers_both,
            inertial_export, verbose=True,
        )


if __name__ == "__main__":
    main()
