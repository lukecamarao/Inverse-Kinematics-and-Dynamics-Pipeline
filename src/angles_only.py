# -*- coding: utf-8 -*-
"""
Angles-only utility: load results NPZ and regenerate angles/plots
without redoing any fits. Supports left-only and bilateral results.

Static offsets (same convention as ``static_calibration.py``): if
``*_hip_offsets.npz`` and/or ``*_knee_gs_offsets.npz`` exist next to the results
(or its parent folder), per-channel means from the static trial are **subtracted**
from hip and knee time series. Expected arrays: ``left`` shape (3,) for
[FE, AbAd, IE] and [knee FE, Var/Val, IE]; optional ``right`` (3,) for bilateral
when present in the file.

Usage:
    python angles_only.py C:/path/to/<walk_basename>_left_chain_results.npz
    or bilateral:
    python angles_only.py C:/path/to/<walk_basename>_bilateral_chain_results.npz

    Optional walk C3D (for Plotly raw markers + ``R_Knee_Lat``): pass as 2nd or 3rd arg
    (``.c3d`` path). If omitted, the script searches next to the NPZ, parent folders, and
    common ``c3d`` directories for ``<walk_basename>.c3d``.
"""
# run in static calib directory
import sys
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any
try:
    import ezc3d
    HAVE_C3D = True
except Exception:
    HAVE_C3D = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio
    HAVE_PLOTLY = True
except Exception:
    go = None  # type: ignore[assignment]
    make_subplots = None  # type: ignore[assignment]
    pio = None  # type: ignore[assignment]
    HAVE_PLOTLY = False


# ---- Plotly viewer styling (used only when HAVE_PLOTLY) ----
PLOTLY_VIEWER_PALETTE: Dict[str, str] = {
    "pelvis": "#7f7f7f",
    "hip": "#1f77b4",
    "knee": "#2ca02c",
    "ankle": "#ff7f0e",
    "foot": "#9467bd",
}
PLOTLY_VIEWER_ANGLE_COLORS: Dict[str, str] = {
    "ankle_pfdf": "#ff7f0e",
    "knee_fe": "#1f77b4",
    "knee_varval": "#d62728",
    "hip_fe": "#9467bd",
}


# ---- LOCAL FALLBACK ANGLE HELPERS ----
def _euler_xyz_from_R(R: np.ndarray):
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


def _enforce_right_handed(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    Rn = U @ Vt
    if np.linalg.det(Rn) < 0:
        U[:, -1] *= -1.0
        Rn = U @ Vt
    return Rn


def hip_angles_isb(R_pelvis: np.ndarray, R_femur: np.ndarray) -> tuple[float, float, float]:
    """
    ISB hip Joint Coordinate System:
      - e1: pelvis X (proximal; flexion/extension axis)
      - e3: femur Z (distal; axial rotation axis)
      - e2: floating axis = e3 × e1
    Angles returned (deg): (FE, AbAd, IE)
    """
    if R_pelvis is None or R_femur is None or (np.isnan(R_pelvis).any() or np.isnan(R_femur).any()):
        return np.nan, np.nan, np.nan
    Rp = _enforce_right_handed(R_pelvis.astype(float))
    Rf = _enforce_right_handed(R_femur.astype(float))
    pX, pY, pZ = Rp[:, 0], Rp[:, 1], Rp[:, 2]
    fX, fY, fZ = Rf[:, 0], Rf[:, 1], Rf[:, 2]
    # Align femur long-axis direction with pelvis Z (proximal/up) by paired X/Z flip if needed
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
    # FE about pelvis X using long axes
    FE = _signed_angle_about_axis(pZ, fZ, e1)
    # Ab/Ad about floating axis; use pelvis Z vs femur long axis
    # (avoid projecting the axis itself: pY || e2 in this frame, which is ill-posed)
    AbAd = _signed_angle_about_axis(pZ, e3, e2)
    # IE about femur long axis using pelvis/femur X projected ⟂ e3
    pX_perp = pX - e3 * float(np.dot(pX, e3))
    fX_perp = fX - e3 * float(np.dot(fX, e3))
    IE = _signed_angle_about_axis(pX_perp, fX_perp, e3)
    return float(FE), float(AbAd), float(IE)


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


def _knee_angles_grood_suntay(R_femur: np.ndarray, R_tibia: np.ndarray):
    def compute_angles(fX_seed: np.ndarray, Rtib: np.ndarray) -> tuple[float, float, float]:
        femur_X = R_femur[:, 0]
        femur_Y = R_femur[:, 1]
        femur_Z = R_femur[:, 2]
        tibia_X = Rtib[:, 0]
        tibia_Y = Rtib[:, 1]
        tibia_Z = Rtib[:, 2]
        # Use femur Z as long axis (proximal)
        fX = fX_seed - femur_Z * float(np.dot(fX_seed, femur_Z))
        if np.linalg.norm(fX) < 1e-12:
            fX = np.cross(femur_Z, femur_Y)
        fX = fX / (np.linalg.norm(fX) + 1e-12)
        # e3: tibia long axis (tibia Z)
        e3 = tibia_Z / (np.linalg.norm(tibia_Z) + 1e-12)
        # e2 = e3 × e1
        e2 = np.cross(e3, fX)
        if np.linalg.norm(e2) < 1e-12:
            e2 = np.cross(e3, tibia_X)
        e2 = e2 / (np.linalg.norm(e2) + 1e-12)
        # FE about e1; Var/Val about e2; IE about e3
        FE = _signed_angle_about_axis(femur_Z, tibia_Z, fX)
        AbAd = _signed_angle_about_axis(femur_Z, e3, e2)
        IE = _signed_angle_about_axis(fX, tibia_X, e3)
        return FE, AbAd, IE

    def best_with_tibia(Rtib: np.ndarray) -> tuple[float, float, float]:
        # Evaluate both femur X seeds and pick by |AbAd| then |IE|
        FE1, AbAd1, IE1 = compute_angles(R_femur[:, 0], Rtib)
        FE2, AbAd2, IE2 = compute_angles(-R_femur[:, 0], Rtib)
        cand = [(FE1, AbAd1, IE1), (FE2, AbAd2, IE2)]
        cand.sort(key=lambda t: (abs(t[2]), abs(t[1])))
        return cand[0]

    # Evaluate tibia as-is and paired-flipped X&Z (det stays +1)
    Rt_as_is = R_tibia
    Rt_flip = R_tibia.copy()
    Rt_flip[:, [0, 2]] *= -1.0
    c1 = best_with_tibia(Rt_as_is)
    c2 = best_with_tibia(Rt_flip)
    FE, AbAd, IE = min([c1, c2], key=lambda t: (abs(t[2]), abs(t[1])))
    # Apply consistent sign alignment for both sides to match static calibration display:
    # flip FE and Var/Val; keep IE unchanged.
    FE = -FE
    AbAd = -AbAd
    return FE, AbAd, IE


# ---- PLOTS ----
def _find_first_npz_by_suffix(search_dirs: list[str], suffix: str) -> str | None:
    """First file ending with ``suffix`` (e.g. ``_hip_offsets.npz``) under ``search_dirs``."""
    seen: set[str] = set()
    for d in search_dirs:
        if not d or (d in seen) or (not os.path.isdir(d)):
            continue
        seen.add(d)
        try:
            for fname in os.listdir(d):
                if fname.endswith(suffix):
                    return os.path.join(d, fname)
        except OSError:
            continue
    return None


def _norm_c3d_label(s: str) -> str:
    """Normalize label for fuzzy match (same idea as static_calibration._norm_label)."""
    s = str(s).lower().replace("\x00", "").strip()
    return s.replace("_", "").replace("-", "").replace(" ", "")


def _clean_c3d_point_labels(c3d: Any, include_labels2: bool = True) -> list[str]:
    """POINT LABELS (+ optional LABELS2) stripped; same spirit as static_calibration.get_labels_xyz."""
    pt = c3d["parameters"]["POINT"]
    labels = list(pt["LABELS"]["value"])
    if include_labels2 and ("LABELS2" in pt):
        labels += list(pt["LABELS2"]["value"])
    clean: list[str] = []
    for s in labels:
        if isinstance(s, bytes):
            s = s.decode("utf-8", errors="ignore")
        clean.append(str(s).replace("\x00", "").strip())
    return clean


def _c3d_n_points_and_labels(c3d: Any) -> tuple[int, list[str]]:
    """
    Align label count with ``data['points']`` columns using POINT:USED when present.
    Tries LABELS+LABELS2 first; if the marker is not found, caller may retry LABELS-only.
    """
    data = c3d["data"]["points"]
    n_data = int(data.shape[1])
    pt = c3d["parameters"]["POINT"]
    n_used = n_data
    if "USED" in pt:
        try:
            n_used = int(np.asarray(pt["USED"]["value"]).ravel()[0])
        except Exception:
            pass
    n_pts = max(0, min(n_data, n_used))
    labels = _clean_c3d_point_labels(c3d, include_labels2=True)
    if len(labels) > n_pts:
        labels = labels[:n_pts]
    elif len(labels) < n_pts:
        labels = labels + [""] * (n_pts - len(labels))
    return n_pts, labels


def _find_r_knee_lat_index(labels: list[str], n_pts: int) -> int | None:
    """Resolve index for right lateral knee marker (names vary by lab / Vicon)."""
    for preferred in ("R_Knee_Lat", "R_Knee_Lateral", "R_KNEE_LAT"):
        j = _find_marker_index(labels, preferred)
        if j is not None and j < n_pts:
            return j
    for i in range(min(len(labels), n_pts)):
        lab = labels[i]
        if not lab:
            continue
        low = lab.lower().replace(" ", "_")
        if "knee" not in low or "lat" not in low:
            continue
        if low.startswith("r_") or low.startswith("right_") or "r_knee" in low or "_r_" in low:
            return i
        if low.startswith("r") and "knee" in low and "lat" in low:
            return i
    return None


def _discover_walk_c3d(npz_path: str, walk_base: str) -> str | None:
    """Find ``<walk_base>.c3d`` next to the NPZ, parent dirs, or common ``c3d`` folders."""
    npz_dir = os.path.dirname(os.path.abspath(npz_path))
    wb = walk_base.replace(" ", "_")
    roots: list[str] = []
    for r in (
        npz_dir,
        os.path.dirname(npz_dir),
        os.path.join(os.path.dirname(npz_dir), "c3d"),
        os.path.join(npz_dir, "..", "..", "c3d"),
        os.path.join(npz_dir, "..", "..", "..", "c3d"),
    ):
        ar = os.path.abspath(r)
        if ar not in roots and os.path.isdir(ar):
            roots.append(ar)
    for ar in roots:
        for ext in (".c3d", ".C3D"):
            p = os.path.join(ar, walk_base + ext)
            if os.path.isfile(p):
                return p
        wb_low = wb.lower()
        try:
            for fn in os.listdir(ar):
                if not fn.lower().endswith(".c3d"):
                    continue
                stem = os.path.splitext(fn)[0].replace(" ", "_")
                if stem.lower() == wb_low:
                    return os.path.join(ar, fn)
        except OSError:
            continue
        hits = glob.glob(os.path.join(ar, "**", wb + ".c3d"), recursive=True)
        if hits:
            return os.path.abspath(hits[0])
        hits2 = glob.glob(os.path.join(ar, "**", wb + ".C3D"), recursive=True)
        if hits2:
            return os.path.abspath(hits2[0])
    return None


def _find_marker_index(labels: list[str], name: str) -> int | None:
    """Index into C3D point array: exact, case-insensitive, normalized, or colon suffix."""
    if name in labels:
        return labels.index(name)
    lower_map = {lab.lower(): i for i, lab in enumerate(labels) if lab}
    if name.lower() in lower_map:
        return lower_map[name.lower()]
    norm_map = {_norm_c3d_label(lab): i for i, lab in enumerate(labels) if lab}
    key = _norm_c3d_label(name)
    if key in norm_map:
        return norm_map[key]
    for i, lab in enumerate(labels):
        if not lab:
            continue
        s = str(lab).strip()
        if s == name or s.endswith(":" + name) or s.split(":")[-1].strip() == name:
            return i
    return None


def _float3_from_npz_key(z: Any, key: str) -> np.ndarray | None:
    if key not in z.files:
        return None
    a = np.asarray(z[key], dtype=float).ravel()
    if a.size < 3 or not np.isfinite(a[:3]).all():
        return None
    return a[:3].copy()


def _artifact_search_dirs(npz_path: str, argv: list[str]) -> list[str]:
    """Ordered unique directories for templates, foot ACS preload, and offset NPZ files."""
    npz_dir = os.path.dirname(os.path.abspath(npz_path))
    parent = os.path.dirname(npz_dir)
    arg2 = argv[2] if len(argv) >= 3 else None
    tpl_from_argv = arg2 if (arg2 and not str(
        arg2).lower().endswith(".c3d")) else None
    out: list[str] = []
    for d in (
        tpl_from_argv,
        npz_dir,
        os.path.join(parent, "static calib"),
        os.path.join(parent, "scripts", "static calib"),
        os.path.join(npz_dir, "static calib"),
        os.path.join(npz_dir, "scripts", "static calib"),
    ):
        if d and os.path.isdir(d) and d not in out:
            out.append(d)
    return out


def _resolve_static_base_from_dirs(dirs: list[str]) -> str:
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        try:
            for fname in os.listdir(d):
                if fname.endswith("_pelvis_tcs_template.npz"):
                    return fname.replace("_pelvis_tcs_template.npz", "")
        except OSError:
            continue
    return "S_Cal02"


def _ankle_euler_xyz_deg(
    R_shank_f: np.ndarray,
    R_foot_acs_f: np.ndarray,
    T_foot_f: np.ndarray | None,
    tpl: dict | None,
) -> tuple[float, float, float]:
    """Tibia–foot relative XYZ Euler (deg); same pathway for left and right."""
    R_ti = _enforce_right_handed(np.asarray(R_shank_f, dtype=float))
    R_fo_dyn = np.asarray(R_foot_acs_f, dtype=float)
    try:
        if (
            T_foot_f is not None
            and isinstance(T_foot_f, np.ndarray)
            and T_foot_f.size
            and tpl is not None
            and ("C_TCS_to_ACS" in tpl)
        ):
            C = np.asarray(tpl["C_TCS_to_ACS"], dtype=float)
            R_fo_dyn = C @ T_foot_f[:3, :3]
    except Exception:
        R_fo_dyn = np.asarray(R_foot_acs_f, dtype=float)
    R_fo = _enforce_right_handed(R_fo_dyn)
    if float(np.dot(R_ti[:, 0], R_fo[:, 0])) < 0.0:
        R_fo = R_fo.copy()
        R_fo[:, [0, 2]] *= -1.0
    R_rel = R_ti.T @ R_fo
    ax, ay, az = _euler_xyz_from_R(R_rel)
    ax = -ax
    if ax < -90.0:
        ax += 180.0
    elif ax > 90.0:
        ax -= 180.0
    ax = -ax
    return float(ax), float(ay), float(az)


def _apply_hip_knee_offsets_for_side(
    hip: np.ndarray,
    knee: np.ndarray,
    hip_path: str | None,
    knee_path: str | None,
    side_key: str,
    log_label: str,
) -> None:
    """Subtract static-trial ``left``/``right`` (3,) offsets from hip and knee columns in-place."""
    if hip_path and os.path.isfile(hip_path):
        z = np.load(hip_path, allow_pickle=True)
        h = _float3_from_npz_key(z, side_key)
        if h is not None:
            hip[:, :3] -= h
            print(f"Subtracted hip static offsets ({
                  log_label}) from {hip_path}: {h}")
        else:
            print(f"Note: no valid '{side_key}' (3,) in {
                  hip_path}; hip ({log_label}) not offset.")
    if knee_path and os.path.isfile(knee_path):
        z = np.load(knee_path, allow_pickle=True)
        k = _float3_from_npz_key(z, side_key)
        if k is not None:
            knee[:, :3] -= k
            print(f"Subtracted knee GS static offsets ({
                  log_label}) from {knee_path}: {k}")
        else:
            print(f"Note: no valid '{side_key}' (3,) in {
                  knee_path}; knee ({log_label}) not offset.")


def _preload_foot_tpl_acs(npz_path: str, argv: list[str]) -> tuple[dict | None, dict | None]:
    """
    Load optional ``C_TCS_to_ACS`` from foot templates so the angle loop can rebuild
    foot ACS from global T before Euler ankle angles (matches static_calibration intent).
    """
    candidates = _artifact_search_dirs(npz_path, argv)
    static_base = _resolve_static_base_from_dirs(candidates)

    def _load_c_only(basename: str) -> dict | None:
        filename = f"{basename}_tcs_template.npz"
        for d in candidates:
            path = os.path.join(d, filename)
            if not os.path.isfile(path):
                continue
            t = np.load(path, allow_pickle=True)
            if "C_TCS_to_ACS" not in t.files:
                return None
            return {"C_TCS_to_ACS": np.asarray(t["C_TCS_to_ACS"], dtype=float)}
        return None

    tpl_l = _load_c_only(f"{static_base}_foot")
    tpl_r = _load_c_only(f"{static_base}_foot_right")
    return tpl_l, tpl_r


def _plotly_scatter3d_frame_dict(Q: np.ndarray | None, f: int) -> dict[str, Any]:
    """Single-frame update dict for Plotly Scatter3d from (F, N, 3) or empty."""
    if Q is None:
        return {"type": "scatter3d", "x": [], "y": [], "z": [], "mode": "markers"}
    pts = Q[f]
    return {"type": "scatter3d", "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2], "mode": "markers"}


def _plotly_animation_layout_extras(F: int) -> dict[str, Any]:
    """Shared updatemenus + sliders for frame-driven Plotly figures."""
    return {
        "updatemenus": [{
            "type": "buttons",
            "buttons": [
                {"label": "Play", "method": "animate", "args": [
                    None, {"frame": {"duration": 30, "redraw": True}, "fromcurrent": True}]},
                {"label": "Pause", "method": "animate", "args": [
                    [None], {"frame": {"duration": 0}, "mode": "immediate"}]},
            ],
            "showactive": True,
        }],
        "sliders": [{
            "steps": [
                {"args": [[str(k)], {"frame": {"duration": 0, "redraw": True},
                                     "mode": "immediate"}], "label": str(k), "method": "animate"}
                for k in range(F)
            ],
            "currentvalue": {"prefix": "Frame index: "},
        }],
    }


def _apply_T_series(T_series: np.ndarray, C_local: np.ndarray) -> np.ndarray:
    """Map TCS marker locals through global rigid motion: ``T_series`` (F,4,4), ``C_local`` (N,3) → (F,N,3)."""
    nF = T_series.shape[0]
    N = C_local.shape[0]
    Q = np.zeros((nF, N, 3), dtype=float)
    C = C_local.astype(float)
    for fi in range(nF):
        T = T_series[fi]
        R = T[:3, :3]
        t = T[:3, 3]
        Q[fi] = (R @ C.T).T + t
    return Q


def _find_tcs_template_file(search_dirs: list[str], stem: str) -> str | None:
    filename = f"{stem}_tcs_template.npz"
    for d in search_dirs:
        path = os.path.join(d, filename)
        if os.path.isfile(path):
            return path
    return None


def _load_tcs_template(search_dirs: list[str], stem: str) -> Dict[str, Any] | None:
    path = _find_tcs_template_file(search_dirs, stem)
    if path is None:
        print(f"Warning: template not found for {
              stem}; skipping markers for {stem}")
        return None
    tpl = np.load(path, allow_pickle=True)
    out: Dict[str, Any] = {
        "labels": tpl["marker_labels"].tolist(),
        "C_local": tpl["C_local"],
    }
    if "C_TCS_to_ACS" in tpl:
        out["C_TCS_to_ACS"] = tpl["C_TCS_to_ACS"]
    return out


def _c3d_indices_for_marker_list(names: list[str], labels: list[str], n_pts: int) -> list[int]:
    out: list[int] = []
    for m in names:
        j = _find_marker_index(labels, m)
        if j is not None and j < n_pts:
            out.append(j)
    return out


def _plotly_empty_segment_scatter(color: str, opacity: float = 0.9, name: str = "") -> Any:
    return go.Scatter3d(
        x=[], y=[], z=[], mode="markers",
        marker=dict(size=4, color=color, opacity=opacity), name=name, showlegend=False,
    )


def _plotly_r_knee_lat_placeholder() -> Any:
    return go.Scatter3d(
        x=[], y=[], z=[], mode="markers",
        marker=dict(size=5, color="#c41e1e", opacity=1.0,
                    symbol="circle", line=dict(width=0.5, color="#7a0e0e")),
        name="R_Knee_Lat",
        showlegend=False,
    )


def _plotly_add_angle_line_specs(fig: Any, x: np.ndarray, line_specs: list[tuple], row: int, col: int) -> None:
    for yv, nm, clr in line_specs:
        fig.add_trace(
            go.Scatter(x=x, y=yv, mode="lines", name=nm,
                       line=dict(color=clr, width=2)),
            row=row, col=col,
        )


def _plotly_add_angle_marker_dots(fig: Any, line_specs: list[tuple], row: int, col: int) -> list[int]:
    start = len(fig.data)
    for _, _, color in line_specs:
        fig.add_trace(
            go.Scatter(
                x=[0], y=[np.nan], mode="markers",
                marker=dict(size=11, color=color, line=dict(
                    width=1.5, color="#ffffff")),
                showlegend=False,
                name="",
            ),
            row=row, col=col,
        )
    return list(range(start, len(fig.data)))


def _plotly_right_view_frame_updates(
    f: int,
    Q_pel: np.ndarray | None,
    Q_thi_r: np.ndarray | None,
    Q_sha_r: np.ndarray | None,
    Q_foo_r: np.ndarray | None,
    Q_thi: np.ndarray | None,
    Q_sha: np.ndarray | None,
    Q_foo: np.ndarray | None,
    Q_rk_lat: np.ndarray | None,
    hipR: np.ndarray,
    kneeR: np.ndarray,
    ankR: np.ndarray,
) -> list[dict[str, Any]]:
    """One animation frame: pelvis, right (TCS or C3D), left context, R_Knee_Lat, angle markers."""
    data_updates = [
        _plotly_scatter3d_frame_dict(Q_pel, f),
        _plotly_scatter3d_frame_dict(Q_thi_r, f),
        _plotly_scatter3d_frame_dict(Q_sha_r, f),
        _plotly_scatter3d_frame_dict(Q_foo_r, f),
        _plotly_scatter3d_frame_dict(Q_thi, f),
        _plotly_scatter3d_frame_dict(Q_sha, f),
        _plotly_scatter3d_frame_dict(Q_foo, f),
        _plotly_scatter3d_frame_dict(Q_rk_lat, f),
    ]
    for val in (ankR[f, 0], kneeR[f, 0], kneeR[f, 1], hipR[f, 0]):
        data_updates.append(
            {"type": "scatter", "x": [f], "y": [val], "mode": "markers"})
    return data_updates


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


def main():
    if len(sys.argv) < 2:
        npz_path = "C:/Users/lmcam/Documents/Grad project/scripts/static calib/subject 02 - S_Cal02/Walk_R04_bilateral_chain_results.npz"
        print(f"Using default results: {npz_path}")
    else:
        npz_path = sys.argv[1]
    npz_path = os.path.abspath(npz_path)
    npz_dir = os.path.dirname(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    # Extract ACS rotations
    R_pelvis = data["pelvis_acs_R"]  # (F,3,3)
    R_femur = data["l_thigh_acs_R"]
    R_tibia = data["l_shank_acs_R"]
    R_ankle = data["l_foot_acs_R"]
    have_right = all(k in data.files for k in (
        "r_thigh_acs_R", "r_shank_acs_R", "r_foot_acs_R"))
    R_femur_R = data["r_thigh_acs_R"] if have_right else None
    R_tibia_R = data["r_shank_acs_R"] if have_right else None
    R_ankle_R = data["r_foot_acs_R"] if have_right else None
    # Extract global TCS transforms (for marker reconstruction)
    T_pelvis = data["pelvis_global"]
    T_thigh = data["l_thigh_global"]
    T_shank = data["l_shank_global"]
    T_foot = data["l_foot_global"]
    # Right-side global transforms if available
    T_thigh_R = data["r_thigh_global"] if (
        "r_thigh_global" in data.files) else None
    T_shank_R = data["r_shank_global"] if (
        "r_shank_global" in data.files) else None
    T_foot_R = data["r_foot_global"] if (
        "r_foot_global" in data.files) else None

    F = int(R_pelvis.shape[0])
    angles = {"hipL": [], "kneeL": [], "ankleL": []}
    if have_right:
        angles.update({"hipR": [], "kneeR": [], "ankleR": []})

    tpl_foo, tpl_foo_r = _preload_foot_tpl_acs(npz_path, sys.argv)

    for f in range(F):
        pel = R_pelvis[f]
        thi = R_femur[f]
        sha = R_tibia[f]
        foo = R_ankle[f]
        if np.isnan(pel).any() or np.isnan(thi).any() or np.isnan(sha).any() or np.isnan(foo).any():
            angles["hipL"].append([np.nan, np.nan, np.nan])
            angles["kneeL"].append([np.nan, np.nan, np.nan])
            angles["ankleL"].append([np.nan, np.nan, np.nan])
            if have_right:
                angles["hipR"].append([np.nan, np.nan, np.nan])
                angles["kneeR"].append([np.nan, np.nan, np.nan])
                angles["ankleR"].append([np.nan, np.nan, np.nan])
            continue
        # Hip: ISB Joint Coordinate System (pelvis–femur)
        FE, AbAd, IE = hip_angles_isb(pel, thi)
        # Invert left hip Ab/Ad and IE post-calculation
        AbAd = -AbAd
        IE = -IE
        angles["hipL"].append([FE, AbAd, IE])

        # Knee: use existing GS with current ACS (legacy fallback here)
        kFE, kVV, kIE = _knee_angles_grood_suntay(thi, sha)
        angles["kneeL"].append([kFE, kVV, kIE])

        Tf = T_foot[f] if (isinstance(T_foot, np.ndarray)
                           and T_foot.size) else None
        ax, ay, az = _ankle_euler_xyz_deg(sha, foo, Tf, tpl_foo)
        if f in (90, 250):
            try:
                np.set_printoptions(precision=6, suppress=True)
                R_ti = _enforce_right_handed(np.asarray(sha, dtype=float))
                R_dbg = _enforce_right_handed(
                    (np.asarray(tpl_foo["C_TCS_to_ACS"], float) @ Tf[:3, :3])
                    if (Tf is not None and tpl_foo and "C_TCS_to_ACS" in tpl_foo)
                    else np.asarray(foo, dtype=float)
                )
                if float(np.dot(R_ti[:, 0], R_dbg[:, 0])) < 0.0:
                    R_dbg = R_dbg.copy()
                    R_dbg[:, [0, 2]] *= -1.0
                print(f"\n--- Ankle debug (frame {f}) ---")
                print("R_tibia (raw):\n", sha)
                print("R_foot  (raw):\n", foo)
                print("R_tibia (aligned RH):\n", R_ti)
                print("R_foot  (aligned RH, XZ-paired-flip if applied):\n", R_dbg)
                print("R_rel = R_tibia.T @ R_foot:\n", R_ti.T @ R_dbg)
            except Exception as _e:
                print(f"Debug print failed at frame {f}: {_e}")
        angles["ankleL"].append([ax, ay, az])
        # Right side (if present)
        if have_right:
            thiR = R_femur_R[f]
            shaR = R_tibia_R[f]
            fooR = R_ankle_R[f]
            if np.isnan(thiR).any() or np.isnan(shaR).any() or np.isnan(fooR).any():
                angles["hipR"].append([np.nan, np.nan, np.nan])
                angles["kneeR"].append([np.nan, np.nan, np.nan])
                angles["ankleR"].append([np.nan, np.nan, np.nan])
            else:
                # Hip (right): ISB JCS
                FE_r, AbAd_r, IE_r = hip_angles_isb(pel, thiR)
                angles["hipR"].append([FE_r, AbAd_r, IE_r])
                # Knee (right): GS + sign alignment identical to static_calibration.py display
                kFE_r, kVV_r, kIE_r = _knee_angles_grood_suntay(thiR, shaR)
                angles["kneeR"].append([kFE_r, kVV_r, kIE_r])
                Tf_r = (
                    T_foot_R[f]
                    if (T_foot_R is not None and isinstance(T_foot_R, np.ndarray) and T_foot_R.size)
                    else None
                )
                ax_r, ay_r, az_r = _ankle_euler_xyz_deg(
                    shaR, fooR, Tf_r, tpl_foo_r)
                angles["ankleR"].append([ax_r, ay_r, az_r])

    out_base = os.path.splitext(os.path.basename(npz_path))[0].replace(
        "_left_chain_results", "").replace("_bilateral_chain_results", "")
    walk_base = out_base
    # Save CSV (apply static trial offsets to match ``static_calibration`` zeroing)
    hip = np.array(angles["hipL"], dtype=float)
    knee = np.array(angles["kneeL"], dtype=float)
    ank = np.array(angles["ankleL"], dtype=float)
    offset_search_dirs = _artifact_search_dirs(npz_path, sys.argv)
    hip_off_path = _find_first_npz_by_suffix(
        offset_search_dirs, "_hip_offsets.npz")
    knee_off_path = _find_first_npz_by_suffix(
        offset_search_dirs, "_knee_gs_offsets.npz")

    try:
        _apply_hip_knee_offsets_for_side(
            hip, knee, hip_off_path, knee_off_path, "left", "left")
    except Exception as e:
        print(f"Warning: could not subtract left hip/knee static offsets: {e}")

    angles["hipL"] = hip.tolist()
    angles["kneeL"] = knee.tolist()
    ang_csv = os.path.join(npz_dir, f"{out_base}_angles_left.csv")
    hdr = "frame,hip_FE,hip_AbAd,hip_IE,knee_FE,knee_VarVal,knee_Axial,ankle_x,ankle_y,ankle_z"
    frames = np.arange(hip.shape[0])
    data_csv = np.column_stack([frames, hip, knee, ank])
    np.savetxt(ang_csv, data_csv, delimiter=",",
               header=hdr, comments="", fmt="%.6f")
    print(f"Saved angles CSV: {os.path.abspath(ang_csv)}")
    # Save right angles CSV if available
    if have_right:
        hipR = np.array(angles["hipR"], dtype=float)
        kneeR = np.array(angles["kneeR"], dtype=float)
        ankR = np.array(angles["ankleR"], dtype=float)
        try:
            _apply_hip_knee_offsets_for_side(
                hipR, kneeR, hip_off_path, knee_off_path, "right", "right")
        except Exception as e:
            print(
                f"Warning: could not subtract right hip/knee static offsets: {e}")
        angles["hipR"] = hipR.tolist()
        angles["kneeR"] = kneeR.tolist()
        framesR = np.arange(hipR.shape[0])
        hdrR = "frame,hip_FE,hip_AbAd,hip_IE,knee_FE,knee_VarVal,knee_Axial,ankle_x,ankle_y,ankle_z"
        data_csv_R = np.column_stack([framesR, hipR, kneeR, ankR])
        ang_csv_R = os.path.join(npz_dir, f"{out_base}_angles_right.csv")
        np.savetxt(ang_csv_R, data_csv_R, delimiter=",",
                   header=hdrR, comments="", fmt="%.6f")
        print(f"Saved angles CSV (right): {os.path.abspath(ang_csv_R)}")

    # Quick plots
    _plot_angle_triplet(angles["hipL"],   f"{out_base} hip (ISB JCS)", [
                        "FE", "AbAd", "IE"], os.path.join(npz_dir, f"{out_base}_hip_angles.png"))
    _plot_angle_triplet(angles["kneeL"],  f"{out_base} knee (GS)", [
                        "FE", "VarVal", "Axial"], os.path.join(npz_dir, f"{out_base}_knee_angles.png"))
    _plot_angle_triplet(angles["ankleL"], f"{out_base} ankle (XYZ)", [
                        "x", "y", "z"], os.path.join(npz_dir, f"{out_base}_ankle_angles.png"))
    if have_right:
        _plot_angle_triplet(angles["hipR"],   f"{out_base} hip (ISB JCS) R", [
                            "FE", "AbAd", "IE"], os.path.join(npz_dir, f"{out_base}_hip_angles_right.png"))
        _plot_angle_triplet(angles["kneeR"],  f"{out_base} knee (GS) R", [
                            "FE", "VarVal", "Axial"], os.path.join(npz_dir, f"{out_base}_knee_angles_right.png"))
        _plot_angle_triplet(angles["ankleR"], f"{out_base} ankle (XYZ) R", [
                            "x", "y", "z"], os.path.join(npz_dir, f"{out_base}_ankle_angles_right.png"))

    # Optional interactive 3D + angles viewer using Plotly
    if not HAVE_PLOTLY:
        print("Plotly not installed; skipping interactive 3D viewer.")
        return

    search_dirs = _artifact_search_dirs(npz_path, sys.argv)
    static_base = _resolve_static_base_from_dirs(search_dirs)

    tpl_pel = _load_tcs_template(search_dirs, f"{static_base}_pelvis")
    tpl_thi = _load_tcs_template(search_dirs, f"{static_base}_femur")
    tpl_sha = _load_tcs_template(search_dirs, f"{static_base}_tibia")
    tpl_foo = _load_tcs_template(search_dirs, f"{static_base}_foot")
    tpl_thi_r = _load_tcs_template(search_dirs, f"{static_base}_femur_right")
    tpl_sha_r = _load_tcs_template(search_dirs, f"{static_base}_tibia_right")
    tpl_foo_r = _load_tcs_template(search_dirs, f"{static_base}_foot_right")

    Q_pel = _apply_T_series(T_pelvis, tpl_pel["C_local"]) if (
        T_pelvis.size and tpl_pel is not None) else None
    Q_thi = _apply_T_series(T_thigh,  tpl_thi["C_local"]) if (
        T_thigh.size and tpl_thi is not None) else None
    Q_sha = _apply_T_series(T_shank,  tpl_sha["C_local"]) if (
        T_shank.size and tpl_sha is not None) else None
    Q_foo = _apply_T_series(T_foot,   tpl_foo["C_local"]) if (
        T_foot.size and tpl_foo is not None) else None
    # Right-side reconstructions from TCS (prefer these over raw C3D if available)
    Q_thi_R_tcs = _apply_T_series(T_thigh_R, tpl_thi_r["C_local"]) if (
        T_thigh_R is not None and tpl_thi_r is not None) else None
    Q_sha_R_tcs = _apply_T_series(T_shank_R, tpl_sha_r["C_local"]) if (
        T_shank_R is not None and tpl_sha_r is not None) else None
    Q_foo_R_tcs = _apply_T_series(T_foot_R,  tpl_foo_r["C_local"]) if (
        T_foot_R is not None and tpl_foo_r is not None) else None

    # Optional: load right-side raw markers from C3D (+ lateral knee landmark)
    Q_right_thi = Q_right_sha = Q_right_foo = None
    Q_right_knee_lat: np.ndarray | None = None
    walk_c3d_path: str | None = None
    if len(sys.argv) >= 3 and str(sys.argv[2]).lower().endswith(".c3d"):
        walk_c3d_path = os.path.abspath(sys.argv[2])
    if walk_c3d_path is None and len(sys.argv) >= 4 and str(sys.argv[3]).lower().endswith(".c3d"):
        walk_c3d_path = os.path.abspath(sys.argv[3])
    if walk_c3d_path is None:
        local_c3d = os.path.join(npz_dir, f"{walk_base}.c3d")
        if os.path.isfile(local_c3d):
            walk_c3d_path = os.path.abspath(local_c3d)
    if walk_c3d_path is None:
        walk_c3d_path = _discover_walk_c3d(npz_path, walk_base)

    if not HAVE_C3D:
        print("Plotly 3D: ezc3d not installed — cannot load C3D markers (R_Knee_Lat).")
    elif walk_c3d_path is None or (not os.path.isfile(walk_c3d_path)):
        print(
            "Plotly 3D: no walk C3D found for markers. Tried argv .c3d, folder "
            f"{walk_base}.c3d next to the NPZ, and discover_walk_c3d(). Pass the trial explicitly, e.g.\n"
            f"  python angles_only.py \"{
                npz_path}\" \"C:/path/{walk_base}.c3d\""
        )
    else:
        try:
            c3d = ezc3d.c3d(walk_c3d_path)
            pts = np.transpose(c3d["data"]["points"]
                               [:3, :, :], (2, 1, 0))  # (F,N,3)
            n_pts, labels = _c3d_n_points_and_labels(c3d)
            pts = pts[:, :n_pts, :]
            RIGHT_THIGH = ["R_Thigh_PS", "R_Thigh_AS",
                           "R_Thigh_PI", "R_Thigh_AI"]
            RIGHT_SHANK = ["R_Shank_AS", "R_Shank_PS",
                           "R_Shank_AI", "R_Shank_PI"]
            RIGHT_FOOT = ["R_Calc", "R_Ank_Lat",
                          "R_Midfoot_Sup", "R_Midfoot_Lat"]

            idx_thi = _c3d_indices_for_marker_list(RIGHT_THIGH, labels, n_pts)
            idx_sha = _c3d_indices_for_marker_list(RIGHT_SHANK, labels, n_pts)
            idx_foo = _c3d_indices_for_marker_list(RIGHT_FOOT, labels, n_pts)
            if len(idx_thi) >= 1:
                Q_right_thi = pts[:, idx_thi, :].astype(float)
            if len(idx_sha) >= 1:
                Q_right_sha = pts[:, idx_sha, :].astype(float)
            if len(idx_foo) >= 1:
                Q_right_foo = pts[:, idx_foo, :].astype(float)

            ik = _find_r_knee_lat_index(labels, n_pts)
            if ik is None:
                labels_l1 = _clean_c3d_point_labels(c3d, include_labels2=False)
                n1 = min(len(labels_l1), n_pts)
                ik = _find_r_knee_lat_index(labels_l1[:n1], n1)

            if ik is not None:
                Q_right_knee_lat = pts[:, ik: ik + 1, :].astype(float)
                Fc3d = int(Q_right_knee_lat.shape[0])
                if Fc3d < F:
                    pad = np.full((F, 1, 3), np.nan, dtype=float)
                    pad[:Fc3d] = Q_right_knee_lat
                    Q_right_knee_lat = pad
                elif Fc3d > F:
                    Q_right_knee_lat = Q_right_knee_lat[:F]
                print(f"Plotly 3D: loaded R_Knee_Lat from {
                      walk_c3d_path} (point index {ik}).")
            else:
                knee_lat_cands = [
                    lab for lab in labels
                    if lab and "knee" in lab.lower() and "lat" in lab.lower()
                ]
                print(
                    "Plotly 3D: R_Knee_Lat not matched. C3D: {} | labels with 'knee'+'lat': {}".format(
                        walk_c3d_path,
                        knee_lat_cands or "(none)",
                    )
                )
        except Exception as e:
            print(f"Could not load right-side markers from C3D: {e}")

    # One row: 3D scene | angle traces (reduces empty margins vs stacked 2×1)
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        horizontal_spacing=0.05,
        column_widths=[0.48, 0.52],
    )

    P = PLOTLY_VIEWER_PALETTE
    AC = PLOTLY_VIEWER_ANGLE_COLORS
    fig.add_trace(_plotly_empty_segment_scatter(
        P["pelvis"], opacity=0.9, name="Pelvis (L)"), row=1, col=1)
    fig.add_trace(_plotly_empty_segment_scatter(
        P["hip"], opacity=0.9, name="Thigh (L)"), row=1, col=1)
    fig.add_trace(_plotly_empty_segment_scatter(
        P["knee"], opacity=0.9, name="Shank (L)"), row=1, col=1)
    fig.add_trace(_plotly_empty_segment_scatter(
        P["foot"], opacity=0.9, name="Foot (L)"), row=1, col=1)
    fig.add_trace(_plotly_empty_segment_scatter(
        P["hip"], opacity=0.5, name="Thigh (R)"), row=1, col=1)
    fig.add_trace(_plotly_empty_segment_scatter(
        P["knee"], opacity=0.5, name="Shank (R)"), row=1, col=1)
    fig.add_trace(_plotly_empty_segment_scatter(
        P["foot"], opacity=0.5, name="Foot (R)"), row=1, col=1)
    fig.add_trace(_plotly_r_knee_lat_placeholder(), row=1, col=1)
    idx_3d = [0, 1, 2, 3, 4, 5, 6, 7]

    x = np.arange(F)
    hip = np.asarray(hip, dtype=float)
    knee = np.asarray(knee, dtype=float)
    ank = np.asarray(ank, dtype=float)
    line_specs_left = [
        (ank[:, 0], "Ankle PF/DF (L)", AC["ankle_pfdf"]),
        (knee[:, 0], "Knee FE (L)", AC["knee_fe"]),
        (knee[:, 1], "Knee Var/Val (L)", AC["knee_varval"]),
        (hip[:, 0], "Hip FE (L)", AC["hip_fe"]),
    ]
    _plotly_add_angle_line_specs(fig, x, line_specs_left, row=1, col=2)
    idx_markers = _plotly_add_angle_marker_dots(
        fig, line_specs_left, row=1, col=2)

    frames = []
    for f in range(F):
        data_updates = [
            _plotly_scatter3d_frame_dict(Q, f)
            for Q in (Q_pel, Q_thi, Q_sha, Q_foo, Q_right_thi, Q_right_sha, Q_right_foo, Q_right_knee_lat)
        ]
        for val in (ank[f, 0], knee[f, 0], knee[f, 1], hip[f, 0]):
            data_updates.append(
                {"type": "scatter", "x": [f], "y": [val], "mode": "markers"})
        frames.append(go.Frame(data=data_updates,
                      traces=idx_3d + idx_markers, name=str(f)))

    fig.frames = frames
    fig.update_layout(
        title_text=f"Joint Angles: Hip, Knee & Ankle ({
            'Right' if have_right else 'Left'} Session, Left View)",
        width=1400,
        height=560,
        margin=dict(l=4, r=120, t=48, b=72),
        scene=dict(xaxis_title="Y", yaxis_title="X",
                   zaxis_title="Z", aspectmode="data"),
        xaxis=dict(title="Frame"),
        yaxis=dict(title="deg"),
        legend=dict(x=1.01, y=1.0, xanchor="left", yanchor="top"),
        **_plotly_animation_layout_extras(F),
    )

    html_path = os.path.join(npz_dir, f"{out_base}_viewer.html")
    try:
        pio.write_html(fig, file=html_path, auto_open=True)
        print(f"Saved interactive viewer: {os.path.abspath(html_path)}")
    except Exception as e:
        print(f"Failed to save interactive viewer: {e}")
    # Separate right-only angles HTML (feature-parity with left; no overlay)
    if have_right:
        try:
            figR = make_subplots(
                rows=1,
                cols=2,
                specs=[[{"type": "scene"}, {"type": "xy"}]],
                horizontal_spacing=0.05,
                column_widths=[0.48, 0.52],
            )
            figR.add_trace(_plotly_empty_segment_scatter(
                P["pelvis"], opacity=0.9, name="Pelvis"), row=1, col=1)
            figR.add_trace(_plotly_empty_segment_scatter(
                P["hip"], opacity=0.9, name="Thigh (R)"), row=1, col=1)
            figR.add_trace(_plotly_empty_segment_scatter(
                P["knee"], opacity=0.9, name="Shank (R)"), row=1, col=1)
            figR.add_trace(_plotly_empty_segment_scatter(
                P["foot"], opacity=0.9, name="Foot (R)"), row=1, col=1)
            figR.add_trace(_plotly_empty_segment_scatter(
                P["hip"], opacity=0.4, name="Thigh (L)"), row=1, col=1)
            figR.add_trace(_plotly_empty_segment_scatter(
                P["knee"], opacity=0.4, name="Shank (L)"), row=1, col=1)
            figR.add_trace(_plotly_empty_segment_scatter(
                P["foot"], opacity=0.4, name="Foot (L)"), row=1, col=1)
            figR.add_trace(_plotly_r_knee_lat_placeholder(), row=1, col=1)
            idx3d_R = [0, 1, 2, 3, 4, 5, 6, 7]

            hipR = np.asarray(hipR, dtype=float)
            kneeR = np.asarray(kneeR, dtype=float)
            ankR = np.asarray(ankR, dtype=float)
            line_specs_right = [
                (ankR[:, 0], "Ankle PF/DF (R)", AC["ankle_pfdf"]),
                (kneeR[:, 0], "Knee FE (R)", AC["knee_fe"]),
                (kneeR[:, 1], "Knee Var/Val (R)", AC["knee_varval"]),
                (hipR[:, 0], "Hip FE (R)", AC["hip_fe"]),
            ]
            _plotly_add_angle_line_specs(
                figR, x, line_specs_right, row=1, col=2)
            idx_markers_R = _plotly_add_angle_marker_dots(
                figR, line_specs_right, row=1, col=2)

            Qr_thi = Q_thi_R_tcs if Q_thi_R_tcs is not None else Q_right_thi
            Qr_sha = Q_sha_R_tcs if Q_sha_R_tcs is not None else Q_right_sha
            Qr_foo = Q_foo_R_tcs if Q_foo_R_tcs is not None else Q_right_foo
            frames_R = [
                go.Frame(
                    data=_plotly_right_view_frame_updates(
                        f, Q_pel, Qr_thi, Qr_sha, Qr_foo, Q_thi, Q_sha, Q_foo, Q_right_knee_lat,
                        hipR, kneeR, ankR,
                    ),
                    traces=idx3d_R + idx_markers_R,
                    name=str(f),
                )
                for f in range(F)
            ]
            figR.frames = frames_R
            figR.update_layout(
                title_text="Joint Angles: Hip, Knee & Ankle (Right View)",
                width=1400,
                height=560,
                margin=dict(l=4, r=120, t=48, b=72),
                scene=dict(xaxis_title="Y", yaxis_title="X",
                           zaxis_title="Z", aspectmode="data"),
                xaxis=dict(title="Frame"),
                yaxis=dict(title="deg"),
                legend=dict(x=1.01, y=1.0, xanchor="left", yanchor="top"),
                **_plotly_animation_layout_extras(F),
            )
            html_path_R = os.path.join(
                npz_dir, f"{out_base}_angles_right.html")
            pio.write_html(figR, file=html_path_R, auto_open=True)
            print(
                f"Saved right-only angles viewer: {os.path.abspath(html_path_R)}")
        except Exception as e:
            print(f"Failed to save right angles HTML: {e}")


if __name__ == "__main__":
    main()
