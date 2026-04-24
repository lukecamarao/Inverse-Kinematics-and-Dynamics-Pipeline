# -*- coding: utf-8 -*-
"""
Plotly viewer: foot, shank, and thigh raw markers + foot ACS triad (3D), shank ACS at knee (3D), ankle moment
and knee Grood–Suntay JCS moments vs time (2D), frame-synced via a slider / animation.

Loads:
  - Walk C3D (ezc3d): foot marker trajectories in lab frame (mm).
  - ``*_bilateral_chain_results.npz``: ``r_foot_acs_R`` / ``r_foot_acs_O``, ``r_shank_acs_*``
    (or ``l_*``).
  - ``*_foot_ankle_inverse_dynamics.npz``: ``R_M_joint_ankle_angle_frame_Nm`` + ``R_time`` (or ``L_*``).
  - ``*_leg_inverse_dynamics.npz`` (optional): ``R_M_knee_jcs_Nm`` for knee FE / Ab–Ad / IE
    (default path derived from foot ID NPZ basename).

Example::

    python plot_foot_moment_plotly.py ^
      --c3d "../../c3d/subject 02/Walk_R04.c3d" ^
      --bilateral "subject 02 - S_Cal02/Walk_R04_bilateral_chain_results.npz" ^
      --id-npz "subject 02 - S_Cal02/Walk_R04_foot_ankle_inverse_dynamics.npz" ^
      --leg-npz "subject 02 - S_Cal02/Walk_R04_leg_inverse_dynamics.npz" ^
      --side R ^
      --out "subject 02 - S_Cal02/Walk_R04_foot_moment_viewer.html"

Run with **no arguments** to use default paths next to this script
(``subject 02 - S_Cal02/Walk_R04_*`` and ``../../c3d/subject 02/Walk_R04.c3d``).

@author: lmcam + assistant
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# Default trial basename when CLI paths are omitted (same layout as inverse_dynamics_newton_euler.main).
DEFAULT_WALK_BASE = "Walk_R04"


def _default_input_paths(
    script_dir: str, walk_base: str = DEFAULT_WALK_BASE
) -> tuple[str | None, str | None, str | None, str | None]:
    """Return (c3d, bilateral, id_npz, leg_npz) for default subject/trial if those files exist."""
    subject_dir = os.path.join(script_dir, "subject 02 - S_Cal02")
    bilateral = os.path.join(
        subject_dir, f"{walk_base}_bilateral_chain_results.npz")
    id_npz = os.path.join(
        subject_dir, f"{walk_base}_foot_ankle_inverse_dynamics.npz")
    leg_npz = os.path.join(
        subject_dir, f"{walk_base}_leg_inverse_dynamics.npz")
    c3d_candidates = [
        os.path.join(script_dir, "..", "..", "c3d",
                     "subject 02", f"{walk_base}.c3d"),
        os.path.join(script_dir, "..", "..", "c3d",
                     "subject 02", "grf", f"{walk_base}.c3d"),
        os.path.join(subject_dir, f"{walk_base}.c3d"),
    ]
    c3d: str | None = None
    for c in c3d_candidates:
        ac = os.path.abspath(c)
        if os.path.isfile(ac):
            c3d = ac
            break
    bi_p = os.path.abspath(bilateral) if os.path.isfile(bilateral) else None
    id_p = os.path.abspath(id_npz) if os.path.isfile(id_npz) else None
    leg_p = os.path.abspath(leg_npz) if os.path.isfile(leg_npz) else None
    return c3d, bi_p, id_p, leg_p


def _leg_npz_from_foot_id_path(foot_id_npz: str) -> str | None:
    """``..._foot_ankle_inverse_dynamics.npz`` → ``..._leg_inverse_dynamics.npz`` in same folder."""
    bn = os.path.basename(foot_id_npz)
    suf = "_foot_ankle_inverse_dynamics.npz"
    if not bn.endswith(suf):
        return None
    base = bn[: -len(suf)]
    cand = os.path.join(os.path.dirname(os.path.abspath(foot_id_npz)), f"{
                        base}_leg_inverse_dynamics.npz")
    return cand if os.path.isfile(cand) else None


# Foot marker sets (same naming as svd_kabsch / forceplate_preprocess)
RIGHT_FOOT_MARKERS = ["R_Calc", "R_Ank_Lat", "R_Midfoot_Sup", "R_Midfoot_Lat"]
LEFT_FOOT_MARKERS = ["L_Calc", "L_Ank_Lat", "L_Midfoot_Sup", "L_Midfoot_Lat"]
RIGHT_FOOT_MARKER_INDICES = [0, 1, 2, 3]
LEFT_FOOT_MARKER_INDICES = [0, 1, 2, 3]

# Same names as svd_kabsch / angles_only (lab raw markers in walk C3D)
RIGHT_SHANK_MARKERS = ["R_Shank_AS", "R_Shank_PS", "R_Shank_AI", "R_Shank_PI"]
LEFT_SHANK_MARKERS = ["L_Shank_AS", "L_Shank_PS", "L_Shank_AI", "L_Shank_PI"]
RIGHT_THIGH_MARKERS = ["R_Thigh_PS", "R_Thigh_AS", "R_Thigh_PI", "R_Thigh_AI"]
LEFT_THIGH_MARKERS = ["L_Thigh_PS", "L_Thigh_AS", "L_Thigh_PI", "L_Thigh_AI"]
RIGHT_SHANK_MARKER_INDICES = [0, 1, 2, 3]
LEFT_SHANK_MARKER_INDICES = [0, 1, 2, 3]
RIGHT_THIGH_MARKER_INDICES = [0, 1, 2, 3]
LEFT_THIGH_MARKER_INDICES = [0, 1, 2, 3]

# Foot ACS column c -> (line color, legend). Columns of R_foot are foot axes in lab.
FOOT_ACS_TRIAD_STYLE: tuple[tuple[str, str], ...] = (
    ("#e41a1c", "Foot X — lateral"),
    ("#377eb8", "Foot Y — anterior"),
    ("#4daf4a", "Foot Z — proximal"),
)

# Shank ACS at knee (same column convention as foot; distinct colors)
SHANK_ACS_TRIAD_STYLE: tuple[tuple[str, str], ...] = (
    ("#a65628", "Shank X"),
    ("#7570b3", "Shank Y"),
    ("#66c2a5", "Shank Z"),
)


def _labels_are_generic(labels: list) -> bool:
    if not labels:
        return True
    s0 = str(labels[0])
    return s0.startswith("*") or s0.isdigit()


def _marker_index(labels: list, name: str):
    try:
        return list(labels).index(name)
    except ValueError:
        return None


def _resolve_marker_indices(
    labels: list,
    marker_names: list[str],
    marker_indices_fallback: list[int],
) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    n_m = len(labels)
    generic = _labels_are_generic(labels)
    for k, name in enumerate(marker_names):
        idx = _marker_index(labels, name)
        if idx is not None:
            out.append((name, idx))
            continue
        if generic and k < len(marker_indices_fallback):
            j = int(marker_indices_fallback[k])
            if 0 <= j < n_m:
                out.append((name, j))
    return out


def load_c3d_points(c3d_path: str) -> tuple[list[str], np.ndarray]:
    """Merge POINT LABELS + LABELS2, align with USED / data width (same as static_calibration)."""
    import ezc3d  # type: ignore

    c3d = ezc3d.c3d(c3d_path)
    pt = c3d["parameters"]["POINT"]
    labels = list(pt["LABELS"]["value"])
    if "LABELS2" in pt:
        labels += list(pt["LABELS2"]["value"])
    clean: list[str] = []
    for s in labels:
        if isinstance(s, bytes):
            s = s.decode("utf-8", errors="ignore")
        clean.append(str(s).replace("\x00", "").strip())
    data = c3d["data"]["points"]
    n_data = int(data.shape[1])
    n_used = n_data
    if "USED" in pt:
        try:
            n_used = int(np.asarray(pt["USED"]["value"]).ravel()[0])
        except Exception:
            pass
    n_pts = max(0, min(n_data, n_used))
    if len(clean) > n_pts:
        clean = clean[:n_pts]
    elif len(clean) < n_pts:
        clean = clean + [""] * (n_pts - len(clean))
    xyz = np.transpose(data[:3, :n_pts, :], (2, 1, 0)).astype(float)
    return clean, xyz


def build_figure(
    marker_xyz: np.ndarray,  # (F, M, 3) same units as ACS origin (usually mm)
    foot_pairs: list[tuple[str, int]],
    shank_pairs: list[tuple[str, int]],
    thigh_pairs: list[tuple[str, int]],
    O_foot: np.ndarray,  # (F, 3)
    R_foot: np.ndarray,  # (F, 3, 3)
    O_shank: np.ndarray,  # (F, 3) knee joint center = shank ACS origin
    R_shank: np.ndarray,  # (F, 3, 3)
    time_s: np.ndarray,  # (F,)
    M_ankle_angle: np.ndarray,  # (F, 3)
    M_knee_jcs: np.ndarray,  # (F, 3) FE, Ab/Ad, IE
    *,
    triad_scale: float = 80.0,
    stride: int = 1,
    title: str = "Foot markers + ACS + ankle & knee moments",
) -> "go.Figure":
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    F = int(
        min(
            marker_xyz.shape[0],
            O_foot.shape[0],
            R_foot.shape[0],
            O_shank.shape[0],
            R_shank.shape[0],
            time_s.shape[0],
            M_ankle_angle.shape[0],
            M_knee_jcs.shape[0],
        )
    )
    idxs = np.arange(0, F, stride, dtype=int)
    n_frames = int(idxs.size)

    Mpf = np.asarray(M_ankle_angle[:F, 0], dtype=float)
    M_kfe = np.asarray(M_knee_jcs[:F, 0], dtype=float)
    M_kab = np.asarray(M_knee_jcs[:F, 1], dtype=float)
    M_kie = np.asarray(M_knee_jcs[:F, 2], dtype=float)
    t = np.asarray(time_s[:F], dtype=float)

    ymin_a = float(np.nanmin(Mpf)) - 5.0
    ymax_a = float(np.nanmax(Mpf)) + 5.0
    if not np.isfinite(ymin_a) or not np.isfinite(ymax_a) or ymin_a >= ymax_a:
        ymin_a, ymax_a = -1.0, 1.0

    ymin_k = float(np.nanmin([M_kfe.min(), M_kab.min(), M_kie.min()])) - 5.0
    ymax_k = float(np.nanmax([M_kfe.max(), M_kab.max(), M_kie.max()])) + 5.0
    if not np.isfinite(ymin_k) or not np.isfinite(ymax_k) or ymin_k >= ymax_k:
        ymin_k, ymax_k = -1.0, 1.0

    def markers_group_at(fi: int, pairs: list[tuple[str, int]]):
        xs, ys, zs, texts = [], [], [], []
        n_m = marker_xyz.shape[1]
        for name, mi in pairs:
            if not (0 <= mi < n_m):
                continue
            p = marker_xyz[fi, mi, :]
            if np.all(np.isfinite(p)):
                xs.append(float(p[0]))
                ys.append(float(p[1]))
                zs.append(float(p[2]))
                texts.append(name)
        return xs, ys, zs, texts

    def triad_at(O: np.ndarray, R: np.ndarray, styles: tuple[tuple[str, str], ...]) -> list[tuple]:
        O = np.asarray(O, dtype=float).reshape(3)
        R = np.asarray(R, dtype=float).reshape(3, 3)
        segs = []
        for c in range(3):
            color, leg = styles[c]
            c0 = O
            c1 = O + triad_scale * R[:, c]
            segs.append((c0, c1, color, leg))
        return segs

    k0 = int(idxs[0])
    mx, my, mz, mtxt = markers_group_at(k0, foot_pairs)
    sx, sy, sz, stxt = markers_group_at(k0, shank_pairs)
    tx, ty, tz, ttxt = markers_group_at(k0, thigh_pairs)
    segs_foot = triad_at(O_foot[k0], R_foot[k0], FOOT_ACS_TRIAD_STYLE)

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scatter3d"}, {"type": "xy"}]],
        horizontal_spacing=0.05,
        column_widths=[0.48, 0.52],
        subplot_titles=(
            "",
            "",
        ),
    )

    fig.add_trace(
        go.Scatter3d(
            x=mx,
            y=my,
            z=mz,
            mode="markers",
            marker=dict(size=4.5, color="#984ea3",
                        line=dict(width=1, color="#333")),
            name="Foot markers",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=sx,
            y=sy,
            z=sz,
            mode="markers",
            marker=dict(size=4.5, color="#2ca02c",
                        line=dict(width=1, color="#1a5c1a")),
            name="Shank markers",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter3d(
            x=tx,
            y=ty,
            z=tz,
            mode="markers",
            marker=dict(size=4.5, color="#1f77b4",
                        line=dict(width=1, color="#0d3d5c")),
            name="Thigh markers",
            showlegend=False,
        ),
        row=1, col=1,
    )
    for c0, c1, col, leg in segs_foot:
        fig.add_trace(
            go.Scatter3d(
                x=[c0[0], c1[0]],
                y=[c0[1], c1[1]],
                z=[c0[2], c1[2]],
                mode="lines",
                line=dict(color=col, width=8),
                name=leg,
                showlegend=False,
            ),
            row=1, col=1,
        )
    # Shank ACS triad intentionally hidden for cleaner presentation.
    line_specs = (
        (Mpf, "#ff7f0e", "Ankle PF/DF"),
        (M_kfe, "#1f77b4", "Knee FE"),
        (M_kab, "#d62728", "Knee Ab/Ad"),
    )
    for y, color, name in line_specs:
        fig.add_trace(
            go.Scatter(
                x=t,
                y=y,
                mode="lines",
                line=dict(color=color, width=2),
                name=name,
            ),
            row=1,
            col=2,
        )
    marker_start_idx = len(fig.data)
    for _, color, _ in line_specs:
        fig.add_trace(
            go.Scatter(
                x=[t[k0]],
                y=[np.nan],
                mode="markers",
                marker=dict(size=9.0, color=color, line=dict(
                    width=1.5, color="#ffffff")),
                showlegend=False,
                name="",
            ),
            row=1,
            col=2,
        )
    idx_markers = list(
        range(marker_start_idx, marker_start_idx + len(line_specs)))

    # Traces 0–5: 3d (foot + shank + thigh markers, 3 foot triad)
    n_3d = 6
    frames = []
    for j in range(n_frames):
        fi = int(idxs[j])
        mx, my, mz, mtxt = markers_group_at(fi, foot_pairs)
        sx, sy, sz, stxt = markers_group_at(fi, shank_pairs)
        tx, ty, tz, ttxt = markers_group_at(fi, thigh_pairs)
        sf = triad_at(O_foot[fi], R_foot[fi], FOOT_ACS_TRIAD_STYLE)
        frame_data = [
            go.Scatter3d(
                x=mx,
                y=my,
                z=mz,
                mode="markers",
                marker=dict(size=4.5, color="#984ea3",
                            line=dict(width=1, color="#333")),
            ),
            go.Scatter3d(
                x=sx,
                y=sy,
                z=sz,
                mode="markers",
                marker=dict(size=4.5, color="#2ca02c",
                            line=dict(width=1, color="#1a5c1a")),
            ),
            go.Scatter3d(
                x=tx,
                y=ty,
                z=tz,
                mode="markers",
                marker=dict(size=4.5, color="#1f77b4",
                            line=dict(width=1, color="#0d3d5c")),
            ),
        ]
        for seg in sf:
            frame_data.append(
                go.Scatter3d(
                    x=[seg[0][0], seg[1][0]],
                    y=[seg[0][1], seg[1][1]],
                    z=[seg[0][2], seg[1][2]],
                    mode="lines",
                    line=dict(color=seg[2], width=8),
                )
            )
        # Shank ACS triad intentionally omitted from animated frame updates.
        vals = [Mpf[fi], M_kfe[fi], M_kab[fi]]
        for val in vals:
            frame_data.append(go.Scatter(x=[t[fi]], y=[val], mode="markers"))
        frames.append(
            go.Frame(
                data=frame_data,
                traces=list(range(0, n_3d)) + idx_markers,
                name=str(fi),
            )
        )

    fig.frames = frames

    fig.update_layout(
        title_text=title,
        width=1400,
        height=560,
        margin=dict(l=4, r=120, t=48, b=72),
        scene=dict(
            aspectmode="data",
            xaxis_title="Y",
            yaxis_title="X",
            zaxis_title="Z",
        ),
        xaxis=dict(title="Time (s)"),
        yaxis=dict(title="Moment (Nm/kg)"),
        legend=dict(x=1.01, y=1.0, xanchor="left", yanchor="top"),
        updatemenus=[
            dict(
                type="buttons",
                showactive=True,
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None, dict(frame=dict(
                            duration=30, redraw=True), fromcurrent=True)],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[[None], dict(frame=dict(
                            duration=0), mode="immediate")],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                pad=dict(t=35),
                steps=[
                    dict(
                        method="animate",
                        args=[[fr.name], dict(
                            mode="immediate", frame=dict(duration=0, redraw=True))],
                        label=str(fr.name),
                    )
                    for fr in frames
                ],
                x=0.05,
                xanchor="left",
                len=0.92,
                currentvalue=dict(prefix="Frame index: "),
            )
        ],
    )
    return fig


def main(argv: list[str] | None = None) -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(
        description="Plotly: foot + shank ACS at knee, ankle & knee JCS moments (Nm/kg, synced frames). "
        "Omit paths to use defaults under this script folder.",
    )
    p.add_argument("--c3d", default=None,
                   help="Walk trial C3D path (default: auto-detect)")
    p.add_argument("--bilateral", default=None,
                   help="*_bilateral_chain_results.npz (default: auto-detect)")
    p.add_argument("--id-npz", default=None,
                   help="*_foot_ankle_inverse_dynamics.npz (default: auto-detect)")
    p.add_argument(
        "--leg-npz",
        default=None,
        help="*_leg_inverse_dynamics.npz (default: same folder as --id-npz, trial leg file)",
    )
    p.add_argument("--side", choices=("R", "L"), default="R")
    p.add_argument("--out", default="",
                   help="Output HTML path (default: next to id-npz)")
    p.add_argument("--triad-scale-mm", type=float,
                   default=80.0, help="Triad arrow length (mm)")
    p.add_argument("--stride", type=int, default=1,
                   help="Use every Nth frame in slider (smaller HTML)")
    p.add_argument(
        "--no-open",
        action="store_true",
        help="Do not open the HTML in a browser after writing (default: open automatically).",
    )
    args = p.parse_args(argv)

    c3d_path = args.c3d
    bilateral_path = args.bilateral
    id_npz_path = args.id_npz
    leg_npz_path = args.leg_npz
    if not c3d_path or not bilateral_path or not id_npz_path:
        dc3d, dbi, did, dleg = _default_input_paths(script_dir)
        c3d_path = c3d_path or dc3d
        bilateral_path = bilateral_path or dbi
        id_npz_path = id_npz_path or did
        if not leg_npz_path and id_npz_path:
            leg_npz_path = _leg_npz_from_foot_id_path(id_npz_path) or dleg
        elif not leg_npz_path:
            leg_npz_path = dleg
    elif not leg_npz_path and id_npz_path:
        leg_npz_path = _leg_npz_from_foot_id_path(id_npz_path)

    missing: list[str] = []
    if not c3d_path or not os.path.isfile(c3d_path):
        missing.append("--c3d (walk C3D)")
    if not bilateral_path or not os.path.isfile(bilateral_path):
        missing.append("--bilateral")
    if not id_npz_path or not os.path.isfile(id_npz_path):
        missing.append("--id-npz")
    if not leg_npz_path or not os.path.isfile(leg_npz_path):
        missing.append(
            "--leg-npz (or save *_leg_inverse_dynamics.npz next to foot ID)")
    if missing:
        print(
            "Missing or not found: {}. Pass paths explicitly, e.g.\n"
            "  python plot_foot_moment_plotly.py --c3d PATH.c3d --bilateral PATH_bilateral.npz "
            "--id-npz PATH_foot_ankle.npz --leg-npz PATH_leg.npz".format(
                ", ".join(missing)),
            file=sys.stderr,
        )
        return 2

    try:
        import plotly.io as pio
    except ImportError:
        print("Install plotly: pip install plotly", file=sys.stderr)
        return 1

    print(
        "Using:\n  C3D: {}\n  bilateral: {}\n  id: {}\n  leg: {}".format(
            c3d_path, bilateral_path, id_npz_path, leg_npz_path
        ),
        flush=True,
    )

    side = args.side.upper()
    pre_bi = "r" if side == "R" else "l"
    pre_id = "R" if side == "R" else "L"
    marker_names = RIGHT_FOOT_MARKERS if side == "R" else LEFT_FOOT_MARKERS
    fallback_idx = RIGHT_FOOT_MARKER_INDICES if side == "R" else LEFT_FOOT_MARKER_INDICES
    shank_names = RIGHT_SHANK_MARKERS if side == "R" else LEFT_SHANK_MARKERS
    shank_fb = RIGHT_SHANK_MARKER_INDICES if side == "R" else LEFT_SHANK_MARKER_INDICES
    thigh_names = RIGHT_THIGH_MARKERS if side == "R" else LEFT_THIGH_MARKERS
    thigh_fb = RIGHT_THIGH_MARKER_INDICES if side == "R" else LEFT_THIGH_MARKER_INDICES

    labels, xyz = load_c3d_points(c3d_path)
    foot_pairs = _resolve_marker_indices(labels, marker_names, fallback_idx)
    shank_pairs = _resolve_marker_indices(labels, shank_names, shank_fb)
    thigh_pairs = _resolve_marker_indices(labels, thigh_names, thigh_fb)
    if len(foot_pairs) < 2:
        print("Could not resolve enough foot markers from C3D. Labels sample:",
              labels[:12], file=sys.stderr)
        return 1
    if not shank_pairs:
        print("Warning: no shank markers matched in C3D; 3D shank cluster will be empty.", file=sys.stderr)
    if not thigh_pairs:
        print("Warning: no thigh markers matched in C3D; 3D thigh cluster will be empty.", file=sys.stderr)

    bi = np.load(bilateral_path, allow_pickle=True)
    kin_id = np.load(id_npz_path, allow_pickle=True)
    kin_leg = np.load(leg_npz_path, allow_pickle=True)

    R_foot = np.asarray(bi[f"{pre_bi}_foot_acs_R"], dtype=float)
    O_foot = np.asarray(bi[f"{pre_bi}_foot_acs_O"], dtype=float)
    R_shank = np.asarray(bi[f"{pre_bi}_shank_acs_R"], dtype=float)
    O_shank = np.asarray(bi[f"{pre_bi}_shank_acs_O"], dtype=float)

    M_key = f"{pre_id}_M_joint_ankle_angle_frame_Nm"
    t_key = f"{pre_id}_time"
    M_knee_key = f"{pre_id}_M_knee_jcs_Nm"
    if M_key not in kin_id.files:
        print(f"Missing {M_key} in {id_npz_path}", file=sys.stderr)
        return 1
    if M_knee_key not in kin_leg.files:
        print(f"Missing {M_knee_key} in {leg_npz_path}", file=sys.stderr)
        return 1

    M_ank = np.asarray(kin_id[M_key], dtype=float)
    M_knee = np.asarray(kin_leg[M_knee_key], dtype=float)
    if t_key in kin_id.files:
        time_s = np.asarray(kin_id[t_key], dtype=float).ravel()
    elif f"{pre_id}_time" in kin_leg.files:
        time_s = np.asarray(kin_leg[f"{pre_id}_time"], dtype=float).ravel()
    else:
        time_s = np.arange(M_ank.shape[0], dtype=float) / 100.0

    n = int(
        min(
            xyz.shape[0],
            R_foot.shape[0],
            O_foot.shape[0],
            R_shank.shape[0],
            O_shank.shape[0],
            M_ank.shape[0],
            M_knee.shape[0],
            time_s.shape[0],
        )
    )
    if n < 2:
        print("Aligned length too short.", file=sys.stderr)
        return 1
    if (
        xyz.shape[0] != n
        or R_foot.shape[0] != n
        or O_foot.shape[0] != n
        or M_ank.shape[0] != n
        or M_knee.shape[0] != n
    ):
        print(
            "Warning: trimming to n={} (c3d {}, bilateral {}, ankle {}, knee {}).".format(
                n, xyz.shape[0], R_foot.shape[0], M_ank.shape[0], M_knee.shape[0]
            ),
            file=sys.stderr,
        )
    xyz = xyz[:n]
    R_foot = R_foot[:n]
    O_foot = O_foot[:n]
    R_shank = R_shank[:n]
    O_shank = O_shank[:n]
    M_ank = M_ank[:n]
    M_knee = M_knee[:n]
    time_s = time_s[:n]

    out_path = args.out
    if not out_path:
        base = os.path.splitext(os.path.basename(id_npz_path))[0]
        out_path = os.path.join(
            os.path.dirname(os.path.abspath(id_npz_path)), f"{
                base}_foot_knee_moment_viewer.html"
        )

    fig = build_figure(
        xyz,
        foot_pairs,
        shank_pairs,
        thigh_pairs,
        O_foot,
        R_foot,
        O_shank,
        R_shank,
        time_s,
        M_ank,
        M_knee,
        triad_scale=float(args.triad_scale_mm),
        stride=max(1, int(args.stride)),
        title=f"Inverse Dynamics: Ankle & Knee Moments ({
            'Right' if side == 'R' else 'Left'} Leg)",
    )
    pio.write_html(fig, file=out_path, include_plotlyjs="cdn",
                   auto_open=not args.no_open)
    print("Wrote", os.path.abspath(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
