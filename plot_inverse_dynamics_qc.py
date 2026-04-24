# -*- coding: utf-8 -*-
"""
QC figures for inverse dynamics: GRF, **ankle and knee** moments (including knee
abduction in Grood–Suntay JCS), relative joint ω, power, and work.

Loads:
  - ``*_COM_kinematics.npz`` — ``time``, optional ``grf_N``, ``{L|R}_{foot|shank|thigh}_omega_rad_s``
  - ``*_leg_inverse_dynamics.npz`` — ``{L|R}_*`` arrays from ``load_leg_chain_id_from_pipeline_outputs``
  - ``*_bilateral_chain_results.npz`` — ``{l|r}_*_acs_R`` for ω_lab = R @ ω_body

**Knee abduction moment:** component 1 of ``M_knee_jcs_Nm`` (Grood–Suntay conjugate to
abduction/adduction / varus–valgus), same ordering as ``joint_angles.knee_angles_grood_suntay``
(FE, AbAd, IE).

**Relative ω (lab):** ankle: ω_foot−ω_shank; knee: ω_shank−ω_thigh.

**Power:** P = M_joint · ω_rel (lab moment and relative ω).

**Work:** cumulative ∫ P dt and stance-only totals for **ankle and knee** only.

Example::

    python plot_inverse_dynamics_qc.py ^
      --com-kin "subject 02 - S_Cal02/Walk_R04_COM_kinematics.npz" ^
      --leg-id "subject 02 - S_Cal02/Walk_R04_leg_inverse_dynamics.npz" ^
      --bilateral "subject 02 - S_Cal02/Walk_R04_bilateral_chain_results.npz" ^
      --side R ^
      --out "subject 02 - S_Cal02/Walk_R04_inverse_dynamics_qc.pdf"

Run with no arguments to use default paths next to this script when present.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Match kinematic_derivatives checkpoint stance rule (no SciPy import for this script).
CHECKPOINT_STANCE_FZ_THRESHOLD_N = 50.0
DEFAULT_WALK_BASE = "Walk_R04"

# Grood–Suntay knee JCS: columns align with FE, AbAd, IE (see joint_angles.knee_angles_grood_suntay).
KNEE_JCS_FE_AXIS = 0
KNEE_JCS_ABAD_AXIS = 1
KNEE_JCS_IE_AXIS = 2


def _default_paths(script_dir: str, walk_base: str = DEFAULT_WALK_BASE) -> tuple[str | None, str | None, str | None]:
    subject_dir = os.path.join(script_dir, "subject 02 - S_Cal02")
    com_kin = os.path.join(subject_dir, f"{walk_base}_COM_kinematics.npz")
    leg_id = os.path.join(subject_dir, f"{walk_base}_leg_inverse_dynamics.npz")
    bilateral = os.path.join(
        subject_dir, f"{walk_base}_bilateral_chain_results.npz")
    return (
        com_kin if os.path.isfile(com_kin) else None,
        leg_id if os.path.isfile(leg_id) else None,
        bilateral if os.path.isfile(bilateral) else None,
    )


def _bilateral_prefix(side: str) -> str:
    return "l" if side.upper().startswith("L") else "r"


def _side_seg_names(side: str) -> tuple[str, str, str]:
    s = side.upper()[:1]
    return f"{s}_foot", f"{s}_shank", f"{s}_thigh"


def _omega_lab_series(
    bi: np.lib.npyio.NpzFile,
    kin: np.lib.npyio.NpzFile,
    prefix: str,
    seg_inertial: str,
    n: int,
) -> np.ndarray:
    """Angular velocity in lab frame: ω_lab = R_seg @ ω_body (COM kinematics convention)."""
    om_b = np.asarray(kin[f"{seg_inertial}_omega_rad_s"], dtype=float)[:n]
    part = seg_inertial.split("_", 1)[1]
    R = np.asarray(bi[f"{prefix}_{part}_acs_R"], dtype=float)[:n]
    return np.einsum("fij,fj->fi", R, om_b)


def _align_n(*arrays: np.ndarray) -> int:
    lengths = [int(a.shape[0]) for a in arrays if a is not None and getattr(
        a, "shape", (0,))[0] > 0]
    return min(lengths) if lengths else 0


def _cumulative_trapezoid(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float).ravel()
    t = np.asarray(t, dtype=float).ravel()
    n = min(y.size, t.size)
    if n < 2:
        return np.zeros(n, dtype=float)
    y, t = y[:n], t[:n]
    dt = np.diff(t)
    inc = 0.5 * (y[1:] + y[:-1]) * dt
    out = np.zeros(n, dtype=float)
    out[1:] = np.cumsum(inc)
    return out


def _stance_index_windows(stance: np.ndarray) -> list[tuple[int, int]]:
    st = np.asarray(stance, dtype=bool).ravel()
    wins: list[tuple[int, int]] = []
    n = st.size
    i = 0
    while i < n:
        if not st[i]:
            i += 1
            continue
        j = i
        while j < n and st[j]:
            j += 1
        wins.append((i, j - 1))
        i = j
    return wins


def _trapezoid_interval(t: np.ndarray, y: np.ndarray, i0: int, i1: int) -> float:
    if i1 <= i0:
        return 0.0
    tt = np.asarray(t, dtype=float)[i0: i1 + 1]
    yy = np.asarray(y, dtype=float).ravel()[i0: i1 + 1]
    if tt.size < 2:
        return 0.0
    dt = np.diff(tt)
    return float(np.sum(0.5 * (yy[1:] + yy[:-1]) * dt))


def work_trapezoid_over_stance(P: np.ndarray, t: np.ndarray, stance: np.ndarray) -> float:
    """Total mechanical work (J) = sum over stance windows of ∫ P dt (trapezoid)."""
    total = 0.0
    for i0, i1 in _stance_index_windows(stance):
        total += _trapezoid_interval(t, P, i0, i1)
    return total


def relative_joint_omega_ankle_knee_lab(
    w_foot_lab: np.ndarray,
    w_shank_lab: np.ndarray,
    w_thigh_lab: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Ankle and knee relative ω in lab (rad/s), shape (N,3): distal − proximal."""
    wf = np.asarray(w_foot_lab, dtype=float)
    ws = np.asarray(w_shank_lab, dtype=float)
    wt = np.asarray(w_thigh_lab, dtype=float)
    return wf - ws, ws - wt


def knee_moment_jcs_components(M_knee_jcs_Nm: np.ndarray) -> dict[str, np.ndarray]:
    """
    Scalar components from Grood–Suntay ``M_knee_jcs_Nm`` (N,3).

    Returns keys ``fe``, ``abduction`` (AbAd / varus–valgus conjugate), ``ie``.
    """
    M = np.asarray(M_knee_jcs_Nm, dtype=float)
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    return {
        "fe": M[:, KNEE_JCS_FE_AXIS],
        "abduction": M[:, KNEE_JCS_ABAD_AXIS],
        "ie": M[:, KNEE_JCS_IE_AXIS],
    }


def joint_power_lab(M_joint_lab_Nm: np.ndarray, omega_rel_lab: np.ndarray) -> np.ndarray:
    """Scalar joint power (W): P = M · ω_rel, both in lab frame, shape (N,)."""
    M = np.asarray(M_joint_lab_Nm, dtype=float)
    w = np.asarray(omega_rel_lab, dtype=float)
    return np.sum(M * w, axis=1)


def _shade_stance(ax, t: np.ndarray, stance: np.ndarray, *, color: str = "0.85", zorder: float = 0) -> None:
    st = np.asarray(stance, dtype=bool).ravel()
    if st.size == 0 or t.size == 0:
        return
    n = min(st.size, t.size)
    st, t = st[:n], t[:n]
    i = 0
    while i < n:
        if not st[i]:
            i += 1
            continue
        j = i
        while j < n and st[j]:
            j += 1
        ax.axvspan(float(t[i]), float(t[j - 1]),
                   facecolor=color, alpha=0.35, lw=0, zorder=zorder)
        i = j


def _mask_to_stance(y: np.ndarray, stance: np.ndarray) -> np.ndarray:
    """Return y with non-stance samples replaced by NaN (for stance-only plotting)."""
    yy = np.asarray(y, dtype=float).copy()
    st = np.asarray(stance, dtype=bool).ravel()
    n = min(yy.shape[0], st.shape[0])
    if n == 0:
        return yy
    if yy.ndim == 1:
        yy[:n] = np.where(st[:n], yy[:n], np.nan)
    else:
        yy[:n, ...] = np.where(st[:n, None], yy[:n, ...], np.nan)
    return yy


def plot_inverse_dynamics_qc(
    com_kinematics_npz: str,
    leg_inverse_dynamics_npz: str,
    bilateral_npz: str,
    *,
    side: str = "R",
    out_path: str,
    fz_stance_n: float = CHECKPOINT_STANCE_FZ_THRESHOLD_N,
) -> None:
    kin = np.load(com_kinematics_npz, allow_pickle=True)
    leg = np.load(leg_inverse_dynamics_npz, allow_pickle=True)
    bi = np.load(bilateral_npz, allow_pickle=True)

    pre = side.upper()[:1]
    pfx = _bilateral_prefix(side)
    foot_s, shank_s, thigh_s = _side_seg_names(side)

    t_kin = np.asarray(kin["time"], dtype=float)
    tk = f"{pre}_time"
    if tk in leg.files:
        t_id = np.asarray(leg[tk], dtype=float)
    elif "time" in leg.files:
        t_id = np.asarray(leg["time"], dtype=float)
    else:
        t_id = t_kin

    n = _align_n(t_kin, t_id)
    need_leg = [f"{pre}_M_joint_lab_Nm", f"{pre}_M_knee_lab_Nm"]
    for k in need_leg:
        if k not in leg.files:
            raise KeyError(f"Missing {k} in {leg_inverse_dynamics_npz}")
    mkj = f"{pre}_M_knee_jcs_Nm"
    if mkj not in leg.files:
        raise KeyError(
            f"Missing {mkj} in {
                leg_inverse_dynamics_npz} (needed for knee FE / abduction / IE in JCS)"
        )
    for seg in (foot_s, shank_s, thigh_s):
        wk = f"{seg}_omega_rad_s"
        if wk not in kin.files:
            raise KeyError(f"Missing {wk} in {
                           com_kinematics_npz} (needed for joint power)")

    M_ank = np.asarray(leg[f"{pre}_M_joint_lab_Nm"], dtype=float)
    M_kn = np.asarray(leg[f"{pre}_M_knee_lab_Nm"], dtype=float)
    M_kn_jcs = np.asarray(leg[mkj], dtype=float)
    n = min(n, M_ank.shape[0], M_kn.shape[0], M_kn_jcs.shape[0])

    t_kin = t_kin[:n]
    t_plot = t_id[:n] if t_id.shape[0] >= n else t_kin

    w_foot_lab = _omega_lab_series(bi, kin, pfx, foot_s, n)
    w_shank_lab = _omega_lab_series(bi, kin, pfx, shank_s, n)
    w_thigh_lab = _omega_lab_series(bi, kin, pfx, thigh_s, n)

    om_ank, om_kn = relative_joint_omega_ankle_knee_lab(
        w_foot_lab, w_shank_lab, w_thigh_lab)

    P_ankle = joint_power_lab(M_ank[:n], om_ank)
    P_knee = joint_power_lab(M_kn[:n], om_kn)

    knee_cmp = knee_moment_jcs_components(M_kn_jcs[:n])
    M_knee_abduction = knee_cmp["abduction"]

    stance = np.zeros(n, dtype=bool)
    grf = None
    if "grf_N" in kin.files:
        grf = np.asarray(kin["grf_N"], dtype=float)[:n]
        stance = grf[:, 2] > fz_stance_n
    elif "stance_mask_fp" in kin.files:
        stance = np.asarray(kin["stance_mask_fp"], dtype=float)[
            :n].astype(bool)

    # Stance-only traces for QC plotting.
    M_ank_st = _mask_to_stance(M_ank[:n], stance)
    M_kn_jcs_st = _mask_to_stance(M_kn_jcs[:n], stance)
    M_knee_abduction_st = _mask_to_stance(M_knee_abduction, stance)
    mag_ank_st = _mask_to_stance(np.linalg.norm(om_ank, axis=1), stance)
    mag_kn_st = _mask_to_stance(np.linalg.norm(om_kn, axis=1), stance)
    P_ankle_st_plot = _mask_to_stance(P_ankle, stance)
    P_knee_st_plot = _mask_to_stance(P_knee, stance)

    # For work plots, use stance-only power by default.
    W_ankle = _cumulative_trapezoid(np.where(stance, P_ankle, 0.0), t_plot)
    W_knee = _cumulative_trapezoid(np.where(stance, P_knee, 0.0), t_plot)
    W_sum = W_ankle + W_knee

    P_st_ankle = np.where(stance, P_ankle, 0.0)
    P_st_knee = np.where(stance, P_knee, 0.0)
    W_st_ankle = _cumulative_trapezoid(P_st_ankle, t_plot)
    W_st_knee = _cumulative_trapezoid(P_st_knee, t_plot)
    W_st_sum = W_st_ankle + W_st_knee

    w_st_ank = work_trapezoid_over_stance(P_ankle, t_plot, stance)
    w_st_kn = work_trapezoid_over_stance(P_knee, t_plot, stance)
    w_st_tot = w_st_ank + w_st_kn

    os.makedirs(os.path.dirname(os.path.abspath(out_path))
                or ".", exist_ok=True)

    def _apply_stance(axs):
        for ax in axs:
            _shade_stance(ax, t_plot, stance)

    with PdfPages(out_path) as pdf:
        # --- GRF ---
        fig1, ax1 = plt.subplots(4, 1, sharex=True, figsize=(
            10, 7), constrained_layout=True)
        if grf is not None:
            labels = ("Fx", "Fy", "Fz")
            for i, lab in enumerate(labels):
                ax1[i].plot(t_plot, grf[:, i], "k-", lw=0.7)
                ax1[i].set_ylabel(f"{lab} (N)")
                ax1[i].grid(True, alpha=0.3)
            ax1[3].plot(t_plot, np.linalg.norm(
                grf, axis=1), color="C0", lw=0.7)
            ax1[3].set_ylabel("|F| (N)")
            ax1[3].grid(True, alpha=0.3)
            _apply_stance(ax1)
        else:
            ax1[0].text(0.5, 0.5, "No grf_N in COM kinematics NPZ",
                        ha="center", va="center", transform=ax1[0].transAxes)
            for a in ax1[1:]:
                a.set_visible(False)
        fig1.suptitle(f"GRF (stance Fz > {fz_stance_n:g} N shaded)")
        ax1[-1].set_xlabel("Time (s)")
        pdf.savefig(fig1)
        plt.close(fig1)

        # --- Moments: ankle + knee (JCS includes abduction = col {KNEE_JCS_ABAD_AXIS}) ---
        fig2, axes2 = plt.subplots(
            2, 3, sharex=True, figsize=(11, 6), constrained_layout=True)
        ma_key = f"{pre}_M_joint_ankle_angle_frame_Nm"
        if ma_key in leg.files:
            # This is a MOMENT vector resolved in the ankle-angle frame (not angles).
            M_ank_plot = _mask_to_stance(np.asarray(
                leg[ma_key], dtype=float)[:n], stance)
            ankle_titles = ("Ankle moment PF/DF axis",
                            "Ankle moment axis 2", "Ankle moment axis 3")
        else:
            M_ank_plot = M_ank_st
            ankle_titles = ("Ankle moment Mx lab",
                            "Ankle moment My lab", "Ankle moment Mz lab")
        knee_titles = (
            "Knee FE (JCS, flex–ext)",
            "Knee Ab/Ad (JCS, abduction moment)",
            "Knee IE (JCS)",
        )
        M_kn_plot = M_kn_jcs_st

        for j in range(3):
            axes2[0, j].plot(t_plot, M_ank_plot[:, j], lw=0.7, color="C%d" % j)
            axes2[0, j].set_ylabel("Nm/kg")
            axes2[0, j].set_title(ankle_titles[j])
            axes2[0, j].grid(True, alpha=0.3)
            axes2[1, j].plot(t_plot, M_kn_plot[:, j], lw=0.7, color="C%d" % j)
            axes2[1, j].set_ylabel("Nm/kg")
            axes2[1, j].set_title(knee_titles[j])
            axes2[1, j].grid(True, alpha=0.3)
        _apply_stance(axes2.ravel())
        fig2.suptitle(
            f"Ankle & knee moments ({
                side} leg) — ankle row: moment components; "
            f"knee row: Grood–Suntay JCS (abduction = column {
                KNEE_JCS_ABAD_AXIS})"
        )
        axes2[1, 1].set_xlabel("Time (s)")
        pdf.savefig(fig2)
        plt.close(fig2)

        # --- Knee abduction emphasis (same series as middle JCS panel) ---
        fig2b, axb = plt.subplots(1, 1, figsize=(
            10, 3.2), constrained_layout=True)
        axb.plot(t_plot, M_knee_abduction_st, color="C1", lw=0.9)
        axb.set_ylabel("Nm/kg")
        axb.set_xlabel("Time (s)")
        axb.grid(True, alpha=0.3)
        _shade_stance(axb, t_plot, stance)
        fig2b.suptitle(
            f"Knee abduction moment ({side} leg): M_knee_JCS col {
                KNEE_JCS_ABAD_AXIS} "
            "(Grood–Suntay Ab/Ad conjugate)"
        )
        pdf.savefig(fig2b)
        plt.close(fig2b)

        # --- |ω_rel| and P = M·ω_rel (ankle & knee only) ---
        fig3, ax3 = plt.subplots(2, 2, sharex=True, figsize=(
            11, 5), constrained_layout=True)
        rows = (
            (mag_ank_st, P_ankle_st_plot,
             r"$|\omega_{\mathrm{rel}}| = \|\omega_{\mathrm{foot}}-\omega_{\mathrm{shank}}\|$", r"$P = M\cdot\omega_{\mathrm{rel}}$"),
            (mag_kn_st, P_knee_st_plot,
             r"$|\omega_{\mathrm{rel}}| = \|\omega_{\mathrm{shank}}-\omega_{\mathrm{thigh}}\|$", r"$P = M\cdot\omega_{\mathrm{rel}}$"),
        )
        for r, (mag, P_row, t_om, t_p) in enumerate(rows):
            ax3[r, 0].plot(t_plot, mag, color="0.35", lw=0.7)
            ax3[r, 0].set_ylabel(r"rad/s")
            ax3[r, 0].set_title(t_om)
            ax3[r, 0].grid(True, alpha=0.3)
            ax3[r, 1].plot(t_plot, P_row, color="C%d" % r, lw=0.7)
            ax3[r, 1].set_ylabel("W")
            ax3[r, 1].set_title(t_p)
            ax3[r, 1].grid(True, alpha=0.3)
        ax3[1, 0].set_xlabel("Time (s)")
        ax3[1, 1].set_xlabel("Time (s)")
        _apply_stance(ax3.ravel())
        fig3.suptitle(
            f"Ankle & knee: relative ω (lab) and power — "
            f"P = M_joint · ω_rel,  ω_lab = R·ω_body from COM kinematics"
        )
        pdf.savefig(fig3)
        plt.close(fig3)

        # --- Cumulative work (ankle + knee) ---
        stance_note = (
            f"Stance ∫P dt (trapezoid, Fz>{fz_stance_n:g} N): "
            f"ankle {w_st_ank:.2f} J, knee {
                w_st_kn:.2f} J, sum {w_st_tot:.2f} J"
        )
        fig4, ax4 = plt.subplots(3, 1, sharex=True, figsize=(
            10, 6.5), constrained_layout=True)
        ax4[0].plot(t_plot, W_ankle, label="ankle", lw=0.8)
        ax4[0].plot(t_plot, W_knee, label="knee", lw=0.8)
        ax4[0].set_ylabel("Work (J)")
        ax4[0].legend(loc="upper right", fontsize=8)
        ax4[0].grid(True, alpha=0.3)
        ax4[0].set_title("Cumulative ∫ P dt (stance-only, P=0 in swing)")
        ax4[1].plot(t_plot, W_st_ankle, label="ankle", lw=0.8)
        ax4[1].plot(t_plot, W_st_knee, label="knee", lw=0.8)
        ax4[1].set_ylabel("Work (J)")
        ax4[1].legend(loc="upper right", fontsize=8)
        ax4[1].grid(True, alpha=0.3)
        ax4[1].set_title(
            "Cumulative ∫ P dt with P set to 0 in swing (stance-only accumulation)")
        ax4[2].plot(t_plot, W_sum, color="k", lw=0.9, alpha=0.7,
                    label="sum ankle+knee (all-time)")
        ax4[2].plot(t_plot, W_st_sum, color="C3", lw=0.9,
                    label="sum ankle+knee (stance-only curve)")
        ax4[2].set_ylabel("Work (J)")
        ax4[2].set_xlabel("Time (s)")
        ax4[2].grid(True, alpha=0.3)
        ax4[2].legend(loc="upper right", fontsize=8)
        ax4[2].set_title("Sums — stance totals in suptitle")
        _apply_stance(ax4)
        fig4.suptitle(
            f"Mechanical work (ankle & knee, M·ω_rel in lab) — {stance_note}",
            fontsize=9,
        )
        pdf.savefig(fig4)
        plt.close(fig4)

    print(f"Saved QC PDF: {os.path.abspath(out_path)}", flush=True)
    print(
        "Stance work ∫P dt (J): ankle {:.3f}, knee {:.3f}, sum {:.3f}".format(
            w_st_ank, w_st_kn, w_st_tot
        ),
        flush=True,
    )
    print(
        "Knee abduction moment (JCS col {}): min {:.3f}, max {:.3f} Nm/kg".format(
            KNEE_JCS_ABAD_AXIS,
            float(np.nanmin(M_knee_abduction)),
            float(np.nanmax(M_knee_abduction)),
        ),
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    d_com, d_leg, d_bi = _default_paths(script_dir)

    p = argparse.ArgumentParser(
        description="GRF / ankle & knee moments / power / work QC for leg inverse dynamics.")
    p.add_argument("--com-kin", type=str, default=d_com,
                   help="*_COM_kinematics.npz")
    p.add_argument("--leg-id", type=str, default=d_leg,
                   help="*_leg_inverse_dynamics.npz")
    p.add_argument("--bilateral", type=str, default=d_bi,
                   help="*_bilateral_chain_results.npz")
    p.add_argument("--side", type=str, default="R", choices=("L",
                   "R", "l", "r"), help="Instrumented leg to plot")
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PDF path. Default: next to leg-id NPZ, <trial>_inverse_dynamics_qc.pdf",
    )
    p.add_argument("--fz-stance", type=float, default=CHECKPOINT_STANCE_FZ_THRESHOLD_N,
                   help="Stance shading Fz threshold (N)")
    args = p.parse_args(argv)

    if not args.com_kin or not os.path.isfile(args.com_kin):
        p.error(f"COM kinematics NPZ not found: {args.com_kin}")
    if not args.leg_id or not os.path.isfile(args.leg_id):
        p.error(f"Leg inverse dynamics NPZ not found: {args.leg_id}")
    if not args.bilateral or not os.path.isfile(args.bilateral):
        p.error(f"Bilateral NPZ not found: {args.bilateral}")

    out = args.out
    if out is None:
        base = os.path.basename(args.leg_id).replace(
            "_leg_inverse_dynamics.npz", "")
        out = os.path.join(os.path.dirname(os.path.abspath(args.leg_id)), f"{
                           base}_inverse_dynamics_qc.pdf")

    plot_inverse_dynamics_qc(
        args.com_kin,
        args.leg_id,
        args.bilateral,
        side=args.side,
        out_path=out,
        fz_stance_n=args.fz_stance,
    )


if __name__ == "__main__":
    main()
