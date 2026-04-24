# Lower-body motion capture: IK → kinetics pipeline

**End-to-end processing from raw laboratory C3D files to time series of joint angles and intersegmental moments for a pelvis-to-foot chain** — modular Python scripts, intermediate NPZ/CSV artifacts, static calibration, Grood–Suntay knee conventions, and force-plate preprocessing aligned to the kinematic frame rate.

### IK results

Right-leg walking trial: 3D marker animation (segment ACS fit) synchronized with hip / knee / ankle angle time series (Grood–Suntay knee FE & var–val, ISB-style hip and ankle).

![IK results — markers + joint angles vs frame](IK.gif)

*Source files live in the repo as [`reports/figures/IK_results.gif`](reports/IK.gif) (update by copying from e.g. `Downloads/IK results.gif` if you re-record).

### Inverse dynamics (ID)

Same trial class: ground-reaction–based Newton–Euler moments at the ankle (PF/DF) and knee (FE, abduction/adduction in Grood–Suntay JCS), shown with markers and a moving time cursor.

![Inverse dynamics — ankle & knee moments vs time](reports/ID.gif)

*Source: [`reports/figures/ID.gif`](ID.gif) (e.g. sync from `Downloads/ID.gif`).*

---

## Results (quick read)

- **IK (kinematics):** See the **IK results** GIF above; interactive exports include `Walk_R04_angles_right.html` and bilateral chain NPZ from [`svd_kabsch.py`](scripts/static%20calib/svd_kabsch.py).
- **ID (kinetics):** See the **ID** GIF above; QC PDFs via [`plot_inverse_dynamics_qc.py`](scripts/static%20calib/plot_inverse_dynamics_qc.py) and HTML viewers under `scripts/static calib/subject 02 - S_Cal02/` (e.g. ankle/knee moment dashboards).

### Comparison to literature (poster)

- **Inverse kinematics:** Joint physiologic patterns consistent with gait literature; **knee flexion during stance (~10–20°)** aligns with reported ranges [4].
- **Inverse dynamics:** Joint moments fall within **ACLR** reporting ranges: **knee** ~**0.3–0.5 Nm/kg**, **ankle plantarflexion** ~**1.2–1.4 Nm/kg** [4].

**Reference:** [4] Khandha et al. (2025), *Journal of Biomechanics* (poster Fig. 4–5 captions).

---

## Pipeline / methods (brief)

```mermaid
flowchart LR
  A[C3D markers] --> B[Static calibration]
  C[C3D force plates] --> D[Force-plate preprocess]
  B --> E[SVD / Kabsch dynamic fit]
  E --> F[Joint angles]
  E --> G[Kinematic derivatives]
  D --> G
  H[Inertial segments] --> I[Newton–Euler ID]
  G --> I
  F -. same JCS .- I
```

**Stages (bullets):**

| Stage | Role |
|--------|------|
| **Static calibration** | Anatomical coordinate systems (ACS), joint-center templates — [`static_calibration.py`](scripts/static%20calib/static_calibration.py) |
| **Dynamic IK** | Rigid body fit per frame, bilateral segment rotations — [`svd_kabsch.py`](scripts/static%20calib/svd_kabsch.py) |
| **Angles** | Hip / knee (Grood–Suntay) / ankle — [`angles_only.py`](scripts/static%20calib/angles_only.py), [`joint_angles.py`](scripts/static%20calib/joint_angles.py) |
| **Filtering → COM kinematics** | Low-pass kinematics, COM/joint linear acceleration, segment ω and α — [`kinematic_derivatives.py`](scripts/static%20calib/kinematic_derivatives.py) |
| **Force plates** | GRF, **COP**, optional export NPZ aligned to marker trials — [`forceplate_preprocess.py`](scripts/static%20calib/forceplate_preprocess.py) |
| **Inertia** | Scaled segment mass, COM offset, principal inertias — [`inertial_segments.py`](scripts/static%20calib/inertial_segments.py) |
| **ID** | Foot wrench + bottom-up shank/thigh; knee moments in Grood–Suntay JCS — [`inverse_dynamics_newton_euler.py`](scripts/static%20calib/inverse_dynamics_newton_euler.py) |

**Solver:** Rigid-body **Newton–Euler** inverse dynamics with ground reaction **force** at **center of pressure (COP)** on the instrumented foot, propagated proximally with consistent segment ACS and documented sign conventions.

---

## Poster

PNG lives at the **repository root** as **`poster.png`** (default branch `main`), so GitHub renders the image below and it is also available as  
`https://raw.githubusercontent.com/<your-username>/<your-repo>/main/poster.png`.

![Conference poster — IK and ID results (Figs. 4–5)](poster.png)

Replace **`poster.png`** at the repo root when you export a higher-resolution slide; keep the same filename so this README link stays valid.

**Acknowledgments (as on poster):** Dr. Fiorentino; NIH NIAMS **R21AR077371**; S. Kohbandeloo.

**References (poster):** [1] Wu et al. (2002), *J. Biomech.* 35(4); [2] Kabsch (1976), *Acta Crystallogr. A* 32(5); [3] Winter (2009), *Biomechanics and Motor Control of Human Movement*; [4] Khandha et al. (2025), *J. Biomech.*

---

## Full technical report

Primary write-up (compile to PDF):

- **[`reports/lower_body_pipeline_report.tex`](reports/lower_body_pipeline_report.tex)** — *Lower-Body Biomechanics Pipeline for Kinematic and Kinetic Analysis from Raw Marker Data* (methods, testing, equations by module).

```bash
cd reports
pdflatex lower_body_pipeline_report.tex
```

---

## Repository layout 

| Path | Purpose |
|------|---------|
| `poster.png` | Conference poster (PNG on `main`, repo root — README embed + raw URL) |
| `c3d/` | Raw / organized C3D and GRF exports |
| `scripts/static calib/` | Main pipeline scripts, subject folders, NPZ/HTML outputs |
| `reports/` | LaTeX reports |

**Author:** Luke Camarao — University of Vermont, Biomedical Engineering (see report title pages for mentor and date).
