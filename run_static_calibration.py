from __future__ import annotations

import os

from static_calibration_plot import plot_static_calibration
from static_calibration_core import compute_static_data
from qc_utils import qc_frame_pair, summarize_qc
import numpy as np


# Change this to target a different subject folder name
SUBJECT = "subject 02"

# Base folder containing all subjects' C3D files
BASE_DIR = r"C:/Users/lmcam/Documents/Grad project/c3d"

# Static trial filename for the subject
TRIAL_FILE = "S_Cal02.c3d"

# Frame to visualize
FRAME_IDX = 0

# Axis triad length in mm
AXIS_LEN_MM = 80.0


def main() -> None:
    c3d_path = os.path.join(BASE_DIR, SUBJECT, TRIAL_FILE)
    plot_static_calibration(c3d_path, frame_idx=FRAME_IDX,
                            out_html=None, axis_len_mm=AXIS_LEN_MM)
    '''
    # ---- Quick QC for pelvis, femur (L), tibia (L), foot/ankle (L) ----
    
    try:
        results = compute_static_data(c3d_path, FRAME_IDX)
        labels = results["labels"]; xyz = results["xyz"]; f = results["frame_idx"]

        def pick(name: str):
            if name in labels:
                return xyz[f, labels.index(name), :].astype(float)
            lower = {lab.lower(): i for i, lab in enumerate(labels)}
            if name.lower() in lower:
                return xyz[f, lower[name.lower()], :].astype(float)
            return None

        # Pelvis ACS (A)
        RASIS = pick("R_ASIS"); LASIS = pick("L_ASIS"); RPSIS = pick("R_PSIS"); LPSIS = pick("L_PSIS")
        if all(v is not None for v in (RASIS, LASIS, RPSIS, LPSIS)):
            O_A_pelvis = 0.5 * (RASIS + LASIS)
            x = RASIS - LASIS; x = x / np.linalg.norm(x)
            y_raw = O_A_pelvis - 0.5 * (RPSIS + LPSIS)
            y = y_raw - x * float(np.dot(y_raw, x)); y = y / np.linalg.norm(y)
            z = np.cross(x, y); z = z / np.linalg.norm(z)
            R_A_pelvis = np.column_stack([x, y, z])
            # Pelvis TCS (T)
            base = os.path.splitext(os.path.basename(c3d_path))[0]
            tplt = f"{base}_pelvis_tcs_template.npz"
            if os.path.exists(tplt):
                data = np.load(tplt)
                R_T = data["R_T_world"]; O_T = data["origin_T"]
                qc = qc_frame_pair("Pelvis T vs A", R_T, O_T, R_A_pelvis, O_A_pelvis, angle_thresh_deg=15.0)
                print(summarize_qc(qc))
            else:
                print(f"QC skipped: template not found: {tplt}")
        else:
            print("QC skipped: missing pelvis markers in C3D")

        # Femur (Left) ACS (A)
        L_knee_lat = pick("L_Knee_Lat"); L_knee_med = pick("L_Knee_Med")
        if all(v is not None for v in (L_knee_lat, L_knee_med, RASIS, LASIS, RPSIS, LPSIS)):
            # Harrington LHJC in pelvis ACS computed above
            PW = float(np.linalg.norm(LASIS - RASIS))
            mid_ASIS = 0.5 * (LASIS + RASIS)
            mid_PSIS = 0.5 * (RPSIS + LPSIS)
            PD = float(np.linalg.norm(mid_ASIS - mid_PSIS))
            off_lat = 0.24*PW + 0.0099*PD - 3.91; off_ant = 0.30*PW + 10.9; off_sup = 0.33*PW + 7.3
            LHJC = O_A_pelvis + R_A_pelvis @ np.array([-off_lat, -off_ant, -off_sup], float)
            L_KJC = 0.5 * (L_knee_med + L_knee_lat)
            Y_up = LHJC - L_KJC; Y_up = Y_up / np.linalg.norm(Y_up)
            Z_fwd = y - Y_up * float(np.dot(y, Y_up)); Z_fwd = Z_fwd / np.linalg.norm(Z_fwd)
            X_lat = np.cross(Y_up, Z_fwd); X_lat = X_lat / np.linalg.norm(X_lat)
            R_A_femur = np.column_stack([X_lat, Y_up, Z_fwd]); O_A_femur = LHJC
            # Femur TCS (T)
            tplt = f"{base}_femur_tcs_template.npz"
            if os.path.exists(tplt):
                data = np.load(tplt); R_T = data["R_T_world"]; O_T = data["origin_T"]
                qc = qc_frame_pair("Femur L T vs A", R_T, O_T, R_A_femur, O_A_femur, angle_thresh_deg=15.0)
                print(summarize_qc(qc))
            else:
                print(f"QC skipped (femur L): template not found: {tplt}")

        # Tibia (Left) ACS (A)
        L_ank_lat = pick("L_Ank_Lat"); L_ank_med = pick("L_Ank_Med")
        if all(v is not None for v in (L_knee_lat, L_knee_med, L_ank_lat, L_ank_med)):
            L_KJC = 0.5 * (L_knee_med + L_knee_lat)
            L_AJC = 0.5 * (L_ank_med + L_ank_lat)
            Y_up = L_KJC - L_AJC; Y_up = Y_up / np.linalg.norm(Y_up)
            Z_fwd = y - Y_up * float(np.dot(y, Y_up)); Z_fwd = Z_fwd / np.linalg.norm(Z_fwd)
            X_lat = np.cross(Y_up, Z_fwd); X_lat = X_lat / np.linalg.norm(X_lat)
            R_A_tibia = np.column_stack([X_lat, Y_up, Z_fwd]); O_A_tibia = L_AJC
            # Tibia TCS (T)
            tplt = f"{base}_tibia_tcs_template.npz"
            if os.path.exists(tplt):
                data = np.load(tplt); R_T = data["R_T_world"]; O_T = data["origin_T"]
                qc = qc_frame_pair("Tibia L T vs A", R_T, O_T, R_A_tibia, O_A_tibia, angle_thresh_deg=15.0)
                print(summarize_qc(qc))
            else:
                print(f"QC skipped (tibia L): template not found: {tplt}")

        # Foot/Ankle (Left) ACS (A)
        L_calc = pick("L_Calc"); L_toe_med = pick("L_Toe_Med"); L_toe_lat = pick("L_Toe_Lat"); L_toe_tip = pick("L_Toe_Tip")
        if (L_calc is not None) and (L_toe_med is not None) and (L_toe_lat is not None):
            # Report which foot markers are used for ACS
            used_acs = ["L_Calc", "L_Toe_Med", "L_Toe_Lat"]
            if L_toe_tip is not None:
                used_acs.append("L_Toe_Tip")
            if (L_ank_med is not None) and (L_ank_lat is not None):
                used_acs.extend(["L_Ank_Med", "L_Ank_Lat"])
            print(f"Foot ACS markers (L) used: {', '.join(used_acs)}")
            v1 = L_toe_med - L_calc; v5 = L_toe_lat - L_calc
            Z_sup = np.cross(v5, v1); Z_sup = Z_sup / np.linalg.norm(Z_sup)
            if np.dot(Z_sup, np.array([0.0,0.0,1.0])) < 0: Z_sup = -Z_sup
            yraw = (L_toe_tip - L_calc) if (L_toe_tip is not None) else (0.5*(L_toe_med + L_toe_lat) - L_calc)
            Y_ant = yraw - Z_sup * float(np.dot(yraw, Z_sup)); Y_ant = Y_ant / np.linalg.norm(Y_ant)
            if np.dot(Y_ant, y) < 0: Y_ant = -Y_ant
            X_med = np.cross(Y_ant, Z_sup); X_med = X_med / np.linalg.norm(X_med)
            R_A_foot = np.column_stack([X_med, Y_ant, Z_sup])
            # Origin at ankle if available
            O_A_foot = 0.5*(L_ank_med + L_ank_lat) if (L_ank_med is not None and L_ank_lat is not None) else O_A_pelvis
            # Foot TCS (T)
            tplt = f"{base}_foot_tcs_template.npz"
            if os.path.exists(tplt):
                data = np.load(tplt); R_T = data["R_T_world"]; O_T = data["origin_T"]
                # Report which markers were used to build the saved foot TCS template
                if "marker_labels" in data.files:
                    try:
                        labels_arr = data["marker_labels"].tolist()
                        labels_list = [str(x) for x in (labels_arr if isinstance(labels_arr, (list, tuple)) else [labels_arr])]
                        print(f"Foot TCS template markers: {', '.join(labels_list)}")
                    except Exception:
                        pass
                qc = qc_frame_pair("Foot/Ankle L T vs A", R_T, O_T, R_A_foot, O_A_foot, angle_thresh_deg=15.0)
                print(summarize_qc(qc))
            else:
                print(f"QC skipped (foot L): template not found: {tplt}")
    except Exception as e:
        print(f"QC error: {e}")

'''


if __name__ == "__main__":
    main()
