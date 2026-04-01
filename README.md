# Automated Endothelial Cell Density Estimation After DMEK
### Deep Learning Pipeline for In Vivo Confocal Microscopy Images

This repository provides a full deep‑learning workflow for automatic detection of endothelial cell centers and estimation of endothelial cell density (ECD) from in vivo confocal microscopy images of patients after **Descemet Membrane Endothelial Keratoplasty (DMEK)**.

The implementation corresponds to:

**Karaca E.E., Fabijańska A., Oztoprak K., et al.
*Deep Learning for Assessing Corneal Endothelial Cell Density in Patients After Descemet Membrane Endothelial Keratoplasty.* (2024)**

---

# 📂 Dataset Directory Structure (Generalized)
Below is the full directory tree based on the real dataset layout.

```
D:/
│
├── controls/
│   ├── control-1/
│   │   ├── bw/
│   │   └── org/
│   ├── control-2/
│   ├── ...
│   └── control-20/
│
├── patients/
│   ├── patient-1/
│   │   ├── patient-1 1.month/bw, org
│   │   ├── patient-1 6.month/bw, org
│   │   └── patient-1 9.month/bw, org
│   ├── patient-2/
│   ├── ...
│   └── patient-26/
│
├── patients_bw/
├── patients_preds/
├── control_bw/
├── control_preds/
│
├── dmek/
│   ├── patient-1/<eye> <month>.month/
│   ├── patient-2/
│   ├── ...
│   └── patient-26/
│
├── dmek3/
│   ├── patient-1/<month>/mask+overlay
│   ├── patient-2/
│   ├── ...
│   └── patient-26/
│
├── dmek_fin/
├── dmek_fin_images/
│
├── cell_densities_patients.csv
└── average_densities.xlsx
```

---

# 🧠 Pipeline Overview
```
Raw Images
 ↓
Training Dataset (HDF5)
 ↓
Attention U‑Net Model Training
 ↓
Full‑Image Prediction
 ↓
ROI Extraction + Cell Counting
 ↓
ECD Density Calculation
 ↓
CSV + XLSX Reports + Visualizations
```

---

# 📁 Repository Contents
- AttUNet.py — Attention U-Net definitions
- configuration.txt — dataset paths & parameters
- helpers.py — HDF5 utilities
- get_train_data.py — builds HDF5 datasets
- train_full_images.py — trains the model
- predict_patient_full_image.py — inference
- density_from_predicted_patients.py — ROI + density calculations

Generated outputs:
- train_images_all.hdf5
- train_gts_all.hdf5
- bestWeights_all.h5
- model_paper.json
- cell_densities_patients.csv
- average_densities.xlsx

---
# 🔧 Installation
```
git clone <repo>
pip install -r requirements.txt
```
Update configuration.txt accordingly.

---
# 🚀 Usage
```
python get_train_data.py
python train_full_images.py
python predict_patient_full_image.py
python density_from_predicted_patients.py
```

---
# 📄 Citation
Karaca E.E., Fabijańska A., Oztoprak K., et al. (2024)

```
@article{karaca2024dmekecd,
 title={Deep Learning for Assessing Corneal Endothelial Cell Density in Patients After Descemet Membrane Endothelial Keratoplasty},
 author={Karaca, Emine Esra and Fabijańska, Anna and Oztoprak, Kasım and Işık, Feyza Dicle and Ustael, Ayça Bulut and Kemer, Özlem Evren and Hassanpour, Reza},
 year={2024},
 journal={To Appear}
}
```

---
# 📜 License
TBD
