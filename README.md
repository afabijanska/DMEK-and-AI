# Automated Endothelial Cell Density Estimation After DMEK
### Deep Learning Pipeline for In Vivo Confocal Microscopy Images

This repository provides a full deep‑learning workflow for automatic detection of endothelial cell centers and estimation of endothelial cell density (ECD) from in vivo confocal microscopy images of patients after **Descemet Membrane Endothelial Keratoplasty (DMEK)**.

The implementation corresponds to the approach des:

**Karaca E.E., Fabijańska A., Oztoprak K., et al.
*Deep Learning for Assessing Corneal Endothelial Cell Density in Patients After Descemet Membrane Endothelial Keratoplasty: Towards Improved Evaluation.* (2026)**

---

# 📂 Dataset Directory Structure (Generalized)

Below is the directory tree based on the real dataset layout. **⚠️ This exact structure is REQUIRED by the code.**

All preprocessing, training, prediction, and density‑analysis scripts rely on:
- specific folder names,  
- specific subfolder names (`bw/`, `org/`),  
- specific month‑based and eye-side‑based folder naming conventions,  
- separation of patient/control data,  
- presence of prediction output folders used as inputs for later steps.

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
│   └── ...
│
├── patients_bw/
├── patients_preds/
├── control_bw/
├── control_preds/
│
├── dmek/
│   ├── patient-1/<eye> dmek <month>.month/ #eg. patient-3 left dmek 15.month
│   ├── patient-2/
│   └── ...
│
├── dmek3/
│   ├── patient-1/<eye> <month>.month/
│   ├── patient-2/
│   └── ...
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
   Images Prediction
           ↓
ROI Extraction + Cell Counting
           ↓
  ECD Density Calculation
           ↓
CSV Reports + Visualizations

```

---

# 📁 Repository Contents
- AttUNet.py — Attention U-Net definitions
- configuration.txt — dataset paths & parameters
- helpers.py — HDF5 utilities
- get_train_data.py — builds HDF5 datasets from expert annotated images (cell centers, annotations in pure blue) 
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
# 🚀 Usage
```
python get_train_data.py
python train_full_images.py
python predict_patient_full_image.py
python density_from_predicted_patients.py
```

---
# 📄 Citation
Karaca E.E., Fabijanska A., Oztoprak K., et al. (2026)

```
@article{karaca2026dmekecd,
 title = {Deep Learning for Assessing Corneal Endothelial Cell Density in Patients After Descemet Membrane Endothelial Keratoplasty: Towards Improved Evaluation},
 author = {Karaca, Emine Esra and Fabijańska, Anna and Oztoprak, Kasım and Işık, Feyza Dicle and Ustael, Ayça Bulut and Kemer, Özlem Evren and Hassanpour, Reza},
 year = {2026},
 journal = {To Appear}
}
```
