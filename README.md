# 🪄Applying CNN to dMRI images: Automatic Identification and Classification of the Aging Brain
<a href="https://github.com/minnie105">
  <img src="https://avatars.githubusercontent.com/u/263416734?v=4&s=100" width="100px;" alt="my avatar"/>
  <br /><sub><b>Minnie Wang</b></sub>
</a>

## Short Bio: Just a struggling sophomore student from NCU-TW.

I am currently learning the basics of deep learning, neuroimaging, and related fields (just getting started!).

* **Programming Language**: Python 🐍
* **Field**: Everything learned from EE professors (maybe a little or none of it)
* **Environment**: Linux / VS Code
---
## dMRI Data Processing Pipeline

The current stage of this project focuses on converting diffusion MRI data into fixed-size white-matter image tensors for subsequent CNN training.

The `data_processing.py` script is designed to perform the following steps:

1. Load TOPUP/EDDY-corrected DWI data.
2. Read the corresponding b-values and EDDY-rotated b-vectors.
3. Construct a DIPY gradient table.
4. Generate a brain mask from b0 images.
5. Fit a diffusion tensor model using weighted least squares.
6. Calculate FA, MD, AD, and RD maps.
7. Register the diffusion scalar maps to a common FA template.
8. Apply a common white-matter mask.
9. Export a four-channel CNN input array with the following channel order:

```text
FA, MD, AD, RD
```

The generated CNN input is stored as a compressed `.npz` file with the expected shape:

```text
(4, X, Y, Z)
```

### Current Status

* Raw DWI data from the OpenNeuro ds000221 dataset have been downloaded.
* The dMRI preprocessing pipeline has been implemented.
* Python syntax and program structure have been checked.
* Full end-to-end validation is still in progress because TOPUP and EDDY preprocessing must be completed before running the complete pipeline.
* CNN model development and training will be performed in the next stage.

Large neuroimaging datasets and generated NIfTI files are excluded from this repository.
