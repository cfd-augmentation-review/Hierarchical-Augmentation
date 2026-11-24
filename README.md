This repository provides the anonymized core implementation code and dataset access for the manuscript:

To support reproducibility and verification, this repository provides:

- The core implementation code of the proposed EDC/TCC-guided hierarchical adaptive augmentation framework;
The Capacitor Foil Dataset (CFD) is available for download in the Assets section of [GitHub Releases](https://github.com/cfd-augmentation-review/Hierarchical-Augmentation/releases).
## Dataset Statistics and Splits

The Capacitor Foil Dataset (CFD) contains 2,608 images covering ten industrial defect categories. The dataset is divided into a training set of 1,820 images and a validation set of 788 images, following an approximately 7:3 ratio. The validation set serves as the evaluation set, and no separate test split is defined.

The class-wise sample distribution is presented below.

| Defect Category | Training Set | Validation Set | Total |
| :--- | ---: | ---: | ---: |
| Dirt | 269 | 116 | 385 |
| Edge Crack | 70 | 31 | 101 |
| Foil Head | 66 | 29 | 95 |
| Fold | 31 | 14 | 45 |
| Hard | 417 | 179 | 596 |
| Hole | 403 | 174 | 577 |
| Insect | 23 | 10 | 33 |
| Material | 31 | 15 | 46 |
| Paste | 341 | 147 | 488 |
| Tear | 169 | 73 | 242 |
| **Total** | **1820** | **788** | **2608** |

The dataset exhibits significant class imbalance and substantial variations in defect scales, reflecting practical challenges in industrial capacitor foil inspection.
