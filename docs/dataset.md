# Dataset

## Source

[Human Activity Recognition Using Smartphones, UCI ID 240](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).

Thirty volunteers aged 19 to 48 performed six activities with a waist-mounted smartphone. The data contains numerical accelerometer and gyroscope measurements, not image inputs. Videos were used by the original researchers to annotate activities.

## Expected files

Inside `data/raw/UCI HAR Dataset/`:

| File | Content |
| --- | --- |
| `features.txt` | Names of the 561 calculated features |
| `features_info.txt` | Feature descriptions |
| `activity_labels.txt` | Activity identifiers and names |
| `train/X_train.txt` | 7,352 rows of 561 features |
| `train/y_train.txt` | One activity identifier per training row |
| `train/subject_train.txt` | One participant identifier per training row |
| `test/X_test.txt` | 2,947 rows of 561 features |
| `test/y_test.txt` | One activity identifier per test row |
| `test/subject_test.txt` | One participant identifier per test row |
| `train/Inertial Signals/` and `test/Inertial Signals/` | Windowed sensor signals, 128 readings per row and channel |

The signal channels are total acceleration, body acceleration, and body angular velocity, each with x, y, and z axes. The planned models use the 561 calculated features.

## Activity labels

| ID | Activity |
| --- | --- |
| 1 | Walking |
| 2 | Walking upstairs |
| 3 | Walking downstairs |
| 4 | Sitting |
| 5 | Standing |
| 6 | Lying down |

## Local inspection

The supplied dataset was inspected and extracted locally without changing its data. Inspection confirmed 7,352 training rows and 2,947 test rows, all with 561 columns; six activity identifiers in each split; 21 training participants and 9 test participants; and no participant overlap. This was an inspection, not a committed automated test suite.

Feature names are not all unique. A future loader must retain the column order and use unique names, for example by including each column index. Do not use participant IDs as predictors.

## Limits and attribution

Measurements are from a controlled setting, one smartphone model, and waist placement. Generalization to other settings is untested.

The current UCI page specifies CC BY 4.0. The legacy `.names` description contains an older, different usage notice. A local backup is retained outside the repository; the redundant copy was removed from the dataset directory. Cite the current source accurately.

J. Reyes-Ortiz, D. Anguita, A. Ghio, L. Oneto, and X. Parra, *Human Activity Recognition Using Smartphones*, UCI Machine Learning Repository, 2013. DOI: [10.24432/C54S4K](https://doi.org/10.24432/C54S4K).

D. Anguita et al., “A Public Domain Dataset for Human Activity Recognition Using Smartphones,” ESANN, 2013, pp. 437-442.
