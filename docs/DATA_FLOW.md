# CCS Data Flow

## Complete Quality-Gate Cohort

The complete quality task contains 1,785 images: 1,164 gradable images and 621 `NA` images. The patient-level split is:

| Split | Gradable | NA | Total |
|---|---:|---:|---:|
| Training | 699 | 372 | 1,071 |
| Development validation | 233 | 124 | 357 |
| Frozen internal test | 232 | 125 | 357 |
| Total | 1,164 | 621 | 1,785 |

All images contribute to the binary quality task. Images predicted as NA stop at the quality gate and do not receive an opacity grade, opening-area estimate, or final CCS diagnosis.

## Current Best Grading and Area Cohort

The expanded gradable development manifest contained 1,471 records. This was reduced to the final unique gradable cohort as follows:

| Step | Records |
|---|---:|
| Expanded gradable development manifest | 1,471 |
| Excluded anomalous grouped records | -134 |
| Remaining records before registration deduplication | 1,337 |
| Removed duplicate registrations | -383 |
| Unique gradable images | 954 |

The 954 images were split into 777 training and 177 internal-validation images with zero patient-group and eye-group overlap. Twenty-three boundary images underwent blinded review, and 13 labels changed. The reviewed patient groups were retained in training.

## Why the Counts Must Remain Separate

The 1,471 count is an intermediate gradable-record manifest, not the total obtained by adding NA images to 954. The difference between 1,471 and 954 comes from the anomalous grouped records and duplicate registrations. The NA population belongs to the separate quality-gate cohort and has no reliable opacity-grade or opening-area target.
