# MIMIC-IV-ECG accounting check (post hoc; InfoNCE vs soft mix 1 / soft-tau 0.2)

## Macro test AUROC by correction variant

| family   | concepts         | variant                 |   infonce_mix0.00_st-1.00 |   soft_mix1.00_st0.20 |   gap |
|:---------|:-----------------|:------------------------|--------------------------:|----------------------:|------:|
| clinical | all 8            | direct z.mu subtraction |                     0.718 |                 0.745 | 0.027 |
| clinical | all 8            | pipeline                |                     0.720 |                 0.761 | 0.041 |
| clinical | all 8            | uncorrected             |                     0.704 |                 0.746 | 0.042 |
| clinical | excl. normal ECG | direct z.mu subtraction |                     0.714 |                 0.729 | 0.015 |
| clinical | excl. normal ECG | pipeline                |                     0.732 |                 0.748 | 0.016 |
| clinical | excl. normal ECG | uncorrected             |                     0.686 |                 0.732 | 0.045 |
| reports  | all 8            | direct z.mu subtraction |                     0.824 |                 0.863 | 0.038 |
| reports  | all 8            | pipeline                |                     0.855 |                 0.869 | 0.014 |
| reports  | all 8            | uncorrected             |                     0.774 |                 0.862 | 0.088 |
| reports  | excl. normal ECG | direct z.mu subtraction |                     0.812 |                 0.858 | 0.046 |
| reports  | excl. normal ECG | pipeline                |                     0.847 |                 0.865 | 0.018 |
| reports  | excl. normal ECG | uncorrected             |                     0.754 |                 0.857 | 0.103 |

## Per-concept test AUROC (seed-0 heads)

|                                                         |   ('infonce_mix0.00_st-1.00', 'direct z.mu subtraction') |   ('infonce_mix0.00_st-1.00', 'pipeline') |   ('infonce_mix0.00_st-1.00', 'uncorrected') |   ('soft_mix1.00_st0.20', 'direct z.mu subtraction') |   ('soft_mix1.00_st0.20', 'pipeline') |   ('soft_mix1.00_st0.20', 'uncorrected') |
|:--------------------------------------------------------|---------------------------------------------------------:|------------------------------------------:|---------------------------------------------:|-----------------------------------------------------:|--------------------------------------:|-----------------------------------------:|
| ('clinical', 'atrial_fibrillation')                     |                                                    0.740 |                                     0.753 |                                        0.706 |                                                0.750 |                                 0.768 |                                    0.749 |
| ('clinical', 'intraventricular_conduction_abnormality') |                                                    0.853 |                                     0.857 |                                        0.847 |                                                0.822 |                                 0.830 |                                    0.818 |
| ('clinical', 'left_ventricular_hypertrophy')            |                                                    0.574 |                                     0.626 |                                        0.518 |                                                0.737 |                                 0.747 |                                    0.735 |
| ('clinical', 'normal_ecg')                              |                                                    0.744 |                                     0.632 |                                        0.827 |                                                0.854 |                                 0.850 |                                    0.848 |
| ('clinical', 'reported_infarct_pattern')                |                                                    0.594 |                                     0.627 |                                        0.534 |                                                0.637 |                                 0.646 |                                    0.635 |
| ('clinical', 'sinus_bradycardia')                       |                                                    0.840 |                                     0.837 |                                        0.844 |                                                0.809 |                                 0.834 |                                    0.819 |
| ('clinical', 'sinus_tachycardia')                       |                                                    0.836 |                                     0.837 |                                        0.826 |                                                0.732 |                                 0.765 |                                    0.743 |
| ('clinical', 'st_t_changes')                            |                                                    0.563 |                                     0.590 |                                        0.528 |                                                0.619 |                                 0.647 |                                    0.622 |
| ('reports', 'atrial_fibrillation')                      |                                                    0.871 |                                     0.885 |                                        0.811 |                                                0.872 |                                 0.873 |                                    0.867 |
| ('reports', 'intraventricular_conduction_abnormality')  |                                                    0.930 |                                     0.934 |                                        0.915 |                                                0.933 |                                 0.934 |                                    0.930 |
| ('reports', 'left_ventricular_hypertrophy')             |                                                    0.718 |                                     0.766 |                                        0.643 |                                                0.817 |                                 0.825 |                                    0.818 |
| ('reports', 'normal_ecg')                               |                                                    0.911 |                                     0.910 |                                        0.910 |                                                0.897 |                                 0.898 |                                    0.897 |
| ('reports', 'reported_infarct_pattern')                 |                                                    0.529 |                                     0.646 |                                        0.407 |                                                0.730 |                                 0.728 |                                    0.723 |
| ('reports', 'sinus_bradycardia')                        |                                                    0.967 |                                     0.968 |                                        0.961 |                                                0.947 |                                 0.956 |                                    0.952 |
| ('reports', 'sinus_tachycardia')                        |                                                    0.967 |                                     0.968 |                                        0.959 |                                                0.919 |                                 0.940 |                                    0.925 |
| ('reports', 'st_t_changes')                             |                                                    0.702 |                                     0.762 |                                        0.583 |                                                0.786 |                                 0.800 |                                    0.783 |

## Loading of each uncorrected class vector on the shared text direction unit(mu)

|                                                         |   infonce_mix0.00_st-1.00 |   soft_mix1.00_st0.20 |
|:--------------------------------------------------------|--------------------------:|----------------------:|
| ('clinical', 'atrial_fibrillation')                     |                    -0.403 |                -0.084 |
| ('clinical', 'intraventricular_conduction_abnormality') |                    -0.534 |                -0.134 |
| ('clinical', 'left_ventricular_hypertrophy')            |                    -0.195 |                -0.047 |
| ('clinical', 'normal_ecg')                              |                     0.579 |                -0.070 |
| ('clinical', 'reported_infarct_pattern')                |                    -0.257 |                -0.176 |
| ('clinical', 'sinus_bradycardia')                       |                    -0.148 |                -0.029 |
| ('clinical', 'sinus_tachycardia')                       |                    -0.229 |                 0.047 |
| ('clinical', 'st_t_changes')                            |                    -0.352 |                 0.080 |
| ('reports', 'atrial_fibrillation')                      |                    -0.159 |                 0.001 |
| ('reports', 'intraventricular_conduction_abnormality')  |                    -0.277 |                 0.570 |
| ('reports', 'left_ventricular_hypertrophy')             |                     0.154 |                 0.652 |
| ('reports', 'normal_ecg')                               |                     0.984 |                -0.015 |
| ('reports', 'reported_infarct_pattern')                 |                     0.569 |                 0.563 |
| ('reports', 'sinus_bradycardia')                        |                     0.136 |                 0.350 |
| ('reports', 'sinus_tachycardia')                        |                    -0.020 |                 0.824 |
| ('reports', 'st_t_changes')                             |                     0.564 |                 0.743 |

## AUROC of the patient-level offset z.mu alone, per concept

| concept                                 |   infonce_mix0.00_st-1.00 |   soft_mix1.00_st0.20 |
|:----------------------------------------|--------------------------:|----------------------:|
| atrial_fibrillation                     |                     0.256 |                 0.590 |
| intraventricular_conduction_abnormality |                     0.173 |                 0.593 |
| left_ventricular_hypertrophy            |                     0.313 |                 0.581 |
| normal_ecg                              |                     0.894 |                 0.194 |
| reported_infarct_pattern                |                     0.327 |                 0.567 |
| sinus_bradycardia                       |                     0.417 |                 0.539 |
| sinus_tachycardia                       |                     0.303 |                 0.547 |
| st_t_changes                            |                     0.373 |                 0.594 |

## Gap (soft - InfoNCE), test-patient bootstrap (1000 resamples, seed-0 heads)

| family   | variant                 | concepts         |   gap |    lo |    hi |   P(gap<=0) |
|:---------|:------------------------|:-----------------|------:|------:|------:|------------:|
| clinical | uncorrected             | all 8            | 0.042 | 0.028 | 0.057 |       0.000 |
| clinical | uncorrected             | excl. normal ECG | 0.045 | 0.029 | 0.063 |       0.000 |
| clinical | direct z.mu subtraction | all 8            | 0.027 | 0.017 | 0.037 |       0.000 |
| clinical | direct z.mu subtraction | excl. normal ECG | 0.015 | 0.005 | 0.026 |       0.003 |
| clinical | pipeline                | all 8            | 0.041 | 0.032 | 0.050 |       0.000 |
| clinical | pipeline                | excl. normal ECG | 0.016 | 0.007 | 0.024 |       0.000 |
| reports  | uncorrected             | all 8            | 0.088 | 0.069 | 0.107 |       0.000 |
| reports  | uncorrected             | excl. normal ECG | 0.103 | 0.080 | 0.124 |       0.000 |
| reports  | direct z.mu subtraction | all 8            | 0.038 | 0.025 | 0.051 |       0.000 |
| reports  | direct z.mu subtraction | excl. normal ECG | 0.046 | 0.030 | 0.061 |       0.000 |
| reports  | pipeline                | all 8            | 0.014 | 0.004 | 0.025 |       0.005 |
| reports  | pipeline                | excl. normal ECG | 0.018 | 0.006 | 0.031 |       0.003 |

## Gap by training seed (full test set; seed 0 = frozen run's heads)

|                                                       |     0 |     1 |     2 |   mean |   min |   max |
|:------------------------------------------------------|------:|------:|------:|-------:|------:|------:|
| clinical | uncorrected | all 8                        | 0.042 | 0.055 | 0.038 |  0.045 | 0.038 | 0.055 |
| clinical | uncorrected | excl. normal ECG             | 0.045 | 0.059 | 0.041 |  0.049 | 0.041 | 0.059 |
| clinical | direct z.mu subtraction | all 8            | 0.027 | 0.029 | 0.017 |  0.025 | 0.017 | 0.029 |
| clinical | direct z.mu subtraction | excl. normal ECG | 0.015 | 0.019 | 0.005 |  0.013 | 0.005 | 0.019 |
| clinical | pipeline | all 8                           | 0.041 | 0.038 | 0.028 |  0.036 | 0.028 | 0.041 |
| clinical | pipeline | excl. normal ECG                | 0.016 | 0.019 | 0.007 |  0.014 | 0.007 | 0.019 |
| reports | uncorrected | all 8                         | 0.088 | 0.092 | 0.099 |  0.093 | 0.088 | 0.099 |
| reports | uncorrected | excl. normal ECG              | 0.103 | 0.107 | 0.115 |  0.108 | 0.103 | 0.115 |
| reports | direct z.mu subtraction | all 8             | 0.038 | 0.040 | 0.043 |  0.040 | 0.038 | 0.043 |
| reports | direct z.mu subtraction | excl. normal ECG  | 0.046 | 0.047 | 0.051 |  0.048 | 0.046 | 0.051 |
| reports | pipeline | all 8                            | 0.014 | 0.016 | 0.019 |  0.016 | 0.014 | 0.019 |
| reports | pipeline | excl. normal ECG                 | 0.018 | 0.020 | 0.024 |  0.021 | 0.018 | 0.024 |

## Excluded ECGs by split

| split   |   analysed |   excluded |       All |
|:--------|-----------:|-----------:|----------:|
| test    |   3959.000 |     46.000 |  4005.000 |
| train   |  15807.000 |    198.000 | 16005.000 |
| val     |    994.000 |      7.000 |  1001.000 |
| All     |  20760.000 |    251.000 | 21011.000 |


| status             |    ECGs |
|:-------------------|--------:|
| non-finite samples | 251.000 |

## Concept prevalence (positive / eligible), excluded vs analysed ECGs

| group    |   ECGs |   normal_ecg |   atrial_fibrillation |   reported_infarct_pattern |   intraventricular_conduction_abnormality |   left_ventricular_hypertrophy |   st_t_changes |   sinus_bradycardia |   sinus_tachycardia |
|:---------|-------:|-------------:|----------------------:|---------------------------:|------------------------------------------:|-------------------------------:|---------------:|--------------------:|--------------------:|
| analysed |  20760 |        0.167 |                 0.092 |                      0.1   |                                     0.172 |                          0.06  |          0.38  |               0.132 |               0.088 |
| excluded |    251 |        0.137 |                 0.153 |                      0.144 |                                     0.211 |                          0.049 |          0.442 |               0.124 |               0.104 |
