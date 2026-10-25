# MIMIC-IV-ECG replication (frozen protocol; labels are machine-report-derived)

best soft model by test clinical-centered macro AUROC (reported only): soft_mix1.00_st0.20

## Macro test AUROC by readout (InfoNCE vs best of the 8 soft models)

| readout      | centered   |   InfoNCE |   best soft (max over 8) |   gap |
|:-------------|:-----------|----------:|-------------------------:|------:|
| clinical     | False      |     0.704 |                    0.754 | 0.050 |
| clinical     | True       |     0.720 |                    0.761 | 0.041 |
| names        | False      |     0.719 |                    0.739 | 0.020 |
| names        | True       |     0.715 |                    0.743 | 0.027 |
| report_style | False      |     0.746 |                    0.772 | 0.027 |
| report_style | True       |     0.778 |                    0.786 | 0.008 |
| reports      | False      |     0.774 |                    0.868 | 0.095 |
| reports      | True       |     0.855 |                    0.875 | 0.020 |

## Per-concept AUROC, clinical descriptions (primary) and report prototypes

| readout   | centered   | concept                                 |   infonce_mix0.00_st-1.00 |   soft_mix1.00_st0.20 |
|:----------|:-----------|:----------------------------------------|--------------------------:|----------------------:|
| clinical  | False      | atrial_fibrillation                     |                     0.706 |                 0.749 |
| clinical  | False      | intraventricular_conduction_abnormality |                     0.847 |                 0.818 |
| clinical  | False      | left_ventricular_hypertrophy            |                     0.518 |                 0.735 |
| clinical  | False      | normal_ecg                              |                     0.827 |                 0.848 |
| clinical  | False      | reported_infarct_pattern                |                     0.534 |                 0.635 |
| clinical  | False      | sinus_bradycardia                       |                     0.844 |                 0.819 |
| clinical  | False      | sinus_tachycardia                       |                     0.827 |                 0.743 |
| clinical  | False      | st_t_changes                            |                     0.528 |                 0.622 |
| clinical  | True       | atrial_fibrillation                     |                     0.753 |                 0.768 |
| clinical  | True       | intraventricular_conduction_abnormality |                     0.857 |                 0.830 |
| clinical  | True       | left_ventricular_hypertrophy            |                     0.626 |                 0.747 |
| clinical  | True       | normal_ecg                              |                     0.632 |                 0.850 |
| clinical  | True       | reported_infarct_pattern                |                     0.627 |                 0.646 |
| clinical  | True       | sinus_bradycardia                       |                     0.837 |                 0.834 |
| clinical  | True       | sinus_tachycardia                       |                     0.837 |                 0.765 |
| clinical  | True       | st_t_changes                            |                     0.590 |                 0.647 |
| reports   | False      | atrial_fibrillation                     |                     0.811 |                 0.867 |
| reports   | False      | intraventricular_conduction_abnormality |                     0.915 |                 0.930 |
| reports   | False      | left_ventricular_hypertrophy            |                     0.643 |                 0.818 |
| reports   | False      | normal_ecg                              |                     0.910 |                 0.897 |
| reports   | False      | reported_infarct_pattern                |                     0.407 |                 0.723 |
| reports   | False      | sinus_bradycardia                       |                     0.961 |                 0.952 |
| reports   | False      | sinus_tachycardia                       |                     0.959 |                 0.925 |
| reports   | False      | st_t_changes                            |                     0.583 |                 0.783 |
| reports   | True       | atrial_fibrillation                     |                     0.885 |                 0.873 |
| reports   | True       | intraventricular_conduction_abnormality |                     0.934 |                 0.934 |
| reports   | True       | left_ventricular_hypertrophy            |                     0.766 |                 0.825 |
| reports   | True       | normal_ecg                              |                     0.910 |                 0.898 |
| reports   | True       | reported_infarct_pattern                |                     0.646 |                 0.728 |
| reports   | True       | sinus_bradycardia                       |                     0.968 |                 0.956 |
| reports   | True       | sinus_tachycardia                       |                     0.968 |                 0.940 |
| reports   | True       | st_t_changes                            |                     0.762 |                 0.800 |

## Paired selection (100 resamples; diff vs full labeled validation)

| target                  | selector           |   test_auroc |   diff |   diff_lo |   diff_hi | top                     |   top_freq |
|:------------------------|:-------------------|-------------:|-------:|----------:|----------:|:------------------------|-----------:|
| clinical                | val R@1_dupaware   |        0.753 | -0.001 |    -0.011 |     0.007 | soft_mix1.00_st0.10     |      0.740 |
| clinical                | val R@1_exact      |        0.705 | -0.049 |    -0.067 |    -0.018 | infonce_mix0.00_st-1.00 |      0.330 |
| clinical                | val cka_rbf        |        0.705 | -0.049 |    -0.062 |    -0.035 | infonce_mix0.00_st-1.00 |      0.980 |
| clinical                | val cos_margin     |        0.706 | -0.049 |    -0.062 |    -0.035 | infonce_mix0.00_st-1.00 |      1.000 |
| clinical                | val labels (100)   |        0.754 | -0.000 |    -0.008 |     0.007 | soft_mix1.00_st0.10     |      0.870 |
| clinical                | val labels (25)    |        0.750 | -0.004 |    -0.025 |     0.006 | soft_mix1.00_st0.10     |      0.650 |
| clinical                | val labels (50)    |        0.753 | -0.001 |    -0.010 |     0.007 | soft_mix1.00_st0.10     |      0.840 |
| clinical                | val labels (all)   |        0.754 |  0.000 |     0.000 |     0.000 | soft_mix1.00_st0.10     |      0.910 |
| clinical                | val mutual_knn@5   |        0.707 | -0.047 |    -0.064 |    -0.029 | infonce_mix0.00_st-1.00 |      0.450 |
| clinical                | val procrustes_sym |        0.706 | -0.049 |    -0.062 |    -0.035 | infonce_mix0.00_st-1.00 |      1.000 |
| clinical [centered]     | val R@1_dupaware   |        0.761 | -0.001 |    -0.006 |     0.003 | soft_mix1.00_st0.10     |      0.740 |
| clinical [centered]     | val R@1_exact      |        0.722 | -0.039 |    -0.051 |    -0.016 | infonce_mix0.00_st-1.00 |      0.330 |
| clinical [centered]     | val cka_rbf        |        0.721 | -0.041 |    -0.049 |    -0.032 | infonce_mix0.00_st-1.00 |      0.980 |
| clinical [centered]     | val cos_margin     |        0.721 | -0.041 |    -0.050 |    -0.032 | infonce_mix0.00_st-1.00 |      1.000 |
| clinical [centered]     | val labels (100)   |        0.762 |  0.000 |    -0.003 |     0.003 | soft_mix1.00_st0.20     |      0.840 |
| clinical [centered]     | val labels (25)    |        0.760 | -0.001 |    -0.020 |     0.004 | soft_mix1.00_st0.20     |      0.590 |
| clinical [centered]     | val labels (50)    |        0.761 | -0.001 |    -0.008 |     0.003 | soft_mix1.00_st0.20     |      0.630 |
| clinical [centered]     | val labels (all)   |        0.761 |  0.000 |     0.000 |     0.000 | soft_mix1.00_st0.20     |      0.880 |
| clinical [centered]     | val mutual_knn@5   |        0.724 | -0.038 |    -0.050 |    -0.016 | infonce_mix0.00_st-1.00 |      0.450 |
| clinical [centered]     | val procrustes_sym |        0.721 | -0.041 |    -0.050 |    -0.032 | infonce_mix0.00_st-1.00 |      1.000 |
| names                   | val R@1_dupaware   |        0.735 | -0.003 |    -0.018 |     0.002 | soft_mix1.00_st0.10     |      0.740 |
| names                   | val R@1_exact      |        0.720 | -0.019 |    -0.038 |    -0.003 | infonce_mix0.00_st-1.00 |      0.330 |
| names                   | val cka_rbf        |        0.719 | -0.019 |    -0.030 |    -0.008 | infonce_mix0.00_st-1.00 |      0.980 |
| names                   | val cos_margin     |        0.719 | -0.019 |    -0.030 |    -0.008 | infonce_mix0.00_st-1.00 |      1.000 |
| names                   | val labels (100)   |        0.739 |  0.000 |    -0.003 |     0.006 | soft_mix1.00_st0.10     |      0.960 |
| names                   | val labels (25)    |        0.735 | -0.003 |    -0.018 |     0.004 | soft_mix1.00_st0.10     |      0.640 |
| names                   | val labels (50)    |        0.736 | -0.002 |    -0.018 |     0.000 | soft_mix1.00_st0.10     |      0.750 |
| names                   | val labels (all)   |        0.738 |  0.000 |     0.000 |     0.000 | soft_mix1.00_st0.10     |      0.930 |
| names                   | val mutual_knn@5   |        0.721 | -0.017 |    -0.033 |    -0.005 | infonce_mix0.00_st-1.00 |      0.450 |
| names                   | val procrustes_sym |        0.719 | -0.019 |    -0.030 |    -0.008 | infonce_mix0.00_st-1.00 |      1.000 |
| names [centered]        | val R@1_dupaware   |        0.740 | -0.003 |    -0.015 |     0.004 | soft_mix1.00_st0.10     |      0.740 |
| names [centered]        | val R@1_exact      |        0.718 | -0.024 |    -0.036 |    -0.008 | infonce_mix0.00_st-1.00 |      0.330 |
| names [centered]        | val cka_rbf        |        0.715 | -0.027 |    -0.037 |    -0.015 | infonce_mix0.00_st-1.00 |      0.980 |
| names [centered]        | val cos_margin     |        0.715 | -0.027 |    -0.037 |    -0.015 | infonce_mix0.00_st-1.00 |      1.000 |
| names [centered]        | val labels (100)   |        0.743 |  0.000 |     0.000 |     0.008 | soft_mix1.00_st0.10     |      0.970 |
| names [centered]        | val labels (25)    |        0.740 | -0.003 |    -0.016 |     0.008 | soft_mix1.00_st0.10     |      0.720 |
| names [centered]        | val labels (50)    |        0.741 | -0.001 |    -0.015 |     0.009 | soft_mix1.00_st0.10     |      0.860 |
| names [centered]        | val labels (all)   |        0.743 |  0.000 |     0.000 |     0.000 | soft_mix1.00_st0.10     |      0.950 |
| names [centered]        | val mutual_knn@5   |        0.719 | -0.024 |    -0.035 |    -0.007 | infonce_mix0.00_st-1.00 |      0.450 |
| names [centered]        | val procrustes_sym |        0.715 | -0.027 |    -0.037 |    -0.015 | infonce_mix0.00_st-1.00 |      1.000 |
| report_style            | val R@1_dupaware   |        0.770 | -0.003 |    -0.016 |     0.000 | soft_mix1.00_st0.10     |      0.740 |
| report_style            | val R@1_exact      |        0.746 | -0.028 |    -0.045 |    -0.012 | infonce_mix0.00_st-1.00 |      0.330 |
| report_style            | val cka_rbf        |        0.747 | -0.027 |    -0.036 |    -0.016 | infonce_mix0.00_st-1.00 |      0.980 |
| report_style            | val cos_margin     |        0.747 | -0.027 |    -0.036 |    -0.016 | infonce_mix0.00_st-1.00 |      1.000 |
| report_style            | val labels (100)   |        0.774 |  0.000 |     0.000 |     0.000 | soft_mix1.00_st0.10     |      1.000 |
| report_style            | val labels (25)    |        0.768 | -0.006 |    -0.028 |     0.000 | soft_mix1.00_st0.10     |      0.650 |
| report_style            | val labels (50)    |        0.772 | -0.002 |    -0.020 |     0.000 | soft_mix1.00_st0.10     |      0.860 |
| report_style            | val labels (all)   |        0.774 |  0.000 |     0.000 |     0.000 | soft_mix1.00_st0.10     |      1.000 |
| report_style            | val mutual_knn@5   |        0.747 | -0.027 |    -0.040 |    -0.016 | infonce_mix0.00_st-1.00 |      0.450 |
| report_style            | val procrustes_sym |        0.747 | -0.027 |    -0.036 |    -0.016 | infonce_mix0.00_st-1.00 |      1.000 |
| report_style [centered] | val R@1_dupaware   |        0.779 | -0.002 |    -0.013 |     0.007 | soft_mix1.00_st0.10     |      0.740 |
| report_style [centered] | val R@1_exact      |        0.779 | -0.002 |    -0.018 |     0.011 | infonce_mix0.00_st-1.00 |      0.330 |
| report_style [centered] | val cka_rbf        |        0.779 | -0.002 |    -0.009 |     0.006 | infonce_mix0.00_st-1.00 |      0.980 |
| report_style [centered] | val cos_margin     |        0.779 | -0.002 |    -0.009 |     0.006 | infonce_mix0.00_st-1.00 |      1.000 |
| report_style [centered] | val labels (100)   |        0.781 |  0.001 |    -0.011 |     0.011 | soft_mix1.00_st0.10     |      0.390 |
| report_style [centered] | val labels (25)    |        0.780 | -0.001 |    -0.014 |     0.012 | soft_mix1.00_st0.10     |      0.310 |
| report_style [centered] | val labels (50)    |        0.781 |  0.000 |    -0.015 |     0.010 | soft_mix1.00_st0.02     |      0.290 |
| report_style [centered] | val labels (all)   |        0.781 |  0.000 |     0.000 |     0.000 | soft_mix1.00_st0.10     |      0.450 |
| report_style [centered] | val mutual_knn@5   |        0.780 | -0.001 |    -0.010 |     0.011 | infonce_mix0.00_st-1.00 |      0.450 |
| report_style [centered] | val procrustes_sym |        0.779 | -0.002 |    -0.009 |     0.006 | infonce_mix0.00_st-1.00 |      1.000 |
| reports                 | val R@1_dupaware   |        0.866 | -0.002 |    -0.009 |     0.003 | soft_mix1.00_st0.10     |      0.740 |
| reports                 | val R@1_exact      |        0.782 | -0.086 |    -0.112 |    -0.016 | infonce_mix0.00_st-1.00 |      0.330 |
| reports                 | val cka_rbf        |        0.774 | -0.094 |    -0.112 |    -0.076 | infonce_mix0.00_st-1.00 |      0.980 |
| reports                 | val cos_margin     |        0.774 | -0.094 |    -0.112 |    -0.076 | infonce_mix0.00_st-1.00 |      1.000 |
| reports                 | val labels (100)   |        0.868 | -0.000 |    -0.006 |     0.006 | soft_mix1.00_st0.10     |      0.960 |
| reports                 | val labels (25)    |        0.863 | -0.005 |    -0.020 |     0.000 | soft_mix1.00_st0.10     |      0.510 |
| reports                 | val labels (50)    |        0.867 | -0.001 |    -0.007 |     0.003 | soft_mix1.00_st0.10     |      0.860 |
| reports                 | val labels (all)   |        0.868 |  0.000 |     0.000 |     0.000 | soft_mix1.00_st0.10     |      0.960 |
| reports                 | val mutual_knn@5   |        0.779 | -0.088 |    -0.112 |    -0.063 | infonce_mix0.00_st-1.00 |      0.450 |
| reports                 | val procrustes_sym |        0.774 | -0.094 |    -0.112 |    -0.076 | infonce_mix0.00_st-1.00 |      1.000 |
| reports [centered]      | val R@1_dupaware   |        0.871 |  0.001 |    -0.005 |     0.005 | soft_mix1.00_st0.10     |      0.740 |
| reports [centered]      | val R@1_exact      |        0.858 | -0.012 |    -0.026 |     0.007 | infonce_mix0.00_st-1.00 |      0.330 |
| reports [centered]      | val cka_rbf        |        0.855 | -0.015 |    -0.026 |    -0.004 | infonce_mix0.00_st-1.00 |      0.980 |
| reports [centered]      | val cos_margin     |        0.855 | -0.015 |    -0.026 |    -0.004 | infonce_mix0.00_st-1.00 |      1.000 |
| reports [centered]      | val labels (100)   |        0.870 |  0.000 |    -0.008 |     0.008 | soft_mix1.00_st0.20     |      0.610 |
| reports [centered]      | val labels (25)    |        0.870 | -0.000 |    -0.009 |     0.007 | soft_mix1.00_st0.20     |      0.550 |
| reports [centered]      | val labels (50)    |        0.871 |  0.001 |    -0.007 |     0.011 | soft_mix1.00_st0.20     |      0.440 |
| reports [centered]      | val labels (all)   |        0.870 |  0.000 |     0.000 |     0.000 | soft_mix1.00_st0.20     |      0.650 |
| reports [centered]      | val mutual_knn@5   |        0.858 | -0.012 |    -0.025 |     0.003 | infonce_mix0.00_st-1.00 |      0.450 |
| reports [centered]      | val procrustes_sym |        0.855 | -0.015 |    -0.026 |    -0.004 | infonce_mix0.00_st-1.00 |      1.000 |

## Query source (label-informed; single-concept reports)

| source            |   normal_ecg |   atrial_fibrillation |   reported_infarct_pattern |   intraventricular_conduction_abnormality |   left_ventricular_hypertrophy |   st_t_changes |   sinus_bradycardia |   sinus_tachycardia |
|:------------------|-------------:|----------------------:|---------------------------:|------------------------------------------:|-------------------------------:|---------------:|--------------------:|--------------------:|
| train             |         2590 |                   300 |                        364 |                                       984 |                            111 |           2574 |                1100 |                 706 |
| val seen string   |          114 |                     7 |                         11 |                                        37 |                              2 |             76 |                  56 |                  25 |
| val unseen string |            0 |                     6 |                          8 |                                        21 |                              3 |             94 |                  11 |                   8 |
|   k | classes                                                                                                                               | source            | centered   |   infonce_mix0.00_st-1.00 |   soft_mix1.00_st0.20 |   gap |
|----:|:--------------------------------------------------------------------------------------------------------------------------------------|:------------------|:-----------|--------------------------:|----------------------:|------:|
|   5 | atrial_fibrillation,reported_infarct_pattern,intraventricular_conduction_abnormality,st_t_changes,sinus_bradycardia,sinus_tachycardia | train             | False      |                     0.771 |                 0.843 | 0.072 |
|   5 | atrial_fibrillation,reported_infarct_pattern,intraventricular_conduction_abnormality,st_t_changes,sinus_bradycardia,sinus_tachycardia | train             | True       |                     0.832 |                 0.854 | 0.022 |
|   5 | atrial_fibrillation,reported_infarct_pattern,intraventricular_conduction_abnormality,st_t_changes,sinus_bradycardia,sinus_tachycardia | val seen string   | False      |                     0.748 |                 0.824 | 0.076 |
|   5 | atrial_fibrillation,reported_infarct_pattern,intraventricular_conduction_abnormality,st_t_changes,sinus_bradycardia,sinus_tachycardia | val seen string   | True       |                     0.804 |                 0.838 | 0.034 |
|   5 | atrial_fibrillation,reported_infarct_pattern,intraventricular_conduction_abnormality,st_t_changes,sinus_bradycardia,sinus_tachycardia | val unseen string | False      |                     0.814 |                 0.848 | 0.033 |
|   5 | atrial_fibrillation,reported_infarct_pattern,intraventricular_conduction_abnormality,st_t_changes,sinus_bradycardia,sinus_tachycardia | val unseen string | True       |                     0.843 |                 0.849 | 0.006 |
|  10 | intraventricular_conduction_abnormality,st_t_changes,sinus_bradycardia                                                                | train             | False      |                     0.813 |                 0.880 | 0.066 |
|  10 | intraventricular_conduction_abnormality,st_t_changes,sinus_bradycardia                                                                | train             | True       |                     0.872 |                 0.890 | 0.018 |
|  10 | intraventricular_conduction_abnormality,st_t_changes,sinus_bradycardia                                                                | val seen string   | False      |                     0.789 |                 0.861 | 0.071 |
|  10 | intraventricular_conduction_abnormality,st_t_changes,sinus_bradycardia                                                                | val seen string   | True       |                     0.840 |                 0.876 | 0.036 |
|  10 | intraventricular_conduction_abnormality,st_t_changes,sinus_bradycardia                                                                | val unseen string | False      |                     0.864 |                 0.875 | 0.011 |
|  10 | intraventricular_conduction_abnormality,st_t_changes,sinus_bradycardia                                                                | val unseen string | True       |                     0.863 |                 0.874 | 0.010 |
|  25 | st_t_changes                                                                                                                          | train             | False      |                     0.592 |                 0.781 | 0.189 |
|  25 | st_t_changes                                                                                                                          | train             | True       |                     0.748 |                 0.798 | 0.050 |
|  25 | st_t_changes                                                                                                                          | val seen string   | False      |                     0.480 |                 0.716 | 0.235 |
|  25 | st_t_changes                                                                                                                          | val seen string   | True       |                     0.610 |                 0.753 | 0.143 |
|  25 | st_t_changes                                                                                                                          | val unseen string | False      |                     0.735 |                 0.767 | 0.032 |
|  25 | st_t_changes                                                                                                                          | val unseen string | True       |                     0.750 |                 0.769 | 0.019 |
