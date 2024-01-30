- running hpo on rf yielded:

    | Classifier  | Average Score  | Highest Score  | Lowest Score  | Annotation |
    |---|---|---|---| --- |
    | md = 12, mf = 2, ne = 507, rs=873 | 74.53  %  |  -  | - | RF_3 |
    | md = 5, mf = 5, ne = 2284, rs=629 | 71.20  %  |  -  | - | / |
    | md = 3, mf = 7, ne = 1121, rs=692 | 67.56  %  |  -  | - | / |
    | md = 18, mf = 3, ne = 816, rs=331 | 75.40  %  |  -  | - | / |
    | md = 8, mf = 2, ne = 2761, rs=367 | 72.89  %  |  -  | - | / |
    | md = 3, mf = 7, ne = 2439, rs=12 | 67.66  %  |  -  | - | / |
    | md = 7, mf = 1, ne = 2182, rs=3 | 72.55 %  |  -  | - | / |
    | md = 11, mf = 2, ne = 560, rs=325 | 73.94  %  |  -  | - | / |
    | md = 15, mf = 6, ne = 831, rs=1000 | 75.22  %  |  -  | - | RF_1 |
    | md = 14, mf = 4, ne = 1558, rs=47 | 74.94  %  |  -  | - | RF-2 |
    | md = 12, mf = 1, ne = 2000, rs=42 | 74.87  %  |  -  | - | (Base) |

- running hpo on xgb yielded:

    | Classifier  | Average Score  | Highest Score  | Lowest Score  | Annotation |
    |---|---|---|---| --- |
    | md = 12, lr = 0.02, subsample = 0.8, colsample_bytree = 0.4, ne = 2000, rs=42 | 70.80  %  |  -  | - | (Base) |

- running hpo on lgbm yielded:

    | Classifier  | Average Score  | Highest Score  | Lowest Score  | Annotation |
    |---|---|---|---| --- |
    | md = 12, lr = 0.02, subsample = 0.8, colsample_bytree = 0.4, ne = 2000, rs=42 | 71.24  %  |  -  | - | (Base) |
