# Distinguishing Internally and Externally Oriented Spontaneous Mentation

This research project investigates whether internally-oriented (I) and externally-oriented (E) spontaneous mentation states can be distinguished using machine learning classifiers applied to EEG and fMRI data. The analysis uses Descriptive Experience Sampling (DES) methodology with beep-triggered experience sampling.

## Overview

The pipeline performs time-resolved classification of I/E states across different sliding window sizes and positions relative to the DES beep signal. Key innovations include:

- **Sliding window approach**: Systematic exploration of temporal windows ranging from 0.1-2.0 seconds in size, positioned from -0.5 to -5.0 seconds before the beep
- **TFCE correction**: Threshold-Free Cluster Enhancement for multiple comparison correction using permutation testing
- **Multi-modal support**: Compatible with both EEG and fMRI feature data

## Directory Structure

```
.
├── cl01_classify_ie.m          # Main MATLAB classification script
├── cl02_tfce.ipynb             # Python TFCE analysis notebook
├── classifiers/
│   └── svm_in_ex.m             # SVM classifier with cross-validation
├── functions/                   # Additional helper functions (empty)
├── data/
│   ├── eeg_feat_*.mat          # EEG feature matrices
│   └── fmri_feat_*.mat         # fMRI feature matrices
├── results/                     # Classification outputs
│   ├── classified_ie.mat       # Accuracy and confusion matrices
│   └── *.csv                   # Derived results (masks, predictions, etc.)
└── figures/                     # Publication-ready visualizations
```

## Data Format

### Feature Matrices (.mat files)

The input data should be loaded as a MATLAB struct with the following fields:

- `x2`: 4D array `[features × window_positions × window_sizes × feature_types]`
- `y`: Label vector (1 = internal, 2 = external)
- `sub_id`: Subject identifiers for proper cross-validation

Example data structure:
```matlab
data = load('data/eeg_feat_20240710_151724.mat');
feat = data.x2;      % Features
y = data.y;          % Labels
sub_id = data.sub_id; % Subject IDs
```

## Pipeline Steps

### Step 1: Classification (`cl01_classify_ie.m`)

This MATLAB script performs leave-one-subject-out cross-validated SVM classification:

**Workflow:**
1. Loads feature data from `.mat` file
2. Iterates through all window sizes (1-20) and feature positions (1-26)
3. Trains linear SVM classifier with cost-sensitive learning for class imbalance
4. Stores accuracy and confusion matrix for each configuration
5. Performs 5000 label permutations for null distribution estimation
6. Saves results to `results/classified_ie.mat`

**Key Parameters:**
- Classifier: Linear SVM with cost-sensitive learning
- Cross-validation: Leave-one-subject-out
- Window sizes: 1-20 (corresponds to 0.1-2.0 seconds)
- Window positions: 1-26 (corresponds to -0.5 to -5.0 seconds before beep)
- Permutations: 5000 for TFCE significance testing

### Step 2: TFCE Analysis (`cl02_tfce.ipynb`)

This Jupyter notebook applies TFCE enhancement and statistical testing:

**Workflow:**
1. Loads classification results from Step 1
2. Applies TFCE (Threshold-Free Cluster Enhancement) to accuracy maps
3. Computes TFCE scores for permuted null distributions
4. Calculates p-values by comparing observed vs. null TFCE scores
5. Generates significance mask at p < 0.05
6. Identifies optimal window size/position with maximum significant accuracy
7. Visualizes results with highlighted significant regions

**TFCE Parameters:**
- H (height exponent): 2.0
- E (extent exponent): 0.5
- dh (step size): 0.01
- Significance threshold: p < 0.05

## Output Files

### From `cl01_classify_ie.m`

| File | Description |
|------|-------------|
| `results/classified_ie.mat` | Contains `acc` (accuracy), `cm` (confusion matrices), `acc_perm` (permuted accuracies) |

### From `cl02_tfce.ipynb`

| File | Description |
|------|-------------|
| `tfce_mask.csv` | Binary mask of significant (p<0.05) window configurations |
| `best_feature.csv` | Features at optimal window configuration |
| `best_cm.csv` | Confusion matrix at optimal configuration |
| `y_hat.csv` | Predicted labels at optimal configuration |

## Results Interpretation

### Example Output
```
Maximum accuracy of 65.4% (balanced accuracy 64.3%) 
found in a window of size 0.1 sec, 
and -4.1 sec before the beep
```

This indicates that neural features approximately 4 seconds before the beep, analyzed in a 100ms window, contain discriminative information about I/E state with 65.4% accuracy (significantly above chance after TFCE correction).

### Visualization

The pipeline generates heatmaps showing:
- **X-axis**: Window position relative to beep (negative = before beep)
- **Y-axis**: Window size in seconds
- **Color**: Classification accuracy
- **Yellow markers**: Statistically significant regions (TFCE-corrected p<0.05)

## Dependencies

### MATLAB
- Statistics and Machine Learning Toolbox
- Parallel Computing Toolbox (for parallel processing)

### Python
- numpy
- pandas
- scipy
- matplotlib
- scikit-learn
- python-matlab-util (for .mat file reading)

