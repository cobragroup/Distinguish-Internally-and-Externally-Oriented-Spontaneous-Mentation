# Distinguishing Internally and Externally Oriented Spontaneous Mentation

This research project investigates whether internally-oriented (I) and externally-oriented (E) spontaneous mentation states can be distinguished using machine learning classifiers applied to EEG and fMRI data. The analysis uses Descriptive Experience Sampling (DES) methodology with beep-triggered experience sampling.

This repository contains the official implementation for the paper "Brain network and oscillation dynamics mirror the inward and outward orientation of spontaneous though" by Hampejs et al. (2025).
Preprint of the paper is available on bioRxiv at <https://doi.org/10.1101/2025.10.31.684884>

## Overview

The pipeline performs time-resolved classification of I/E states across different sliding window sizes and positions relative to the DES beep signal. Systematic exploration of temporal windows ranging from 0.1-2.0 seconds in size, positioned from -0.5 to -5.0 seconds before the beep.

## Directory Structure

```
.
├── cl01_classify_ie.m          # Main MATLAB classification script
├── cl02_tfce.ipynb             # Python TFCE analysis notebook
├── .gitignore                  
├── classifiers/
│   └── svm_in_ex.m             # SVM classifier with cost-sensitive learning and LOOCV
├── data/
│   ├── eeg_feat.mat            # EEG feature matrices
│   └── fmri_feat.mat           # fMRI feature matrices
├── results/
│   ├── eeg_results.mat         # EEG classification outputs (accuracy, balanced accuracy, permutations)
│   └── fmri_results.mat        # fMRI classification outputs (accuracy, balanced accuracy, permutations)
└── figures/
    ├── tfce_acc_eeg.png        # TFCE visualization for EEG results
    └── tfce_acc_fmri.png       # TFCE visualization for fMRI results
```

## Data Format

### Feature Matrices (.mat files)

The input data is loaded as a MATLAB struct with the following fields:

- `x`: 4D array `[features × window_positions × window_sizes × feature_types]`
- `y`: Label vector (1 = internal, 2 = external)
- `sub_id`: Subject identifiers for proper cross-validation

Example data structure:
```matlab
data = load('data/fmri_feat.mat');
x = data.x;      % Features
y = data.y;      % Labels
sub_id = data.sub_id; % Subject IDs
```

## Pipeline Steps

### Step 1: Classification (`cl01_classify_ie.m`)

This MATLAB script performs leave-one-subject-out cross-validated SVM classification:

**Workflow:**
1. Loads feature data from `.mat` file (EEG or fMRI)
2. Iterates through all window sizes positions
3. Trains linear SVM classifier with cost-sensitive learning for class imbalance
4. Stores accuracy and confusion matrix for each configuration
5. Performs 5000 label permutations for null distribution estimation
6. Saves results to `results/{modality}_results.mat`

**Key Parameters:**
- Classifier: Linear SVM with cost-sensitive learning
- Cross-validation: Leave-one-subject-out (LOOCV)
- Window sizes correspond to 0.1-2.0 seconds
- Window positions correspond to -0.5 to -5.0 seconds before beep
- Permutations: 5000 for TFCE significance testing

**Parallel Processing:**
- Uses parallel pool with 50 workers for accelerated computation

### Step 2: TFCE Analysis (`cl02_tfce.ipynb`)

This Jupyter notebook applies TFCE enhancement and statistical testing:

**Workflow:**
1. Loads classification results from Step 1 (`{modality}_results.mat`)
2. Applies TFCE (Threshold-Free Cluster Enhancement) to accuracy maps
3. Computes TFCE scores for permuted null distributions (5000 permutations)
4. Calculates p-values by comparing observed vs. null TFCE scores
5. Generates significance mask at p < 0.05
6. Identifies optimal window size/position with maximum significant accuracy
7. Visualizes results with highlighted significant regions
8. Saves figure to `figures/tfce_acc_{modality}.png`

**TFCE Parameters:**
- H (height exponent): 2.0
- E (extent exponent): 0.5
- dh (step size): 0.01
- Significance threshold: p < 0.05

## Output Files

### Classification Results (`results/*.mat`)

| File | Contents |
|------|----------|
| `eeg_results.mat` | `acc` (accuracy), `acc_bal` (balanced accuracy), `acc_perm` (permuted accuracies) |
| `fmri_results.mat` | `acc` (accuracy), `acc_bal` (balanced accuracy), `acc_perm` (permuted accuracies) |

### Visualization Outputs (`figures/`)

| File | Description |
|------|-------------|
| `tfce_acc_eeg.png` | EEG classification accuracy heatmap with TFCE-significant regions highlighted |
| `tfce_acc_fmri.png` | fMRI classification accuracy heatmap with TFCE-significant regions highlighted |

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
- **X-axis**: Window position relative to beep (negative values = before beep, ranging from -0.5 to -5.0 seconds)
- **Y-axis**: Window size in seconds (ranging from 0.1 to 2.0 seconds)
- **Color**: Classification accuracy (red colormap)
- **Yellow markers**: Statistically significant regions (TFCE-corrected p<0.05)

## Dependencies

### MATLAB
- Statistics and Machine Learning Toolbox (for `fitcsvm`, `crossval`, `kfoldPredict`)
- Parallel Computing Toolbox (for `parpool`, `parfor`)

### Python
- numpy
- pandas
- scipy (for `ndimage`, `io`)
- matplotlib
- scikit-learn (not directly used, but may be needed for extensions)

## Usage

### Running the MATLAB Classification

```matlab
% Select modality - 'eeg' or 'fmri'
modality = 'fmri';

switch modality
    case 'fmri'
        data = load('data/fmri_feat.mat');
    case 'eeg'
        data = load('data/eeg_feat.mat');
end
```

### Running the TFCE Analysis

```python
# Select modality - 'eeg' or 'fmri'
modality = 'eeg'

match modality:
    case 'fmri':
        data = sio.loadmat('data/fmri_feat.mat')
        results = sio.loadmat('results/fmri_results.mat')
    case 'eeg':
        data = sio.loadmat('data/eeg_feat.mat')
        results = sio.loadmat('results/eeg_results.mat')
```
