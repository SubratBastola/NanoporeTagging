NanoporeTagging

Python tools for annotation, automated landmark detection, feature extraction, and clustering of nanopore event recordings containing optical and electrical signals.

The repository provides an end-to-end workflow for working with Axon Binary Format (.abf) recordings:

ABF recording
      │
      ▼
Manual event/window tagging
   Tagger_GUI.py
      │
      ▼
Event/window CSV or Excel
      │
      ▼
Automated landmark detection
   NeuralNetwork.py
      │
      ▼
Predicted event features
      │
      ▼
Multi-file event clustering
    clustering.py
Repository Contents
Tagger_GUI.py

Interactive graphical tool for manually identifying and annotating events in ABF recordings.

The application displays three signal channels:

Optical signal
Reference optical signal
Electrical / ionic-current signal

For each event, the user can mark optical and electrical landmarks.

Optical landmarks
O1 — event start
O2 — event plateau
O3 — event end
Electrical landmarks
E1 — entry baseline
E2 — entry peak
E3 — exit peak
E4 — exit baseline

The program calculates and exports event measurements including:

Event duration
Optical step change (OSC)
Reference optical step change (RefOSC)
Entry baseline and peak current
Exit peak and baseline current
Entry spike
Exit spike
Event and landmark timestamps
Analysis window start/end
Sample metadata

The GUI is implemented using Dash/Plotly and automatically opens in a local web browser.

Run:

python Tagger_GUI.py

You will be prompted to select an .abf recording and optionally an existing .csv or .xlsx event file.

NeuralNetwork.py

Automated event-landmark prediction and feature-extraction pipeline.

The script processes event windows defined in CSV/Excel files, locates the corresponding ABF recordings, extracts the appropriate signal segments, and predicts/refines event landmarks.

Optical landmark model

The included neural network is a 1D convolutional neural network (StrongOpt1D) that predicts three normalized optical landmark positions:

Optical baseline / event start
Optical plateau
Optical event end

The network operates on four representations of the optical signal:

Raw optical trace
Fast moving-average optical trace
Slow moving-average optical trace
Normalized temporal position

These signals are resampled to a fixed-length representation before inference.

A trained PyTorch checkpoint can be selected when the program starts. The repository includes:

best_opt_strict.pt
Hybrid landmark detection

The pipeline does not rely exclusively on neural-network output.

Optical predictions can also be refined using:

z-score transition detection
Bessel low-pass filtering
lag-difference edge detection
historical median/MAD statistics
persistence filtering
level-crossing refinement
coarse event localization as a fallback

Electrical landmarks are determined relative to the detected optical event and then refined using the raw electrical signal.

Resumable processing

The pipeline is designed for large datasets.

Phase A — ABF → NPZ

Event windows are extracted from ABF recordings and stored as temporary .npz files.

Previously generated NPZ files are automatically reused.

Phase B — inference and feature extraction

The trained model predicts event landmarks and calculates event features.

Progress is checkpointed after each event. Interrupted jobs can therefore resume without processing completed events again.

Run:

python NeuralNetwork.py

The program will ask you to:

Select a folder containing event files and ABF recordings.
Select a trained .pt checkpoint.
Review the detected Excel/ABF file matches.
Start processing.

Predictions are written beside the source event file as:

<original_name>__predicted.xlsx

A resumable checkpoint is also generated:

<original_name>__predicted.checkpoint.json

Temporary event segments are stored in:

_tmp_npz/
Required input columns

Event CSV/Excel files supplied to NeuralNetwork.py should contain:

event_id
file_name
sensor
analytes
solution
window_start
window_end

file_name is used to match each row to the corresponding ABF recording.

Default ABF channel mapping

The current automated pipeline uses:

Electrical     = channel 0
Optical        = channel 2
OpticalRef     = channel 3

These settings can be modified in NeuralNetwork.py if the acquisition channel layout is different.

clustering.py

Interactive tool for unsupervised clustering of extracted nanopore events.

Multiple CSV or Excel files can be loaded simultaneously and pooled for analysis.

Supported clustering features include:

Duration
Optical step change (OSC)
Reference optical step change (RefOSC)
Entry peak current
Exit peak current
Entry spike magnitude
Exit spike magnitude

When raw baseline and peak measurements are available, spike amplitudes are derived directly from them:

entry spike = |entry peak - entry baseline|
exit spike  = |exit peak - exit baseline|
Clustering procedure

The program:

Loads and combines events from selected files.
Resolves compatible feature columns.
Optionally applies duration filters.
Handles missing values.
Log-transforms duration when used.
Standardizes clustering features.
Fits full-covariance Gaussian Mixture Models across candidate values of K.
Calculates BIC and AIC.
Performs sequential parametric bootstrap likelihood-ratio tests.
Selects the final number of clusters.
Assigns events to clusters.
Produces visualizations and summary tables.

Clusters are numbered by population size:

C0 = largest cluster
C1 = second-largest cluster
C2 = third-largest cluster
...

The GUI also supports:

Interactive point inspection
Manual event exclusion
Removal and recomputation of clusters
Minimum-cluster-size filtering
Custom cluster names
2D and 3D visualization
Cluster-size summaries
BIC/AIC visualization
Cluster evolution across values of K
Comparison between groups/folders

Run:

python clustering.py
Clustering outputs

Results can be exported as:

pooled_cluster_labels.csv
cluster_summary.csv
cluster_counts_by_file.csv
bootstrap_lrt.csv
k_breakdown.csv
manually_excluded_points.csv
removed_cluster_history.csv
clustering_results.xlsx

A separate labeled CSV is also generated for each source event file.

The program can additionally generate a self-contained interactive:

cluster_report.html

The HTML report can be viewed in a browser or printed to PDF.

Installation
1. Clone the repository
git clone https://github.com/SubratBastola/NanoporeTagging.git
cd NanoporeTagging
2. Create a Python environment

Using Conda:

conda create -n nanoporetagging python=3.11
conda activate nanoporetagging

or using venv:

python -m venv .venv

Windows:

.venv\Scripts\activate

macOS/Linux:

source .venv/bin/activate
3. Install dependencies
pip install numpy pandas scipy matplotlib scikit-learn torch pyabf openpyxl xlrd plotly dash

The graphical desktop components also require Tkinter, which is included with most standard Windows Python installations.

Typical Workflow
1. Identify candidate events

Run:

python Tagger_GUI.py

Open an ABF recording and define event windows.

Export the annotations to CSV.

2. Automatically detect landmarks

Place the event files and corresponding ABF recordings in the same working folder.

Run:

python NeuralNetwork.py

Select:

best_opt_strict.pt

or another compatible trained checkpoint.

The resulting file will contain automatically estimated optical/electrical landmarks and derived event features.

3. Cluster detected events

Run:

python clustering.py

Select one or more predicted/event files.

Choose the desired clustering features and parameters, then run the GMM analysis.

Inspect the resulting clusters and export the results or HTML report.

Example Data Flow
experiment.abf
      │
      │ Tagger_GUI.py
      ▼
experiment_event.csv
      │
      │ NeuralNetwork.py
      │ + best_opt_strict.pt
      ▼
experiment_event__predicted.xlsx
      │
      │ clustering.py
      ▼
clustering_results.xlsx
cluster_report.html
Main Output Features

A processed event can contain measurements such as:

Feature	Description
event_start (s)	Estimated optical event onset
event_end (s)	Estimated optical event termination
event_plateau	Plateau landmark
duration	Event dwell time
Base (V)	Optical baseline
Step (V)	Optical event level
RefBase (V)	Reference optical baseline
RefStep (V)	Reference optical event level
OSC	Optical step change
RefOSC	Reference optical step change
entry Base (pA)	Electrical baseline before entry
entry Peak (pA)	Entry current peak
exit Peak (pA)	Exit current peak
exit Base (pA)	Electrical baseline after exit
entry spike	Entry peak-to-baseline amplitude
exit spike	Exit peak-to-baseline amplitude
Notes
The automated model was designed for the signal/channel organization used during development. Verify the channel mapping before processing data acquired with a different configuration.
The .pt checkpoint must be compatible with the StrongOpt1D architecture defined in NeuralNetwork.py.
Event windows should contain the complete event and enough surrounding baseline for reliable landmark detection.
Automated predictions should be visually inspected when applying the model to data substantially different from the training data.
Temporary _tmp_npz files are intentionally retained so interrupted runs can resume efficiently.
License

No license has currently been specified for this repository.

If the project will be distributed publicly, add an appropriate LICENSE file.

Author

Subrat Bastola

Biomedical signal processing, machine learning, and single-molecule nanopore data analysis.

GitHub: SubratBastola
