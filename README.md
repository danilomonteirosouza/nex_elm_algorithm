# NEX-ELM v69

Reproducible implementation of **Native Explainability for Extreme Learning Machines (NEX-ELM)**, a model-specific glocal explainability framework for ensembles of **Extreme Learning Machines (ELMs)**.

NEX-ELM combines:

- a native local attribution mechanism, **Interventional Robust Prefix for NEX-ELM (IRP-NEX)**;
- registered-set explanation signatures;
- class-conditional libraries of medoid prototypes learned only from calibration data;
- prototype-based routing that connects local, regional, and global explanations;
- expanded comparisons with established local, global, and glocal explainability methods.

The main executable file is:

```text
nexlm.py
```

Running the script without command-line arguments executes the complete CUDA study plan reported for v69:

```text
8 source datasets
9 experimental scenarios
30 independent seeds per scenario
270 scenario-seed experimental runs
```

The Electrical Grid Stability dataset is evaluated in two scenarios: one retaining `stab` and one excluding it. The default workflow produces predictive evaluations, local, global, and glocal explanation analyses, statistical tests, runtime measurements, CPU--CUDA numerical audits, consolidated tables, figures, and `relatorio.pdf`.

---

## 1. Terminology

### 1.1 NEX-ELM

**NEX-ELM** denotes **Native Explainability for Extreme Learning Machines**. The name emphasizes that the proposed explanation mechanism uses the internal structure and outputs of the trained ELM ensemble rather than relying only on an external surrogate.

### 1.2 IRP-NEX

**IRP-NEX** denotes **Interventional Robust Prefix for NEX-ELM**. It is the local component of NEX-ELM. IRP-NEX constructs deterministic feature orderings under an empirical interventional game and represents each explanation through registered feature prefixes.

### 1.3 Glocal explanation

A glocal explanation connects three complementary levels:

- **Local:** the feature-attribution pattern associated with one prediction.
- **Regional:** a representative explanatory pattern shared by a subset of observations.
- **Global:** class-conditional and dataset-level summaries derived from the collection of explanatory prototypes.

NEX-ELM preserves the local explanation of each observation while routing it to a reusable class-specific prototype.

---

## 2. Method Overview

NEX-ELM is designed for ELM ensembles whose predictions are distributed across randomized hidden representations, nonlinear activations, output weights, and multiple ensemble members. A single coefficient or a single global ranking is therefore insufficient to describe how the original input features influence all observations.

The method addresses this problem by deriving native local attribution signatures and organizing recurrent signatures into finite class-conditional prototype libraries.

### 2.1 Core Components

1. **Predictive ELM ensemble**  
   An ensemble of ELM estimators is fitted using uniform estimator weights. Each estimator contains a randomized hidden layer and obtains its output weights through Ridge-regularized linear optimization.

2. **IRP-NEX local attribution**  
   For each observation, IRP-NEX explains the probability assigned to the class predicted by the ensemble. The target class remains fixed throughout the perturbation trajectory, including cases in which a perturbed observation crosses a decision boundary.

3. **Empirical interventional game**  
   Feature contributions are evaluated relative to an empirical background distribution. The solver searches for a complete feature-insertion order that reconstructs the model output associated with the fixed target class.

4. **Registered-set explanation signature**  
   The complete feature order is converted into registered prefixes. Two explanations can then be compared through a distance defined over their registered prefix sets.

5. **Class-conditional prototype library**  
   Explanation signatures computed exclusively from the calibration partition are grouped by ensemble-predicted class. Each class is represented by at most four medoids obtained through deterministic k-medoids clustering.

6. **Glocal inference routing**  
   For an unseen observation, the ensemble-predicted class selects the relevant prototype library. The observation's IRP-NEX signature is routed to the nearest fixed medoid within that class.

7. **Regional and global summaries**  
   The routed prototype provides the regional explanation. Weighted prototype collections are used to derive class-conditional and dataset-level analytical summaries.

The final explanatory output contains:

- an observation-specific IRP-NEX explanation;
- a representative class-specific prototype;
- a regional explanation associated with that prototype;
- class-conditional and dataset-level global summaries.

---

## 3. Fitting, Calibration, and External Test Separation

The protocol prevents the external test partition from influencing model fitting, prototype construction, or method selection.

1. The complete dataset is divided into **80% outer training data** and **20% external test data**.
2. The outer training partition is divided again using `calibration_fraction = 0.30`.
3. Relative to the complete dataset, the approximate allocation is:

| Partition | Approximate proportion | Purpose |
|---|---:|---|
| Ensemble fitting | 56% | Fits the ELM estimators |
| Calibration | 24% | Builds explanations, prototypes, routing structures, and auxiliary calibration quantities |
| External test | 20% | Evaluates prediction, fidelity, routing, runtime, and numerical behavior |

The external test partition is not used to:

- fit the ELM ensemble;
- select the number of prototypes;
- estimate prototype medoids;
- tune the registered-set distance;
- choose a prototype using the ground-truth label;
- optimize explanation fidelity on test observations;
- alter the frozen NEX-ELM mathematical core.

At inference time, routing uses the **ensemble-predicted class**, not the ground-truth test label.

---

## 4. Compared Methods

### 4.1 Local Explanation Methods

| Method | Role in the study |
|---|---|
| NEX-ELM / IRP-NEX | Proposed native local attribution method |
| Kernel SHAP | Model-agnostic Shapley-value approximation |
| LIME | Model-agnostic local surrogate |
| Integrated Gradients | Path-based gradient attribution |
| Gradient SHAP | Expected-gradient attribution with sampled baselines |
| Random ranking | Auxiliary fidelity control used by the frozen core |

### 4.2 Global Explanation Methods

| Method | Role in the study |
|---|---|
| NEX-ELM | Prototype-conditioned analytical global summary |
| Aggregated Kernel SHAP | Aggregation of local absolute SHAP values |
| X-ELM | ELM-specific global importance baseline |
| Permutation Feature Importance (PFI) | Global predictive-performance perturbation baseline |
| SAGE | Global subset-based feature-importance baseline |

### 4.3 Glocal Explanation Methods

| Method | Role in the study |
|---|---|
| NEX-ELM | Class-conditional prototype-conditioned glocal framework |
| R-LOCO | Regional LOCO-based glocal comparator |

The current implementation evaluates a **K-Means-based R-LOCO configuration** with four regions. It should therefore be described as an implementation of the R-LOCO framework rather than as an exact reproduction of every default choice in the original R-LOCO article.

The R-LOCO evaluation is supplementary to the original primary confirmatory family. It does not modify the frozen NEX-ELM mathematical core.

### 4.4 Runtime Comparisons

The principal runtime claims distinguish different scopes:

- local NEX-ELM versus local Kernel SHAP;
- the complete NEX-ELM workflow versus the combined X-ELM plus Kernel SHAP workflow;
- supplementary timing records for LIME, Integrated Gradients, Gradient SHAP, PFI, SAGE, and R-LOCO.

The study does **not** claim that isolated NEX-ELM is faster than isolated X-ELM.

---

## 5. Explanation Fidelity and Diagnostics

### 5.1 Area Under the Curve Metrics

Explanation fidelity is evaluated through perturbation trajectories.

- **Deletion Area Under the Curve (AUC):** evaluates the model-output change when the highest-ranked features are removed first.
- **Insertion Area Under the Curve (AUC):** evaluates the recovery of the target model output when the highest-ranked features are inserted first.
- **Composite Area Under the Curve (AUC):** combines the registered deletion and insertion evidence used in the global and glocal analyses.

Under the orientation implemented in this project, larger values indicate higher explanation fidelity.

### 5.2 Additional Diagnostics

The workflow also evaluates:

- local and global agreement;
- numerical completeness;
- prototype usage and routing;
- functional stability across seeds;
- literal prototype-signature repetition;
- class-conditional and dataset-level importance;
- CPU--CUDA numerical consistency;
- runtime and GPU-memory behavior.

Agreement with SHAP is treated as a structural diagnostic, not as ground truth. The principal fidelity evidence comes from deletion, insertion, and composite AUC trajectories.

---

## 6. Statistical Analysis

The analysis suite includes:

- Shapiro--Wilk normality diagnostics;
- one-sided paired *t* tests;
- one-sided Wilcoxon signed-rank tests;
- paired permutation tests;
- repeated-measures ANOVA;
- Friedman tests;
- Kendall's coefficient of concordance (*W*);
- Nemenyi post hoc comparisons;
- exact binomial tests for seed-level win rates;
- exact Clopper--Pearson confidence intervals;
- Holm correction for multiple testing;
- standardized paired effect sizes.

The confirmatory rule considers effect direction, confidence intervals, paired tests, effect size, multiplicity correction, and consistency across independent seeds.

The original primary family and the later expanded benchmark are reported separately:

- **Primary confirmatory family:** the frozen pre-specified comparisons.
- **Supplementary expanded benchmark:** additional local, global, and glocal methods added without changing NEX-ELM.

---

## 7. Datasets and Experimental Scenarios

The complete study contains **eight source datasets evaluated under nine scenarios**.

| Scenario identifier | Observations | Features | Classes | Source or experimental role |
|---|---:|---:|---:|---|
| `electrical_grid_stability` | 10,000 | 13 | 2 | UCI Electrical Grid Stability retaining `stab` to reproduce the X-ELM reference setting |
| `electrical_grid_stability_without_stab` | 10,000 | 12 | 2 | Controlled proxy-removal scenario excluding `stab` |
| `pima_indians_diabetes` | 768 | 8 | 2 | Pima Indians Diabetes |
| `wisconsin_breast_cancer_original` | 683 | 9 | 2 | Wisconsin Breast Cancer Original after cleaning |
| `ionosphere_binary` | 351 | 34 | 2 | UCI Ionosphere |
| `wine_multiclass` | 178 | 13 | 3 | scikit-learn Wine |
| `iris_multiclass` | 150 | 4 | 3 | scikit-learn Iris |
| `digits_multiclass` | 1,797 | 64 | 10 | scikit-learn Digits |
| `breast_cancer_diagnostic` | 569 | 30 | 2 | scikit-learn Breast Cancer Wisconsin Diagnostic |

The two Electrical Grid rows correspond to two scenarios derived from the same source dataset. This is why the study reports eight datasets but nine experimental scenarios.

The scikit-learn datasets are loaded from the installed library. Electrical Grid Stability, Pima Indians Diabetes, Wisconsin Breast Cancer Original, and Ionosphere may be downloaded by the script into `reference_data`.

---

## 8. Reported v69 Results

The following statements summarize the completed 30-seed v69 experiment. They are descriptive of the evaluated scenarios and must not be interpreted as universal claims.

### 8.1 Predictive Performance

- Mean balanced accuracy across scenarios ranged from approximately **0.7254 to 0.9766**.
- Predictive metrics were computed on the complete external test partition.

### 8.2 Primary Confirmatory Family

- **50 of 54** primary hypotheses were confirmed under the pre-specified criterion.
- NEX-ELM achieved higher local deletion and insertion AUC than Kernel SHAP in all **18 of 18** primary local comparisons.
- NEX-ELM achieved higher composite global AUC than X-ELM in all **9 of 9** primary global comparisons.
- Against aggregated Kernel SHAP, **7 of 9** primary global comparisons satisfied the full confirmatory rule.

### 8.3 Expanded Local Benchmark

Across the evaluated local contrasts, NEX-ELM obtained higher deletion and insertion AUC than:

- Kernel SHAP;
- LIME;
- Integrated Gradients;
- Gradient SHAP.

The expanded local benchmark produced positive mean differences in all evaluated contrasts, with paired and seed-level statistical support.

### 8.4 Expanded Global Benchmark

- NEX-ELM exceeded X-ELM across all evaluated composite global comparisons.
- NEX-ELM exceeded PFI across the evaluated composite, deletion, and insertion contrasts.
- NEX-ELM obtained higher mean global AUC than SAGE across the evaluated scenarios, although not every SAGE contrast satisfied every inferential criterion in the high-dimensional multiclass setting.
- Comparisons with aggregated Kernel SHAP were favorable overall but were not uniformly confirmed by every statistical rule.

### 8.5 Expanded Glocal Benchmark

Against the evaluated K-Means-based R-LOCO configuration:

- NEX-ELM achieved higher composite, deletion, and insertion AUC in all **27 of 27** glocal contrasts;
- each contrast produced **30 wins across 30 seeds**;
- confidence intervals for the paired differences remained positive;
- paired *t* tests, exact binomial tests, repeated-measures ANOVA, and recalculated Wilcoxon tests supported the observed direction after Holm correction;
- all standardized paired effects were large in the evaluated experiment.

Because R-LOCO was added after the original primary family, these results are reported as **supplementary glocal evidence**.

### 8.6 Runtime

- The complete NEX-ELM workflow was faster than the combined X-ELM plus Kernel SHAP workflow in **8 of 9** scenarios.
- The mean workflow speedup was **3.38x**.
- The high-dimensional, ten-class Digits scenario was the computational exception.
- NEX-ELM is not claimed to be the fastest method among all local baselines; LIME was faster in the expanded timing comparison.

### 8.7 CUDA Audit

- The CPU--CUDA numerical audit passed for all **270** scenario-seed runs within the registered tolerance.
- This result establishes agreement for the evaluated implementation, hardware, inputs, and tolerance. It is not a universal proof of identical behavior across all systems.

### 8.8 Prototype Stability

Prototype stability must be interpreted at two levels:

- **Functional stability:** similarity of importance patterns, routing behavior, and fidelity across seeds.
- **Literal stability:** exact repetition of the same prototype signature across seeds.

The experiment supports functional stability, whereas exact signature repetition can be low because independent ELM seeds generate distinct hidden representations.

---

## 9. Effective Configuration

### 9.1 Experimental Protocol and Predictive Model

| Component | Effective configuration |
|---|---|
| Software version | `v69` |
| Packaging revision | `v69-expanded-xai-benchmarks-frozen-v68-nex-core` |
| Study plan | `complete` |
| Data mode | `real` |
| Source datasets | 8 |
| Experimental scenarios | 9 |
| Independent repetitions per scenario | 30 |
| Scenario-seed runs | 270 |
| Initial random seed | 100000 |
| Seed increment | 1009 |
| Final random seed | 129261 |
| External test proportion | 20% |
| Calibration fraction within outer training data | 30% |
| ELM estimators | 400 |
| Hidden neurons per ELM | 50 |
| Hidden-layer activation | `tanh` |
| Ridge regularization | `1e-2` |
| Estimator weighting | Uniform |
| Maximum NEX prototypes per predicted class | 4 |
| Minimum calibration rows per prototype slot | 2 |
| NEX prototype clustering | Deterministic k-medoids |
| R-LOCO regional partition | K-Means-based, 4 regions in the reported run |
| Prototype-library organization | Conditioned on ensemble-predicted class |
| Registered minimum seed-level win rate | 0.80 |
| Minimum wins over 30 seeds | 24 |
| Significance level | 0.05 |
| Registered minimum effect size | 0.50 |

### 9.2 Frozen-Core Explanation Parameters

| Parameter | Effective value |
|---|---:|
| Ablation protocol | `expected_background` |
| Kernel SHAP background size | 350 |
| Local observations explained by Kernel SHAP | 50 |
| `shap_nsamples` | Delegated to SHAP |
| NEX probabilistic nodes | 16 |
| Convergence-analysis nodes | `8,16,32` |
| Convergence-analysis observations | 2 |
| Global calibration explanations | 96 |
| Global bootstrap iterations | 128 |
| Tuning observations | 30 |
| Interaction candidates | 5 |
| Interaction grid | `0,0.25,0.50,0.75` |
| Candidate-pool grid | `50` |
| Redundancy grid | `0,0.05,0.10,0.20` |
| Deletion weight in calibration | 0.50 |
| Global external-test observations | 100 |
| Random-fidelity control repetitions | 5 |
| Inferential bootstrap iterations | 5,000 |
| Permutation-test iterations | 100,000 |
| Non-inferiority margin | 0.01 |

### 9.3 Expanded-Baseline Controls

The v69 wrapper exposes the following controls:

```text
--expanded-baselines {full,fast,off}
--lime-samples INTEGER
--integrated-gradients-nodes INTEGER
--gradient-shap-samples INTEGER
--permutation-importance-repeats INTEGER
--sage-permutations INTEGER
--sage-evaluation-rows INTEGER
--rloco-regions INTEGER
--rloco-max-features INTEGER
```

The exact resolved values are written to the run protocol and configuration artifacts. The `--quick` option and `--expanded-baselines fast` reduce supplementary baseline workloads without changing the frozen NEX-ELM mathematical core.

### 9.4 CUDA and Runtime Configuration

| Component | Effective configuration |
|---|---|
| Default device | `cuda` |
| GPU profile | `rtx3060_12gb` |
| Numerical precision | `float32` |
| TF32 | Enabled |
| CUDA allocator | `cudaMallocAsync` |
| GPU prediction batch size | 32,768 |
| Estimator batch size | 256 |
| NEX instance batch size | 4 |
| NEX background block size | 256 |
| NEX gradient batch size | 16,384 |
| Ablation-task batch size | 1,024 |
| Minimum batch size after an out-of-memory event | 1,024 |
| CUDA warm-up rows | 512 |
| CUDA warm-up | Enabled |
| `torch.compile` | Disabled |
| CPU--CUDA numerical audit | Enabled |
| Numerical-audit tolerance | `2e-4` |

### 9.5 Reported Computing Environment

The reported run used:

| Resource | Configuration |
|---|---|
| Processor | Intel Core i9-10850K, 10 physical cores and 20 logical threads |
| System memory | 64 GB DDR4 |
| GPU | NVIDIA GeForce RTX 3060, 12 GB GDDR6, Compute Capability 8.6 |
| Storage | NVMe SSD |
| Python | 3.10 |
| PyTorch | 2.12.1 with CUDA 12.6 |

For strict reproduction, preserve the generated protocol JSON and record the active environment:

```bash
pip freeze > requirements-lock.txt
```

---

## 10. Software Requirements and Installation

### 10.1 Required Software

The implementation requires:

- Python 3.10 or 3.11;
- NumPy;
- pandas;
- SciPy;
- Matplotlib;
- scikit-learn;
- SHAP;
- PyTorch;
- ReportLab, when `relatorio.pdf` is required.

A CUDA-capable NVIDIA GPU is required for the default `complete` study plan. A reduced custom audit can be executed on a CPU.

### 10.2 Create a Virtual Environment

#### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

#### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

### 10.3 Install Dependencies

Install a PyTorch build compatible with the available GPU and CUDA runtime first. Then install the remaining packages:

```bash
pip install numpy pandas scipy matplotlib scikit-learn shap reportlab
```

For a CPU-only audit environment:

```bash
pip install torch numpy pandas scipy matplotlib scikit-learn shap reportlab
```

When `requirements.txt` is available:

```bash
pip install -r requirements.txt
```

Verify the environment:

```bash
python -c "import torch, shap, sklearn, numpy, pandas, scipy; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 11. Recommended Project Layout

```text
nex_elm_project/
├── nexlm_v69_expanded_xai_benchmarks.py
├── README.md
├── requirements.txt
├── requirements-lock.txt             # recommended for strict reproduction
├── reference_data/
└── results/
```

The integrity-checked mathematical core is embedded in the main script. An adjacent file named:

```text
nexlm_v68_glopro_complete_core.py
```

is optional and is loaded only when its SHA-256 digest matches:

```text
fc78619795db892180c7b978366a67b15a48741103f0fcc98eeab7b7dcf2031f
```

The v68 identifier in the optional core filename is intentional: v69 adds external local, global, and glocal baselines while retaining the frozen NEX-ELM mathematical core. A non-matching adjacent core is ignored.

---

## 12. Execution Instructions

### 12.1 Display Command-Line Help

```bash
python nexlm_v69_expanded_xai_benchmarks.py --help
```

### 12.2 Reproduce the Complete Experiment

Running without arguments selects CUDA, the RTX 3060 12 GB profile, 30 repetitions, all nine scenarios, expanded baselines, and PDF generation:

```bash
python nexlm_v69_expanded_xai_benchmarks.py
```

Specify the output directory:

```bash
python nexlm_v69_expanded_xai_benchmarks.py \
  --output-dir nex_elm_v69_results
```

PowerShell:

```powershell
python nexlm_v69_expanded_xai_benchmarks.py --output-dir nex_elm_v69_results
```

> The default `complete` plan requires CUDA. The script raises an error when the complete plan resolves to a CPU device.

### 12.3 Run a GPU Smoke Test

```bash
python nexlm_v69_expanded_xai_benchmarks.py \
  --quick \
  --output-dir nex_elm_v69_smoke_test
```

The `--quick` option reduces each phase to one repetition and reduces expanded-baseline workloads.

### 12.4 Run a Reduced CPU Audit

```bash
python nexlm_v69_expanded_xai_benchmarks.py \
  --study-plan custom \
  --datasets iris \
  --n-repeats 1 \
  --device cpu \
  --expanded-baselines fast \
  --skip-pdf-report \
  --output-dir nex_elm_v69_cpu_audit
```

The alias `iris` resolves to `iris_multiclass`.

### 12.5 Run Selected Scenarios

```bash
python nexlm_v69_expanded_xai_benchmarks.py \
  --study-plan custom \
  --datasets iris,wine,breast_diagnostic \
  --n-repeats 5 \
  --device cuda \
  --expanded-baselines full \
  --output-dir nex_elm_v69_custom_results
```

Accepted aliases include:

| Alias | Canonical scenario identifier |
|---|---|
| `grid` | `electrical_grid_stability` |
| `grid_without_stab` or `grid_no_stab` | `electrical_grid_stability_without_stab` |
| `pima` | `pima_indians_diabetes` |
| `wisconsin` | `wisconsin_breast_cancer_original` |
| `ionosphere` | `ionosphere_binary` |
| `wine` | `wine_multiclass` |
| `iris` | `iris_multiclass` |
| `digits` | `digits_multiclass` |
| `breast_diagnostic` | `breast_cancer_diagnostic` |

### 12.6 Run Only the Replication Phase

```bash
python nexlm_v69_expanded_xai_benchmarks.py \
  --study-plan replication \
  --replication-repeats 30 \
  --replication-random-state 100000 \
  --output-dir nex_elm_v69_replication_results
```

The default replication phase includes:

- Electrical Grid Stability with `stab`;
- Pima Indians Diabetes;
- Wisconsin Breast Cancer Original.

### 12.7 Run Only the Generalization Phase

```bash
python nexlm_v69_expanded_xai_benchmarks.py \
  --study-plan generalization \
  --generalization-repeats 30 \
  --generalization-random-state 100000 \
  --output-dir nex_elm_v69_generalization_results
```

### 12.8 Run the Journal Plan

```bash
python nexlm_v69_expanded_xai_benchmarks.py \
  --study-plan journal \
  --output-dir nex_elm_v69_journal_plan
```

The `journal` plan retains the preceding reduced scenario plan, whereas `complete` includes the complete nine-scenario benchmark.

### 12.9 Select the Expanded-Baseline Workload

Full benchmark:

```bash
python nexlm_v69_expanded_xai_benchmarks.py --expanded-baselines full
```

Reduced supplementary benchmark:

```bash
python nexlm_v69_expanded_xai_benchmarks.py --expanded-baselines fast
```

Disable the supplementary v69 comparators while retaining the frozen core:

```bash
python nexlm_v69_expanded_xai_benchmarks.py --expanded-baselines off
```

### 12.10 Configure R-LOCO Regions

```bash
python nexlm_v69_expanded_xai_benchmarks.py \
  --rloco-regions 4 \
  --output-dir nex_elm_v69_rloco_k4
```

Changing the number of R-LOCO regions does not alter NEX-ELM's frozen prototype count.

### 12.11 Use Local Dataset Files Without Downloading

```bash
python nexlm_v69_expanded_xai_benchmarks.py \
  --data-dir reference_data \
  --no-download
```

Recognized filenames include:

```text
reference_data/
├── Data_for_UCI_named.csv
├── pima-indians-diabetes.data.csv
├── breast-cancer-wisconsin.data
└── ionosphere.data
```

### 12.12 Use the Conservative GPU Profile

```bash
python nexlm_v69_expanded_xai_benchmarks.py \
  --gpu-profile conservative \
  --output-dir nex_elm_v69_conservative_gpu_results
```

The GPU profile changes execution and batching parameters only. It does not change the frozen NEX-ELM mathematics.

### 12.13 Recompute Statistical Analyses Without Rerunning Models

```bash
python nexlm_v69_expanded_xai_benchmarks.py \
  --statistics-only-from nex_elm_v69_results \
  --statistics-output-dir nex_elm_v69_statistical_reanalysis
```

The source directory must contain `seed_metrics.csv` and, when available, `estatistica_entre_seeds.csv`, either directly or under `combined/tabelas`.

### 12.14 Regenerate Only the PDF Report

```bash
python nexlm_v69_expanded_xai_benchmarks.py \
  --report-only-from nex_elm_v69_results \
  --report-output nex_elm_v69_report.pdf
```

Skip PDF generation during a model run:

```bash
python nexlm_v69_expanded_xai_benchmarks.py --skip-pdf-report
```

---

## 13. Output Structure

A standard run produces a structure similar to:

```text
resultados_v69_expanded_xai_YYYYMMDD_HHMMSS/
├── plano_confirmatorio.csv
├── registro_protocolo_confirmatorio.json
├── relatorio.pdf
├── per_seed/
│   └── <dataset>/<scenario>/seed_<value>/
│       ├── manifest.json
│       └── tabelas/
└── combined/
    ├── tabelas/
    ├── graficos_estatisticos/
    ├── graficos_preditivos/
    └── additional consolidated artifacts/
```

Some generated filenames remain in Portuguese because they are part of the implementation contract.

### 13.1 Predictive Outputs

| File | Content |
|---|---|
| `predictive_performance_per_seed.csv` | Predictive metrics for each seed |
| `predictive_performance_summary.csv` | Across-seed means and confidence intervals |
| `predictive_performance_article_table.csv` | Manuscript-ready predictive table |
| `predictive_confusion_matrix.csv` | Complete confusion matrices |
| `predictive_confusion_matrix_summary.csv` | Across-seed confusion-matrix summary |
| `predictive_class_metrics.csv` | Per-class predictive metrics |
| `predictive_class_metrics_summary.csv` | Across-seed per-class summary |
| `predictive_dataset_summary.csv` | Dataset dimensions and task characteristics |

### 13.2 NEX-ELM and Prototype Outputs

| File | Content |
|---|---|
| `local_fidelity_summary.csv` | Local fidelity by method and seed |
| `global_fidelity_summary.csv` | Global fidelity by method and seed |
| `local_agreement.csv` | Local structural agreement diagnostics |
| `global_importance.csv` | Dataset-level global importance |
| `global_class_importance.csv` | Class-conditional global importance |
| `prototype_library.csv` | Complete explanatory prototype library |
| `prototype_library_class_summary.csv` | Prototype summary by predicted class |
| `prototype_routing.csv` | Observation-to-prototype routing |
| `prototype_routing_summary.csv` | Routing summary |
| `prototype_stability_between_seeds.csv` | Across-seed prototype diagnostics |

### 13.3 Expanded v69 Baseline Outputs

The v69 wrapper adds consolidated tables for:

- LIME local attributions and fidelity;
- Integrated Gradients attributions, fidelity, and completeness;
- Gradient SHAP attributions and fidelity;
- PFI global importance and fidelity;
- SAGE global importance and fidelity;
- R-LOCO regional importance, routing, fidelity, and runtime;
- expanded local, global, and glocal seed metrics;
- supplementary statistical comparisons and method ranks.

The exact filenames are recorded in the generated manifest and are consolidated under `combined/tabelas`.

### 13.4 Statistical, Runtime, and Audit Outputs

| File | Content |
|---|---|
| `seed_metrics.csv` | Central across-seed inferential dataset |
| `inferential_test_summary.csv` | Summary of confirmatory tests |
| `paired_t_tests.csv` | Paired *t*-test results |
| `shapiro_wilk_tests.csv` | Normality diagnostics |
| `repeated_measures_anova.csv` | Repeated-measures ANOVA |
| `friedman_tests.csv` | Friedman-test results |
| `average_method_ranks.csv` | Average ranks of compared methods |
| `nemenyi_posthoc.csv` | Nemenyi post hoc comparisons |
| `binomial_win_rate_tests.csv` | Exact binomial tests and Clopper--Pearson intervals |
| `multiple_testing_families.csv` | Multiple-testing families and Holm corrections |
| `timing.csv` | Local and workflow runtime measurements |
| `cuda_audit.csv` | CPU--CUDA numerical-equivalence audit |
| `gpu_runtime.csv` | GPU device, runtime, and memory information |

---

## 14. Integrity and Reproducibility Controls

Before the experimental workflow begins, the script:

1. loads the embedded mathematical core or a matching adjacent core;
2. verifies the mathematical-core SHA-256 digest;
3. runs internal self-tests;
4. confirms the registered-set and routing wrappers;
5. fixes the NEX prototype configuration at `K = 4`;
6. records the public protocol and resolved configuration;
7. checks independence from the preceding confirmatory seed battery;
8. records datasets, scenarios, seeds, methods, and claim boundaries;
9. confirms that external baselines do not alter NEX-ELM mathematics;
10. audits the availability of required report and numerical packages.

The NEX prototype configuration is frozen. These commands intentionally raise an error:

```bash
python nexlm_v69_expanded_xai_benchmarks.py --prototype-count 3
python nexlm_v69_expanded_xai_benchmarks.py --prototype-min-calibration-rows-per-slot 4
```

The accepted values are:

```text
prototype-count = 4
prototype-min-calibration-rows-per-slot = 2
```

The second value is a prototype-capacity parameter. It is not a post-clustering minimum cluster-size constraint.

The R-LOCO region count is an external baseline parameter and is controlled separately through `--rloco-regions`.

---

## 15. Troubleshooting

### `The shap package is required`

```bash
pip install shap
```

### `The torch package is required for NEX-ELM`

Install a PyTorch build compatible with the intended CPU or CUDA environment.

### `The default complete experiment requires CUDA`

Use a reduced custom CPU audit:

```bash
python nexlm_v69_expanded_xai_benchmarks.py \
  --study-plan custom \
  --datasets iris \
  --n-repeats 1 \
  --device cpu \
  --expanded-baselines fast
```

### CUDA Out-of-Memory Error

Start with the conservative profile:

```bash
python nexlm_v69_expanded_xai_benchmarks.py --gpu-profile conservative
```

A custom profile may reduce the principal batch sizes:

```bash
python nexlm_v69_expanded_xai_benchmarks.py \
  --gpu-profile custom \
  --gpu-batch-size 8192 \
  --estimator-batch-size 32 \
  --nex-instance-batch-size 1 \
  --nex-background-block-size 64
```

The supplementary baselines can also be reduced:

```bash
python nexlm_v69_expanded_xai_benchmarks.py --expanded-baselines fast
```

### Dataset File Not Found with `--no-download`

Check the `--data-dir` path and the recognized filenames listed in Section 12.11.

### PDF Report Was Not Generated

```bash
pip install reportlab
```

Or skip report generation:

```bash
python nexlm_v69_expanded_xai_benchmarks.py --skip-pdf-report
```

### Statistical Reanalysis Cannot Find Source Tables

Confirm that the source contains:

```text
seed_metrics.csv
estatistica_entre_seeds.csv    # when generated
```

These files may be at the experiment root or under `combined/tabelas`.

### Results Differ Across Machines

Small numerical differences may arise from:

- package versions;
- CUDA and cuDNN versions;
- TF32 behavior;
- GPU architecture;
- CPU threading;
- parallel scheduling;
- floating-point reduction order.

Record the complete environment:

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
pip freeze > requirements-lock.txt
```

For strict replication, preserve the script, input datasets, package lock, GPU model, CUDA runtime, protocol JSON, seed registry, and mathematical-core digest.

---

## 16. Scope and Interpretation Boundaries

The generated evidence supports claims concerning:

- predictive performance on the evaluated tabular datasets;
- explanation fidelity under the registered empirical interventional game;
- local, global, and glocal explanatory granularity;
- class-conditional prototype organization;
- consistency across the evaluated independent seeds;
- runtime in the audited hardware and software environment;
- CPU--CUDA numerical agreement within the registered tolerance.

The experiments do not, by themselves, establish:

- causal feature effects;
- algorithmic fairness;
- clinical or operational usefulness to human end users;
- superiority over every available XAI method;
- superiority over every R-LOCO configuration;
- runtime superiority over isolated X-ELM;
- that NEX-ELM is the fastest local method;
- universal scalability to arbitrarily high-dimensional or multiclass data;
- universal absence of GPU memory leaks;
- automatic transfer to images, text, time series, or non-ELM predictors.

The high-dimensional multiclass scenario exposed an important computational limitation. Scalability should therefore be investigated as both feature dimensionality and class cardinality increase.

---

## 17. Citation

When using this implementation in academic work, cite the associated NEX-ELM manuscript. Replace the provisional record with the final metadata and DOI after publication.

```bibtex
@article{nexelm,
  title   = {NEX-ELM: Advancing Glocal Explainable AI for Extreme Learning Machines through Class-Conditional Prototype Libraries},
  author  = {Author information omitted for double-blind review},
  journal = {Neural Computing and Applications},
  year    = {forthcoming}
}
```

---

## 18. License

No license file was included with the supplied implementation. Before public redistribution, add a license compatible with the authors' intended terms, the third-party dependencies, and the licenses or usage conditions of the datasets.
