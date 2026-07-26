# NEX-ELM

Reproducible implementation of **NEX-ELM**, a model-specific glocal explainability method for ensembles of **Extreme Learning Machines (ELMs)**. The method combines intervention-based local explanations with a finite library of explanatory prototypes conditioned on the ensemble-predicted class.

The main executable file is:

```text
nexlm
```

Running the script without command-line arguments reproduces the complete experimental protocol reported in the associated manuscript: **9 dataset configurations × 30 independent random seeds = 270 primary experimental runs**. The default workflow uses CUDA and generates predictive evaluations, local and global explanation analyses, CPU–GPU numerical audits, confirmatory statistical tests, consolidated tables, and a PDF report.

---

## 1. Method Overview

NEX-ELM was designed to connect two complementary levels of explanation:

- **Local explanation:** why the ELM ensemble produced a particular prediction for an individual observation.
- **Global explanation:** which recurrent explanatory patterns characterize the model within a predicted class or across the complete dataset.

Rather than reducing the model to a single global feature ranking, NEX-ELM preserves observation-specific explanations and organizes recurrent local patterns into a compact class-conditional prototype library. The resulting representation is therefore **glocal**: each test observation retains its own explanation while being linked to a recurrent explanatory pattern learned from calibration data.

### 1.1 Core Components

1. **Predictive ELM ensemble**  
   An ensemble of ELM estimators is fitted using uniform estimator weights. Each estimator uses a randomly generated hidden layer and obtains its output weights through Ridge-regularized linear optimization.

2. **Local IRP-NEX explanation**  
   For each observation, the method explains the probability assigned to the class predicted by the ensemble. The target class remains fixed throughout the perturbation trajectory, even when a perturbed observation crosses a decision boundary.

3. **Empirical interventional game**  
   Feature contributions are evaluated with respect to an empirical background distribution. The local solver searches for a complete feature-insertion order that faithfully reconstructs the model output associated with the fixed target class.

4. **Registered-set explanation signature**  
   The complete local feature order is transformed into a signature defined by registered prefixes. This representation enables two complete explanations to be compared using a distance defined over their prefix sets.

5. **Class-conditional prototype library**  
   Signatures computed exclusively from the calibration partition are grouped by ensemble-predicted class. Each class is represented by at most four medoids obtained through deterministic k-medoids clustering.

6. **Glocal inference routing**  
   At inference time, the ensemble-predicted class selects the candidate prototype library. The local explanation signature of an unseen observation is then routed to the nearest fixed prototype within that class-conditional library.

The final explanatory output comprises:

- an observation-specific local explanation;
- a representative prototype associated with the local explanatory pattern;
- a class-conditional and dataset-level global representation obtained from the weighted collection of prototypes.

---

## 2. Fitting, Calibration, and External Test Separation

The experimental protocol prevents the external test partition from influencing model fitting or explanation-library construction.

1. The complete dataset is divided into **80% outer training data** and **20% external test data**.
2. The outer training partition is divided again using `calibration_fraction = 0.30`.
3. Relative to the complete dataset, the resulting approximate allocation is:

| Partition | Approximate proportion | Purpose |
|---|---:|---|
| Ensemble fitting | 56% | Fitting the ELM estimators |
| Calibration | 24% | Calibration explanations, deterministic k-medoids, prototype construction, and auxiliary parameters |
| External test | 20% | Predictive evaluation, explanation fidelity, routing evaluation, and runtime analysis |

The external test partition is not used to:

- fit the ELM ensemble;
- select the number of prototypes;
- estimate the medoids;
- tune method parameters;
- modify the registered-set distance;
- choose a prototype using the ground-truth label;
- optimize explanation fidelity on test observations.

At inference time, routing is based on the **ensemble-predicted class**, not on the ground-truth test label.

---

## 3. Baselines, Evaluation Metrics, and Statistical Evidence

### 3.1 Compared Methods

| Evaluation scope | Compared methods |
|---|---|
| Local explanation | NEX-ELM, Kernel SHAP, and a random-ranking control |
| Global explanation | Prototype-conditioned NEX-ELM, aggregated Kernel SHAP, and X-ELM |
| Local runtime | Local NEX-ELM versus local Kernel SHAP |
| Complete explanatory workflow | NEX-ELM workflow versus the combined X-ELM + Kernel SHAP workflow |

The study does **not** claim that isolated NEX-ELM is faster than isolated X-ELM. The global runtime comparison concerns the complete workflow required to provide both local and global explanatory outputs.

### 3.2 Explanation Fidelity

Explanation fidelity is evaluated in two complementary directions:

- **Deletion AUC:** quantifies the degradation of the target model output when the highest-ranked features are removed first.
- **Insertion AUC:** quantifies the recovery of the target model output when the highest-ranked features are inserted first.
- **Composite AUC:** the registered summary of the deletion and insertion trajectories used for global fidelity analysis.

Within the implemented orientation of these metrics, larger values indicate higher explanation fidelity.

Agreement with SHAP is treated as a structural diagnostic rather than as ground truth. Deletion and insertion trajectories provide the principal model-behavior fidelity evidence.

### 3.3 Confirmatory Statistical Analysis

The analysis suite includes:

- Shapiro–Wilk normality diagnostics;
- one-sided paired *t* tests;
- one-sided Wilcoxon signed-rank tests;
- paired permutation tests;
- repeated-measures ANOVA;
- Friedman tests;
- Kendall's coefficient of concordance (*W*);
- Nemenyi post hoc comparisons;
- exact binomial tests for seed-level win rates;
- exact Clopper–Pearson confidence intervals;
- Holm correction for multiple testing.

The confirmatory decision rule jointly considers effect direction, confidence intervals, paired hypothesis tests, effect size, and consistency across random seeds.

---

## 4. Datasets Included in the Complete Study Plan

| Code identifier | Observations | Features | Classes | Source or experimental role |
|---|---:|---:|---:|---|
| `electrical_grid_stability` | 10,000 | 13 | 2 | UCI Electrical Grid Stability; retains `stab` to reproduce the X-ELM reference setting |
| `electrical_grid_stability_without_stab` | 10,000 | 12 | 2 | Controlled proxy-removal setting; excludes `stab` |
| `pima_indians_diabetes` | 768 | 8 | 2 | Pima Indians Diabetes |
| `wisconsin_breast_cancer_original` | 683 | 9 | 2 | Wisconsin Breast Cancer Original after data cleaning |
| `ionosphere_binary` | 351 | 34 | 2 | UCI Ionosphere |
| `wine_multiclass` | 178 | 13 | 3 | scikit-learn Wine dataset |
| `iris_multiclass` | 150 | 4 | 3 | scikit-learn Iris dataset |
| `digits_multiclass` | 1,797 | 64 | 10 | scikit-learn Digits dataset |
| `breast_cancer_diagnostic` | 569 | 30 | 2 | scikit-learn Breast Cancer Wisconsin Diagnostic dataset |

The scikit-learn datasets are loaded directly from the installed library. Pima Indians Diabetes, Wisconsin Breast Cancer Original, Electrical Grid Stability, and Ionosphere may be downloaded by the script into the `reference_data` directory.

---

## 5. Configuration Used to Generate the Reported Results

The following tables summarize the effective values resolved by the attached implementation when the complete study plan is executed without command-line overrides.

### 5.1 Experimental Protocol and Predictive Model

| Component | Effective configuration |
|---|---|
| Software version | `v68` |
| Method package | `GloPro-Complete` |
| Study plan | `complete` |
| Data mode | `real` |
| Dataset configurations | 9 |
| Independent repetitions per dataset | 30 |
| Primary experimental runs | 270 |
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
| Maximum prototypes per predicted class | 4 |
| Minimum calibration rows per prototype slot | 2 |
| Prototype clustering | Deterministic k-medoids |
| Prototype-library organization | Conditioned on the ensemble-predicted class |
| Registered minimum seed-level win rate | 0.80 |
| Minimum wins over 30 seeds | 24 |
| Significance level | 0.05 |
| Registered minimum effect size | 0.50 |

### 5.2 Explanation and Evaluation Parameters

| Parameter | Effective value |
|---|---:|
| Ablation protocol | `expected_background` |
| Kernel SHAP background size | 350 |
| Local observations explained by Kernel SHAP | 50 |
| `shap_nsamples` | 0, delegated to SHAP |
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

### 5.3 CUDA and Runtime Configuration

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
| CPU–CUDA numerical audit | Enabled |
| Numerical-audit tolerance | `2e-4` |

### 5.4 Experimental Computing Environment

| Resource | Environment used for the reported experiments |
|---|---|
| Motherboard | ASUS ROG STRIX Z590-F GAMING WIFI, LGA 1200 |
| Operating system | Microsoft Windows 11 Pro 64-bit, version 25H2, build 26200.7922 |
| Processor | 10th-generation Intel Core i9-10850K, 10 physical cores and 20 logical threads |
| System memory | 64 GB dual-channel DDR4-2666 SDRAM |
| GPU | NVIDIA GeForce RTX 3060, 12 GB GDDR6, 3,584 CUDA cores, Compute Capability 8.6 |
| Storage | WD Black SN750 2 TB M.2 NVMe SSD |
| Python | 3.10 |
| PyTorch | 2.12.1 with CUDA 12.6 |

For strict computational reproduction, record all package versions from the active environment:

```bash
pip freeze > requirements-lock.txt
```

---

## 6. Software Requirements and Installation

### 6.1 Required Software

The implementation requires:

- Python 3.10 or 3.11;
- NumPy;
- pandas;
- SciPy;
- Matplotlib;
- scikit-learn;
- SHAP;
- PyTorch;
- ReportLab, when generation of `relatorio.pdf` is required.

A CUDA-capable NVIDIA GPU is required for the default `complete` study plan. A reduced custom audit can be executed on a CPU.

### 6.2 Create a Virtual Environment

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

### 6.3 Install Dependencies

Install a PyTorch build compatible with the available GPU and CUDA runtime first. Then install the remaining packages:

```bash
pip install numpy pandas scipy matplotlib scikit-learn shap reportlab
```

For a CPU-only audit environment:

```bash
pip install torch numpy pandas scipy matplotlib scikit-learn shap reportlab
```

When a repository-level `requirements.txt` file is provided, install the declared dependency set with:

```bash
pip install -r requirements.txt
```

Verify the environment:

```bash
python -c "import torch, shap, sklearn, numpy, pandas, scipy; print('torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 7. Recommended Project Layout

```text
nex_elm_project/
├── nexlm_v68_journal_complete_datasets.py
├── README.md
├── requirements.txt                 # optional but recommended
├── reference_data/
└── results/
```

The integrity-checked mathematical core is embedded in the main script. An adjacent file named `nexlm_v68_glopro_complete_core.py` is optional and is loaded only when its SHA-256 digest exactly matches the expected value:

```text
fc78619795db892180c7b978366a67b15a48741103f0fcc98eeab7b7dcf2031f
```

When an adjacent core has a different digest, it is ignored and the embedded integrity-checked core is used instead.

---

## 8. Execution Instructions

### 8.1 Display Command-Line Help

```bash
python nexlm_v68_journal_complete_datasets.py --help
```

### 8.2 Reproduce the Complete Experiment

Running the script without arguments selects CUDA, the 12 GB RTX 3060 profile, 30 repetitions, and all nine dataset configurations:

```bash
python nexlm_v68_journal_complete_datasets.py
```

To specify the output directory:

```bash
python nexlm_v68_journal_complete_datasets.py \
  --output-dir nex_elm_v68_results
```

Equivalent one-line PowerShell command:

```powershell
python nexlm_v68_journal_complete_datasets.py --output-dir nex_elm_v68_results
```

> The `complete` study plan requires the resolved execution device to be CUDA. The script terminates with an error when the complete plan is requested without an available CUDA device.

### 8.3 Run a GPU Smoke Test

The `--quick` option reduces each study phase to one repetition and is suitable for validating dependencies, dataset downloads, CUDA initialization, and output-file creation:

```bash
python nexlm_v68_journal_complete_datasets.py \
  --quick \
  --output-dir nex_elm_smoke_test
```

### 8.4 Run a Reduced CPU Audit

CPU execution requires a non-default study plan, such as `custom`:

```bash
python nexlm_v68_journal_complete_datasets.py \
  --study-plan custom \
  --datasets iris \
  --n-repeats 1 \
  --device cpu \
  --skip-pdf-report \
  --output-dir nex_elm_cpu_audit
```

The alias `iris` is resolved to the canonical identifier `iris_multiclass`.

### 8.5 Run Selected Datasets

The following example runs three datasets with five independent random seeds:

```bash
python nexlm_v68_journal_complete_datasets.py \
  --study-plan custom \
  --datasets iris,wine,breast_diagnostic \
  --n-repeats 5 \
  --device cuda \
  --output-dir nex_elm_custom_results
```

Accepted aliases include:

| Alias | Canonical dataset identifier |
|---|---|
| `grid` | `electrical_grid_stability` |
| `grid_without_stab` | `electrical_grid_stability_without_stab` |
| `pima` | `pima_indians_diabetes` |
| `wisconsin` | `wisconsin_breast_cancer_original` |
| `ionosphere` | `ionosphere_binary` |
| `wine` | `wine_multiclass` |
| `iris` | `iris_multiclass` |
| `digits` | `digits_multiclass` |
| `breast_diagnostic` | `breast_cancer_diagnostic` |

### 8.6 Run Only the Replication Phase

```bash
python nexlm_v68_journal_complete_datasets.py \
  --study-plan replication \
  --replication-repeats 30 \
  --replication-random-state 100000 \
  --output-dir nex_elm_replication_results
```

The default replication phase includes:

- Electrical Grid Stability with `stab`;
- Pima Indians Diabetes;
- Wisconsin Breast Cancer Original.

### 8.7 Run Only the Generalization Phase

```bash
python nexlm_v68_journal_complete_datasets.py \
  --study-plan generalization \
  --generalization-repeats 30 \
  --generalization-random-state 100000 \
  --output-dir nex_elm_generalization_results
```

### 8.8 Use Local Dataset Files Without Downloading

Place the files in the directory supplied to `--data-dir`, then use `--no-download`:

```bash
python nexlm_v68_journal_complete_datasets.py \
  --data-dir reference_data \
  --no-download
```

Recognized filenames are:

```text
reference_data/
├── Data_for_UCI_named.csv
├── pima-indians-diabetes.data.csv
├── breast-cancer-wisconsin.data
└── ionosphere.data
```

### 8.9 Use the Conservative GPU Profile

When the default profile exceeds the available GPU memory, use:

```bash
python nexlm_v68_journal_complete_datasets.py \
  --gpu-profile conservative \
  --output-dir nex_elm_conservative_gpu_results
```

The GPU profile modifies execution and batching parameters only. The integrity-checked mathematical core remains unchanged.

### 8.10 Recompute Statistical Analyses Without Rerunning the Models

```bash
python nexlm_v68_journal_complete_datasets.py \
  --statistics-only-from nex_elm_v68_results \
  --statistics-output-dir nex_elm_statistical_reanalysis
```

The source directory must contain `seed_metrics.csv` and, when available, `estatistica_entre_seeds.csv`, either directly or under `combined/tabelas`.

### 8.11 Regenerate Only the PDF Report

```bash
python nexlm_v68_journal_complete_datasets.py \
  --report-only-from nex_elm_v68_results \
  --report-output nex_elm_report.pdf
```

To run the experiment without generating the PDF report:

```bash
python nexlm_v68_journal_complete_datasets.py --skip-pdf-report
```

---

## 9. Output Structure

A standard run produces a directory with the following general structure:

```text
resultados_v68_glopro_complete_YYYYMMDD_HHMMSS/
├── plano_confirmatorio.csv
├── plano_validacao_completa_v68.csv
├── registro_seeds_v68.csv
├── registro_protocolo_confirmatorio.json
├── limites_de_alegacao_v68.csv
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

Some generated filenames and directory names remain in Portuguese because they are part of the current implementation contract.

### 9.1 Principal Predictive Outputs

| File | Content |
|---|---|
| `predictive_performance_per_seed.csv` | Predictive metrics for each random seed |
| `predictive_performance_summary.csv` | Across-seed means and confidence intervals |
| `predictive_performance_article_table.csv` | Manuscript-ready predictive-performance table |
| `predictive_confusion_matrix.csv` | Complete confusion matrices |
| `predictive_confusion_matrix_summary.csv` | Across-seed confusion-matrix summary |
| `predictive_class_metrics.csv` | Per-class predictive metrics |
| `predictive_class_metrics_summary.csv` | Across-seed summary of per-class metrics |
| `predictive_dataset_summary.csv` | Dataset dimensions and task characteristics |

### 9.2 Explainability and Prototype Outputs

| File | Content |
|---|---|
| `local_fidelity_summary.csv` | Local explanation fidelity by method and seed |
| `global_fidelity_summary.csv` | Global explanation fidelity by method and seed |
| `local_agreement.csv` | Local structural agreement diagnostics |
| `global_importance.csv` | Dataset-level global feature importance |
| `global_class_importance.csv` | Class-conditional global feature importance |
| `prototype_library.csv` | Complete explanatory prototype library |
| `prototype_library_class_summary.csv` | Prototype-library summary by predicted class |
| `prototype_routing.csv` | Observation-to-prototype routing records |
| `prototype_routing_summary.csv` | Summary of routing behavior |
| `prototype_stability_between_seeds.csv` | Across-seed prototype stability diagnostics |

### 9.3 Statistical, Runtime, and Audit Outputs

| File | Content |
|---|---|
| `seed_metrics.csv` | Central across-seed inferential dataset |
| `inferential_test_summary.csv` | Summary of confirmatory tests |
| `paired_t_tests.csv` | Paired *t*-test results |
| `shapiro_wilk_tests.csv` | Normality diagnostics |
| `repeated_measures_anova.csv` | Repeated-measures ANOVA results |
| `friedman_tests.csv` | Friedman-test results |
| `average_method_ranks.csv` | Average ranks of the compared methods |
| `nemenyi_posthoc.csv` | Nemenyi post hoc comparisons |
| `binomial_win_rate_tests.csv` | Exact binomial tests and Clopper–Pearson intervals |
| `multiple_testing_families.csv` | Multiple-testing families and corrections |
| `timing.csv` | Local and complete-workflow runtime measurements |
| `cuda_audit.csv` | CPU–CUDA numerical-equivalence audit |
| `gpu_runtime.csv` | GPU runtime, device, and memory information |

---

## 10. Integrity and Reproducibility Controls

Before running the experimental workflow, the script:

1. loads the embedded mathematical core or an adjacent core with an identical SHA-256 digest;
2. verifies the mathematical-core digest;
3. executes internal self-tests;
4. fixes the prototype configuration at `K = 4`;
5. records the public protocol and resolved configuration;
6. prevents unapproved overlap with the preceding confirmatory seed battery;
7. records the seeds, scenarios, and claim boundaries.

The prototype configuration cannot be changed through the command-line interface. The following commands intentionally raise an error:

```bash
python nexlm_v68_journal_complete_datasets.py --prototype-count 3
python nexlm_v68_journal_complete_datasets.py --prototype-min-calibration-rows-per-slot 4
```

The only accepted values in this implementation are:

```text
prototype-count = 4
prototype-min-calibration-rows-per-slot = 2
```

The second value is a prototype-capacity parameter. It is not a post-clustering minimum cluster-size constraint.

---

## 11. Troubleshooting

### `The shap package is required`

Install SHAP in the active environment:

```bash
pip install shap
```

### `The torch package is required for NEX-ELM`

Install a PyTorch build compatible with the intended CPU or CUDA execution environment before starting the experiment.

### `The default complete experiment requires CUDA`

The complete plan is not available in CPU-only mode. Use a custom reduced audit instead:

```bash
python nexlm_v68_journal_complete_datasets.py \
  --study-plan custom \
  --datasets iris \
  --n-repeats 1 \
  --device cpu
```

### CUDA Out-of-Memory Error

First use the conservative GPU profile:

```bash
python nexlm_v68_journal_complete_datasets.py --gpu-profile conservative
```

A custom profile can also be used to reduce the main batch sizes:

```bash
python nexlm_v68_journal_complete_datasets.py \
  --gpu-profile custom \
  --gpu-batch-size 8192 \
  --estimator-batch-size 32 \
  --nex-instance-batch-size 1 \
  --nex-background-block-size 64
```

### Dataset File Not Found with `--no-download`

Verify the path supplied to `--data-dir` and the recognized filenames listed in Section 8.8.

### PDF Report Was Not Generated

Install ReportLab:

```bash
pip install reportlab
```

Alternatively, skip PDF generation:

```bash
python nexlm_v68_journal_complete_datasets.py --skip-pdf-report
```

### Results Differ Across Machines

Small numerical differences may arise from package versions, CUDA libraries, TF32 behavior, hardware, thread scheduling, and parallel execution. Record the complete environment with:

```bash
python --version
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
pip freeze > requirements-lock.txt
```

For strict replication, preserve the script, dataset files, package lock, GPU model, CUDA runtime, and recorded seed registry.

---

## 12. Scope and Interpretation Boundaries

The generated evidence supports claims concerning:

- explanation fidelity under the registered empirical interventional game;
- local, global, and glocal explanatory granularity;
- consistency across independent random seeds;
- prototype-conditioned explanatory organization;
- computational cost in the audited hardware and software environment.

The experiments do not, by themselves, establish:

- causal feature effects;
- algorithmic fairness;
- usefulness to human end users;
- superiority over every available XAI method;
- runtime superiority over isolated X-ELM;
- universal absence of GPU memory leaks;
- automatic generalization to images, text, time series, or non-ELM predictive models.

The CUDA audit verifies numerical agreement and records memory behavior within the evaluated runs; it should not be interpreted as a universal proof that memory leaks cannot occur in other environments or workloads.

---

## 13. Citation

When using this implementation in academic work, cite the associated NEX-ELM manuscript. Replace the provisional record below with the final bibliographic metadata and DOI after publication.

```bibtex
@article{nexelm,
  title   = {NEX-ELM: Advancing Glocal Explainable AI for Extreme Learning Machines},
  author  = {Author information omitted for double-blind review},
  journal = {Neural Computing and Applications},
  year    = {forthcoming}
}
```

---

## 14. License

No license file was included with the supplied implementation. Before public redistribution, add a license that is compatible with the authors' intended terms, the third-party dependencies, and the licenses or usage conditions of the included datasets.
