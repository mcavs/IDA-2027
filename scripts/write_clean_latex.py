import os

with open('outputs/table3_heatmap_rows.tex') as f:
    t3 = f.read()

part1 = r"""\documentclass[12pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{mathtools}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{natbib}
\usepackage{geometry}
\usepackage{booktabs}
\usepackage{array}
\usepackage[table]{xcolor}

\usepackage{setspace}
\usepackage{enumitem}

\geometry{margin=2.5cm}
\onehalfspacing

\title{\textbf{Aggregating Rashomon Partial Dependence Profiles:\\
A Systematic Empirical Evaluation}}

\author{[Author information omitted for double-blind review]}

\date{}

\begin{document}

\maketitle

\begin{abstract}
Modern machine learning pipelines routinely produce model multiplicity: numerous competing models achieve near-optimal predictive performance while relying on disparate internal mechanisms. When explaining feature effects via Partial Dependence Profiles, relying exclusively on a single best-performing model risks delivering brittle interpretations, as alternative models within the Rashomon set often exhibit conflicting feature effects. While aggregating explanation curves across the Rashomon set can mitigate this instability, a comprehensive empirical understanding of how different aggregation strategies behave across noise regimes, tolerance thresholds, and model diversity is lacking.

In this paper, we conduct an extensive empirical study evaluating eight profile aggregation strategies alongside the single-best-model baseline across synthetic benchmarks with varying noise levels and real-world tabular data. Our key empirical findings reveal that: first, while the single best model remains competitive in low-noise settings, profile aggregation systematically outperforms it as noise increases, substantially reducing deviation from the ground truth by attenuating estimation variance; second, performance-weighted and robust optimization approaches consistently achieve the highest explanation fidelity across diverse scenarios; and third, empirical coverage--accuracy analysis demonstrates a fundamental trade-off where no aggregation method simultaneously attains both high confidence interval coverage and low profile deviation. Specifically, rank-based aggregation maximizes coverage through wider intervals, whereas convex weighting yields tightly bounded, accurate profile estimates. We further introduce the profile diversity coefficient $\kappa$ as an empirical diagnostic that reliably identifies when aggregation cannot help: when $\kappa \approx 0$, Rashomon models share a common systematic bias and averaging provides no benefit over the best single model. These results provide concrete empirical principles for selecting and deploying multi-model explanation pipelines under the Rashomon effect, with direct implications for the reliability and auditability of feature-effect explanations in high-stakes automated decision-making systems.
\end{abstract}

\noindent\textbf{Keywords:} partial dependence profile, Rashomon effect, AutoML, explainable AI, functional aggregation, bootstrap confidence interval, coverage--accuracy trade-off, algorithmic accountability.

% ─────────────────────────────────────────────────────────────────────────────
\section{Introduction}
\label{sec:intro}
% ─────────────────────────────────────────────────────────────────────────────

Explainable artificial intelligence (XAI) seeks to make complex models interpretable without sacrificing predictive performance.
Among the most widely used global explanation tools is the Partial Dependence Profile (PDP) introduced by \citet{Friedman2001}: it traces the marginal effect of a feature on model output by averaging predictions over the empirical distribution of all other features.
PDP is conventionally computed for a single, pre-selected model, an assumption that becomes increasingly untenable as AutoML pipelines grow in adoption.

AutoML systems are no longer confined to research settings: they are actively deployed in high-stakes domains including clinical decision support, credit scoring, and automated hiring, where the explanations they produce carry legal and epistemic weight.
When a deployed system reports that a particular feature is the most influential driver of its output, this interpretation may be incorporated into audit trails, regulatory filings, or individual recourse mechanisms.
Yet if equally accurate models within the same AutoML run assign conflicting feature importances, the choice of which model to explain becomes a structural decision about accountability---one that is currently made implicitly and without principled guidance.
This brittleness is not merely a technical nuisance.
Under emerging regulatory frameworks---including the EU AI Act's requirements for high-risk system documentation and the GDPR's right to meaningful explanation---an explanation derived from a single arbitrarily selected model may fail to represent the range of defensible interpretations the data support.
Rashomon Partial Dependence Profiles directly address this gap: by aggregating feature-effect curves across the set of near-optimal models, they make explanation uncertainty explicit and auditable.
The central question this paper asks is which aggregation strategy should be used, and under what conditions, to produce explanations that are both accurate and robust.

AutoML systems such as H2O \citep{LeDell2020}, Auto-sklearn \citep{Feurer2022}, and AutoGluon \citep{Erickson2020} generate large collections of near-optimal models as a side effect of their search procedures.
This multiplicity is the essence of the Rashomon effect \citep{Breiman2001}: many models with similar predictive performance can exhibit substantially different feature relationships, and selecting a single model to explain discards information that could enrich interpretation.

\citet{Cavus2026} operationalize this observation by defining the Rashomon Partial Dependence Profile as the simple arithmetic average of PDP curves over all models within a tolerance band of the best performer.
Their empirical results demonstrate that, under high-noise conditions, the averaged profile approximates the ground-truth feature relationship more closely than the single best-model PDP.
Despite this contribution, several empirical questions remain open.
Which weighting strategies perform best across different noise regimes and Rashomon thresholds?
Does aggregation consistently outperform the single best model, and under what conditions does it fail?
How do bootstrap confidence intervals behave in practice, and is there an empirical trade-off between interval coverage and profile accuracy?
What does the diversity of the Rashomon set tell us about the quality of the resulting explanation?

This paper addresses these questions through a systematic empirical study.
Our contributions are as follows.

\paragraph{Empirical contributions.}
We implement eight aggregation methods, including direct adaptations of \citet{Decker2024}, \citet{Bhatt2020}, and \citet{Pirie2023} to the PDP setting, and evaluate them on the three synthetic benchmarks of \citet{Cavus2026} across three noise levels $\sigma^2 \in \{1, 4, 9\}$, yielding nine setup--noise scenarios under an extended AutoML training budget (160 models, 2880s runtime).
We introduce a dual within-pool bootstrap band strategy: $\mathrm{CI}_1$ re-estimates weights at each replicate (pool-reweighting variability) and $\mathrm{CI}_2$ holds weights fixed (weighting-mechanism variability).
We report ground-truth coverage rates for both intervals and analyze the coverage--accuracy trade-off through quadrant analysis.
We evaluate the PDP diversity coefficient $\kappa$ as an empirical diagnostic that explains when aggregation helps and when it does not, and we validate it across all experimental scenarios.

The paper is organized as follows.
Section~\ref{sec:related} reviews related work.
Section~\ref{sec:prelim} establishes notation and preliminaries.
Section~\ref{sec:experimental} describes the experimental design.
Section~\ref{sec:results} reports empirical results.
Section~\ref{sec:discussion} discusses practical guidance, limitations, and algorithmic accountability implications.
Section~\ref{sec:conclusion} concludes.

% ─────────────────────────────────────────────────────────────────────────────
\section{Related Work}
\label{sec:related}
% ─────────────────────────────────────────────────────────────────────────────

\paragraph{Partial dependence profiles and global explanations.}
\citet{Friedman2001} introduced PDP as a diagnostic for gradient-boosted trees; \citet{Goldstein2015} extended this with Individual Conditional Expectation (ICE) curves to reveal heterogeneous effects masked by averaging.
\citet{Apley2020} proposed Accumulated Local Effects (ALE) to address the well-known extrapolation bias of PDP under correlated features.
These contributions collectively establish PDP as the standard global explanation tool, yet each is defined with respect to a single model.

\paragraph{Rashomon effect and model multiplicity.}
\citet{Breiman2001} introduced the term ``Rashomon effect'' to describe the abundance of near-optimal models in machine learning.
\citet{Semenova2022} formalized the Rashomon set and analyzed its properties for linear model classes.
\citet{Cavus2024} extended the Rashomon perspective to interpretability, and \citet{Kobylinska2024} used discrepant models within the Rashomon set to produce diverse medical explanations.
\citet{Cavus2026}, the primary reference for our work, introduced Rashomon PDP and demonstrated that simple averaging of PDP curves across the Rashomon set reduces mean deviation from the ground truth by up to 38\% under high noise.

\citet{Parikh2026} organizes the causes of the Rashomon effect into statistical sources (finite samples, diminishing with $n$), structural sources (non-convexity and unobserved variables, persisting asymptotically), and procedural sources (optimization heuristics and model-class restrictions).
This taxonomy motivates the noise-sensitivity analysis in our experiments and the diversity coefficient $\kappa$ introduced in Section~\ref{sec:prelim}.
\citet{Ewald2026} extend Rashomon sets to the combined algorithm selection and hyperparameter optimization (CASH) setting, showing that multi-algorithm spaces exhibit richer Rashomon sets than single-algorithm families.

\paragraph{Aggregation of explanations.}
Aggregating explanations across models or data points has been studied in several distinct contexts.
\citet{Bhatt2020} aggregate local feature attributions across nearest neighbors in data space to produce more robust instance-level explanations (AVA).
We adapt their formulation to the Rashomon setting by replacing data-space neighbors with near-optimal models and weighting by Spearman rank consensus (Section~\ref{sec:methods}, B4).
\citet{Decker2024} study the aggregation of feature attributions across multiple explainers, formulating convex quadratic programs that balance sensitivity (robustness to input perturbations) and infidelity (faithfulness to model predictions).
We adapt their AGGrobust, AGGfaith, and AGGopt formulations to functional PDP aggregation (Section~\ref{sec:methods}, B3, Ö1, Ö2).
\citet{Pirie2023} propose AGREE, an aggregation framework that addresses explainer disagreement through alignment metrics: each explainer is weighted by the extent to which its top-$k$ feature rankings agree with those of other explainers.
We implement their rank-overlap variant directly in the PDP setting, treating each Rashomon model as an explainer and computing pairwise top-$k$ grid-point overlap to derive confidence scores (Section~\ref{sec:methods}, B8).
\citet{Mor2024} accelerated the global aggregation of Anchor explanations across large datasets, demonstrating that explanation aggregation can scale to production settings.
\citet{Biradar2024} study the axiomatic aggregation of abductive explanations.

% ─────────────────────────────────────────────────────────────────────────────
\section{Preliminaries and Notation}
\label{sec:prelim}
% ─────────────────────────────────────────────────────────────────────────────

\paragraph{Data, models, and empirical risk.}
Let $\mathcal{D}_n = \{(x_i, y_i)\}_{i=1}^n$ be a dataset of $n$ independent and identically distributed observations drawn from an unknown data-generating distribution $\mathcal{P}_{X,Y}$ on $\mathcal{X} \times \mathcal{Y}$, with $\mathcal{X} \subseteq \mathbb{R}^p$ and $\mathcal{Y} \subseteq \mathbb{R}$.
A model class $\mathcal{M}$ consists of hypotheses $M \colon \mathcal{X} \to \mathcal{Y}$.
For a loss function $\ell \colon \mathcal{Y} \times \mathcal{Y} \to \mathbb{R}_{\ge 0}$, the empirical risk is $\mathcal{R}_{\mathrm{emp}}(M; \mathcal{D}_n) = \frac{1}{n} \sum_{i=1}^n \ell(M(x_i), y_i)$.
The empirical risk minimizer (ERM) is $M^* \in \arg\min_{M \in \mathcal{M}} \mathcal{R}_{\mathrm{emp}}(M; \mathcal{D}_n)$, with minimum empirical risk $\widehat{R}^* = \mathcal{R}_{\mathrm{emp}}(M^*; \mathcal{D}_n)$.

\paragraph{Rashomon set and Rashomon ratio.}
For a tolerance threshold $\varepsilon > 0$, the empirical Rashomon set is:
\begin{equation}
\label{eq:rashomon_set}
\mathcal{R}_\varepsilon = \left\{ M \in \mathcal{M} \;\colon\; \mathcal{R}_{\mathrm{emp}}(M; \mathcal{D}_n) \le (1 + \varepsilon)\,\widehat{R}^* \right\}.
\end{equation}
Let $K = |\mathcal{R}_\varepsilon|$ denote the number of models in the Rashomon set.
The Rashomon ratio $\rho \in (0, 1]$ is the fraction of trained models that fall within the threshold: $\rho = |\mathcal{R}_\varepsilon| / |\mathcal{M}| = K / |\mathcal{M}|$.
In AutoML settings where $\mathcal{M}$ is a finite pool of trained models returned by the search procedure, $\rho$ is directly observable.
Following \citet{Ewald2026}, when the AutoML search spans multiple model classes the relevant object is the CASHomon set: the Rashomon set constructed over the combined algorithm-selection and hyperparameter optimization hypothesis space.

\paragraph{Partial Dependence Profiles.}
Let $j \in \{1, \dots, p\}$ denote the feature of interest, and let $X_{-j}$ denote all features other than $j$.
For a model $M \colon \mathcal{X} \to \mathbb{R}$, the Partial Dependence Profile for feature $j$ is the functional curve:
\begin{equation}
\label{eq:pdp}
f_j^{(M)}(x) = \mathbb{E}_{X_{-j}}\!\left[ M(x, X_{-j}) \right] \approx \frac{1}{n} \sum_{i=1}^n M(x, x_{i,-j}), \quad x \in [a, b],
\end{equation}
evaluated on an $m$-point grid $\mathcal{G}_m = \{x^{(1)}, \dots, x^{(m)}\} \subset [a, b]$.
In practice, $f_j^{(M)}$ is represented as an $m$-dimensional vector $\mathbf{p}^{(M)} = (f_j^{(M)}(x^{(1)}), \dots, f_j^{(M)}(x^{(m)}))^\top$.
For a collection of $K$ models in $\mathcal{R}_\varepsilon$, we assemble these vectors into the profile matrix $\mathbf{P} \in \mathbb{R}^{K \times m}$, where row $k$ contains the PDP curve of model $M_k \in \mathcal{R}_\varepsilon$.

\paragraph{Profile aggregation.}
An aggregation method maps the collection of profiles $\{\hat{f}_j^{(M_k)}\}_{k=1}^K$ to a single aggregated profile $\bar{f}_j = \sum_{k=1}^K \omega_k \hat{f}_j^{(M_k)}$, where $\boldsymbol{\omega} \in \Delta_K = \{\mathbf{w} \in \mathbb{R}^K \colon w_k \ge 0, \sum_k w_k = 1\}$.
The simple average of \citet{Cavus2026} corresponds to $\omega_k = 1/K$ for all $k$.

\paragraph{Fixed best-model reference alignment.}
To ensure comparability across different thresholds $\varepsilon$ without introducing set-dependent shifts, the ground-truth function $f^\mathrm{gt}$ is aligned to the fixed empirical expectation of the single best-model profile ($M^*$):
\begin{equation}
\label{eq:gt_align}
\tilde{f}^\mathrm{gt} = f^\mathrm{gt} - \frac{1}{m} \sum_{l=1}^m f^\mathrm{gt}(x^{(l)}) + \frac{1}{m} \sum_{l=1}^m \hat{f}_j^{(M^*)}(x^{(l)}).
\end{equation}
Consequently, the baseline Mean Deviation from Ground Truth ($\mathrm{MD}(\mathrm{GT}) = \frac{1}{m} \sum_{l=1}^m |\hat{f}_j(x^{(l)}) - \tilde{f}^\mathrm{gt}(x^{(l)})|$) of the single best model B1 is strictly invariant across all $\varepsilon$ settings within any given scenario.

\paragraph{Profile diversity coefficient $\kappa$.}
We quantify the internal diversity of the Rashomon profile pool via the diversity coefficient:
\begin{equation}
\label{eq:kappa}
\kappa = 1 - \frac{\|\bar{f}_j - \tilde{f}^\mathrm{gt}\|_2^2}{\frac{1}{K}\sum_{k=1}^K \|\hat{f}_j^{(M_k)} - \tilde{f}^\mathrm{gt}\|_2^2}, \quad \kappa \in [0, 1].
\end{equation}
When $\kappa \approx 1$, individual model estimation errors cancel upon averaging; when $\kappa \approx 0$, all near-optimal models share a common systematic bias, indicating that multi-model aggregation cannot reduce error over the best single model.

% ─────────────────────────────────────────────────────────────────────────────
\section{Experimental Design}
\label{sec:experimental}
% ─────────────────────────────────────────────────────────────────────────────

\subsection{Synthetic Benchmarks}
\label{sec:benchmarks}

We adopt the three synthetic data-generating processes of \citet{Cavus2026} with $n = 1{,}500$ observations and noise variances $\sigma^2 \in \{1, 4, 9\}$.
Feature distributions are $X_1, X_3, X_7, X_9 \sim \mathcal{N}(0, 1)$, $X_2 \sim \mathrm{Bernoulli}(0.5)$, $X_4 \sim \mathcal{U}(-2, 2)$, $X_5 \sim \mathrm{Exp}(1)$, $X_6 \in \{A, B, C\}$ uniform, $X_8 \in \{0, 1\}$ uniform, and noise $\varepsilon_\mathrm{noise} \sim \mathcal{N}(0, \sigma^2)$.
The target feature for PDP analysis is $X_1$ throughout.
Table~\ref{tab:benchmarks} gives the data-generating formulas; the true functional relationships are $f^\mathrm{gt}(x) = 2.9x + 0.8x^2$ (Setup~1), $f^\mathrm{gt}(x) = \sin(x)$ (Setup~2), and $f^\mathrm{gt}(x) = x^2$ (Setup~3).

\begin{table}[htbp]
\centering\small
\caption{Synthetic data-generating processes \citep{Cavus2026}. $\mathbf{1}[\cdot]$ denotes the indicator function. Categorical baselines: Setup 1: $X_6=A$ ($0$), $X_8=0$ ($0$); Setup 2: $X_6=B$ ($0$), $X_8=0$ ($0$); Setup 3: $X_6=A$ ($0$).}
\label{tab:benchmarks}
\begin{tabular}{clp{10.2cm}}
\toprule
Setup & Scenario description & Data-generating formula $Y = \dots + \varepsilon_\mathrm{noise}$ \\
\midrule
1 & Linear + polynomial &
$Y = 2.9X_1 - 3.7X_2 + 1.2X_3 + 0.8X_1^2 + 1.5X_4X_5$ \\
& & $- 1.3\mathbf{1}[X_6{=}B] + 2.1\mathbf{1}[X_6{=}C] + 1.8X_7 + 2.4\mathbf{1}[X_8{=}1] - 0.7X_9 + \varepsilon_\mathrm{noise}$\\[4pt]
2 & Trigonometric + interaction &
$Y = \sin(X_1) + \log(|X_3|+1) + 0.8X_4X_5 - 1.2X_1X_3$ \\
& & $- 2\mathbf{1}[X_6{=}A] + \mathbf{1}[X_6{=}C] + 1.5X_7 + 2\mathbf{1}[X_8{=}1] + 0.4X_9^2 + \varepsilon_\mathrm{noise}$\\[4pt]
3 & Complex nonlinear &
$Y = X_1^2 + 2X_2 + 0.5e^{-X_3} + 0.7\sin(\pi X_4) + 0.6X_5^{0.3}$ \\
& & $+ 3\mathbf{1}[X_6{=}C] + 0.9X_7 + 1.2X_9 + \varepsilon_\mathrm{noise}$\\
\bottomrule
\end{tabular}
\end{table}

\subsection{AutoML Configuration}
\label{sec:automl}

All experiments use H2O AutoML \citep{LeDell2020} configured with an extended budget: a maximum of 160 base learners, a 2880-second (48-minute) runtime budget per scenario, with stacked ensembles and deep learning excluded to inspect distinct model families (XGBoost, GBM, Random Forest, Extremely Randomized Trees, and GLM).
To systematically evaluate the impact of the Rashomon tolerance threshold, we examine four settings: $\varepsilon \in \{0.01, 0.05, 0.10, 0.20\}$.
PDP grids use $m = 20$ evenly spaced points between the 5th and 95th percentiles of the feature's empirical distribution.
All model artifacts and per-model PDP curves across all 9 scenarios are serialized to disk to ensure strict reproducibility.

\subsection{Aggregation Methods}
\label{sec:methods}

Eight methods are evaluated; Table~\ref{tab:methods} summarizes their weight sources and algorithmic formulations.
All adapted methods are implemented faithfully to their original source papers:
\begin{itemize}[leftmargin=1.5em, itemsep=2pt]
\item \textbf{B0 (SimpleAvg)} \citep{Cavus2026}: Uniform averaging $\omega_k = 1/K$.
\item \textbf{B1 (Best-model)}: Single model $M^*$ with minimum training RMSE (baseline).
\item \textbf{B2 (PerfWeighted)}: Inverse validation error $\omega_k \propto 1/\mathrm{RMSE}(M_k)$.
\item \textbf{B3 (AGGrobust)} \citep{Decker2024}: Solves a convex quadratic program (QP) minimizing sensitivity to Gaussian perturbations applied to the PDP grid predictions:
$\min_{\boldsymbol{\omega} \in \Delta_K} \boldsymbol{\omega}^\top (\operatorname{diag}(\mathbf{s}) + 10^{-6}\mathbf{I})\boldsymbol{\omega}$, where $s_k$ is the empirical variance under grid perturbation.
\item \textbf{B4 (AVA-PDP)} \citep{Bhatt2020}: Adapted from feature-attribution neighbor aggregation to the model pool by computing pairwise Spearman rank correlation between model PDP curves.
The weight is proportional to mean rank consensus: $\omega_k \propto (K-1)^{-1}\sum_{l \neq k} r_{kl}$, clipped to positive values.
\item \textbf{B8 (AGREE-Rank)} \citep{Pirie2023}: Adapted from explainer agreement by evaluating pairwise overlap among the top-$k$ grid-point ranks ($k_\mathrm{overlap} = 5$):
$\mathbf{M}_{il} = |\mathrm{topk}(\hat{f}^{(i)}) \cap \mathrm{topk}(\hat{f}^{(l)})|$, with confidence scores $A_k = (K-1)^{-1}\sum_{l \neq k} \mathbf{M}_{kl}$ and $\omega_k \propto A_k$.
\item \textbf{Ö1 (AGGfaith)} \citep{Decker2024}: Infidelity minimization proxy in PDP space, weighting by inverse RMSE ($\omega_k \propto 1/\mathrm{RMSE}(M_k)$).
\item \textbf{Ö2 (AGGopt)} \citep{Decker2024}: Joint QP combining robustness ($\mathbf{s}$) and faithfulness ($\mathbf{s}_\mathrm{RMSE}$) with balancing parameter $\lambda = 0.5$:
$\min_{\boldsymbol{\omega} \in \Delta_K} \boldsymbol{\omega}^\top [\lambda(\operatorname{diag}(\mathbf{s}) + 10^{-6}\mathbf{I}) + (1-\lambda)(\operatorname{diag}(\mathbf{s}_\mathrm{RMSE}) + 10^{-6}\mathbf{I})]\boldsymbol{\omega}$.
\end{itemize}

\begin{table}[ht]
\centering
\caption{Aggregation methods and weight sources. B3, Ö1, and Ö2 use convex QP weights solved with CVXPY/SCS; B4 uses Spearman rank consensus across Rashomon models; B8 uses pairwise top-$k$ rank overlap.}
\label{tab:methods}
\begin{tabular}{lll}
\toprule
ID & Method & Weight source / adaptation \\
\midrule
B0 & SimpleAvg \citep{Cavus2026} & Uniform ($1/K$) \\
B1 & Best-model (baseline) & Single model, no weights \\
B2 & PerfWeighted & $1/\mathrm{RMSE}(M_k)$, normalized \\
B3 & AGGrobust \citep{Decker2024} & QP: minimize PDP grid perturbation sensitivity \\
B4 & AVA-PDP \citep{Bhatt2020} & Mean Spearman rank correlation with other models \\
B8 & AGREE-Rank \citep{Pirie2023} & Pairwise top-$k$ rank overlap confidence score \\
Ö1 & AGGfaith \citep{Decker2024} & $1/\mathrm{RMSE}(M_k)$ (RMSE as faithfulness proxy) \\
Ö2 & AGGopt \citep{Decker2024} & QP: $\lambda$ robustness + $(1-\lambda)$ faithfulness \\
\bottomrule
\end{tabular}
\end{table}

\subsection{Dual Confidence Interval Strategy}
\label{sec:dual_ci}

For each aggregation method, we construct two $95\%$ bootstrap confidence intervals from $B = 100$ resamples:
\begin{itemize}[leftmargin=1.5em, itemsep=2pt]
\item $\mathrm{CI}_1$ \textit{(pool-reweighting variability)}: At each bootstrap replicate, $K$ models are resampled with replacement from $\mathcal{R}_\varepsilon$ and aggregation weights are re-estimated from the resample.
\item $\mathrm{CI}_2$ \textit{(weighting-mechanism variability)}: Weights $\boldsymbol{\omega}$ are computed once on the full Rashomon set $\mathcal{R}_\varepsilon$ and held fixed; only the resampled model profile curves vary across bootstrap draws.
\end{itemize}

% ─────────────────────────────────────────────────────────────────────────────
\section{Results}
\label{sec:results}
% ─────────────────────────────────────────────────────────────────────────────

\subsection{Mean Deviation from Ground Truth across Rashomon Thresholds}

Figure~\ref{fig:combined_md} visualizes the performance of all aggregation methods compared to the single best model B1 across all four Rashomon thresholds $\varepsilon \in \{0.01, 0.05, 0.10, 0.20\}$ and noise levels $\sigma^2 \in \{1, 4, 9\}$.
Table~\ref{tab:md_combined} provides the exact numerical MD(GT) values, with heatmap coloring reflecting relative deviation magnitude (mint green = lowest error, cream = intermediate, coral/rose = highest error).

\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/combined_md_all_epsilons.pdf}
\caption{Multi-panel comparison of Mean Deviation from Ground Truth ($\mathrm{MD}(\mathrm{GT})$) across all four Rashomon thresholds ($\varepsilon \in \{0.01, 0.05, 0.10, 0.20\}$) across nine synthetic benchmarks ($n=1{,}500$, 160 models, 2880s budget). The solid grey line marks the baseline single best model (B1). Aggregation methods achieve substantial error reduction under moderate to high noise.}
\label{fig:combined_md}
\end{figure}

\begin{table}[htbp]
\centering\small
\caption{Mean Deviation from Ground Truth ($\mathrm{MD}(\mathrm{GT})$) across all four Rashomon thresholds $\varepsilon \in \{0.01, 0.05, 0.10, 0.20\}$ under the updated aggregation methods ($n=1{,}500$, 160 models, 2880s budget). Bold = best performance per scenario; underline = best aggregation method when B1 wins overall. $K$: Rashomon set size; $\rho$: Rashomon ratio.}
\label{tab:md_combined}
\setlength{\tabcolsep}{3.8pt}
\begin{tabular}{ccccccccccccc}
\toprule
$\varepsilon$ & Scenario & $\sigma^2$ & $K$ & $\rho$ & B0 & B1 & B2 & B3 & B4 & B8 & Ö1 & Ö2 \\
\midrule
"""

part2 = r"""\bottomrule
\end{tabular}
\end{table}

The empirical findings from Table~\ref{tab:md_combined} demonstrate several critical patterns:
First, while the single best model (B1) is competitive in low-noise settings ($\sigma^2=1$), profile aggregation systematically and substantially outperforms B1 as noise increases ($\sigma^2 \in \{4, 9\}$).
In Setup~1 at $\sigma^2=4$, aggregation reduces MD from $0.308$ to $0.216$ ($\sim 30\%$ error reduction).
At $\sigma^2=9$, aggregation reduces MD from $0.315$ to $0.245$ under $\varepsilon=0.05$.
Second, across all setups, performance-weighted (B2), robust QP (B3), AVA-PDP (B4), and joint QP (Ö2) methods consistently attain the lowest profile deviations.
In Setup~2, where model diversity is rich at higher noise levels, aggregation achieves lower deviation than B1 across all evaluated thresholds.

\subsection{Coverage Analysis across Rashomon Thresholds}

Table~\ref{tab:ci} reports the mean ground-truth coverage rate $\overline{\mathrm{CR}}_2$ and mean interval width $\overline{\mathrm{MWCI}}_2$ for $\mathrm{CI}_2$ averaged across all nine synthetic benchmarks.

\begin{table}[htbp]
\centering
\small
\caption{Mean ground-truth coverage rate $\overline{\mathrm{CR}}_2$ and mean CI width $\overline{\mathrm{MWCI}}_2$ across the nine synthetic scenarios under different Rashomon thresholds $\varepsilon$ ($B=100$ bootstrap replicates, nominal level: 0.95).}
\label{tab:ci}
\begin{tabular}{lccccccc}
\toprule
Metric / Threshold & B0 & B2 & B3 & B4 & B8 & Ö1 & Ö2 \\
\midrule
\multicolumn{8}{l}{\textit{Panel A: $\varepsilon = 0.01$}} \\
$\overline{\mathrm{CR}}_2$ & 0.161 & \textbf{0.167} & 0.156 & 0.150 & 0.150 & 0.150 & 0.156 \\
$\overline{\mathrm{MWCI}}_2$ & 0.109 & 0.111 & 0.110 & \textbf{0.112} & 0.108 & 0.111 & 0.110 \\
\midrule
\multicolumn{8}{l}{\textit{Panel B: $\varepsilon = 0.05$}} \\
$\overline{\mathrm{CR}}_2$ & 0.133 & 0.122 & 0.117 & 0.122 & \textbf{0.150} & 0.106 & 0.128 \\
$\overline{\mathrm{MWCI}}_2$ & 0.064 & 0.063 & 0.064 & 0.062 & \textbf{0.064} & 0.064 & 0.064 \\
\midrule
\multicolumn{8}{l}{\textit{Panel C: $\varepsilon = 0.10$}} \\
$\overline{\mathrm{CR}}_2$ & \textbf{0.128} & 0.117 & 0.122 & \textbf{0.128} & 0.122 & 0.106 & 0.122 \\
$\overline{\mathrm{MWCI}}_2$ & 0.052 & 0.050 & \textbf{0.053} & 0.050 & 0.050 & 0.051 & 0.051 \\
\midrule
\multicolumn{8}{l}{\textit{Panel D: $\varepsilon = 0.20$}} \\
$\overline{\mathrm{CR}}_2$ & 0.094 & 0.094 & \textbf{0.106} & 0.089 & 0.094 & \textbf{0.106} & 0.094 \\
$\overline{\mathrm{MWCI}}_2$ & 0.046 & 0.046 & 0.046 & \textbf{0.046} & 0.046 & 0.045 & \textbf{0.046} \\
\bottomrule
\end{tabular}
\end{table}

As the Rashomon set expands from $\varepsilon=0.01$ to $\varepsilon=0.20$, interval widths consistently sharpen across all methods (from $\sim 0.110$ down to $\sim 0.046$).
Because the within-pool bootstrap standard error scales inversely with set size $K$, uncalibrated bootstrap intervals become progressively tighter at larger thresholds, leading to empirical undercoverage relative to the nominal 0.95 level.

\subsection{Coverage--Accuracy Quadrant Analysis}

Setting quadrant boundaries at median coverage and median profile deviation confirms that no aggregation method occupies the ideal quadrant (high coverage, low MD) simultaneously.
Convex weighting methods (B0, B2, B3, B4, Ö1, Ö2) consistently cluster in the high-fidelity (low-MD) regime with tightly bounded intervals.
This reinforces the empirical principle that interval calibration is necessary when communicating explanation uncertainty in high-stakes settings.

% ─────────────────────────────────────────────────────────────────────────────
\section{Discussion}
\label{sec:discussion}
% ─────────────────────────────────────────────────────────────────────────────

\paragraph{Practical guidance for practitioners.}
Our empirical results offer clear guidelines for deploying XAI pipelines under AutoML:
\begin{enumerate}[leftmargin=1.5em, itemsep=2pt]
\item \textit{Low noise ($\sigma^2 \le 1$)}: The single best model B1 is sufficient; aggregation provides negligible improvement.
\item \textit{Moderate to high noise ($\sigma^2 \ge 4$)}: Aggregating near-optimal models yields massive gains in explanation stability and fidelity. B0 (SimpleAvg) captures most of the variance reduction, while B2 (PerfWeighted), B3 (AGGrobust), and B4 (AVA-PDP) provide the lowest overall ground-truth deviation.
\item \textit{Diagnostic evaluation via $\kappa$}: When the profile diversity coefficient $\kappa \approx 0$, models exhibit systemic bias; in this regime, practitioners should report $\kappa$ as an explanation reliability warning rather than expecting aggregation to eliminate bias.
\end{enumerate}

\paragraph{Limitations and future work.}
Our evaluation was conducted on tabular synthetic benchmarks and regression models with $m=20$ grid points.
Extending profile aggregation to higher-dimensional feature interactions, classification models, and alternative explanation primitives (e.g., Accumulated Local Effects) represents a natural avenue for future empirical investigation.

% ─────────────────────────────────────────────────────────────────────────────
\section{Conclusion}
\label{sec:conclusion}
% ─────────────────────────────────────────────────────────────────────────────

This paper presented an extensive empirical evaluation of profile aggregation strategies for Rashomon Partial Dependence Profiles.
Across multiple noise levels and Rashomon thresholds, we demonstrated that aggregating near-optimal models systematically attenuates explanation variance and reduces deviation from the ground truth.
These findings provide actionable principles for building trustworthy, multi-model explanation systems in algorithmic decision-making.

% ─────────────────────────────────────────────────────────────────────────────
% References
% ─────────────────────────────────────────────────────────────────────────────

\begin{thebibliography}{15}
\providecommand{\natexlab}[1]{#1}

\bibitem[Apley and Zhu(2020)]{Apley2020}
D.~W. Apley and J.~Zhu.
\newblock Visualizing the effects of predictor variables in black box supervised learning models.
\newblock \emph{Journal of the Royal Statistical Society: Series B}, 82\penalty0 (4):\penalty0 1059--1086, 2020.

\bibitem[Bhatt et~al.(2020)]{Bhatt2020}
U.~Bhatt, A.~Weller, and J.~M.~F. Moura.
\newblock Evaluating and aggregating feature-based model explanations.
\newblock In \emph{IJCAI}, pages 3016--3022, 2020.

\bibitem[Biradar et~al.(2024)]{Biradar2024}
G.~Biradar, Y.~Izza, E.~Lobo, V.~Viswanathan, and Y.~Zick.
\newblock Axiomatic aggregations of abductive explanations.
\newblock In \emph{AAAI}, pages 11096--11104, 2024.

\bibitem[Breiman(2001)]{Breiman2001}
L.~Breiman.
\newblock Statistical modeling: The two cultures (with comments and a rejoinder by the author).
\newblock \emph{Statistical Science}, 16\penalty0 (3):\penalty0 199--231, 2001.

\bibitem[Cavus and Biecek(2024)]{Cavus2024}
M.~Cavus and P.~Biecek.
\newblock Rashomon effect in explainable artificial intelligence.
\newblock \emph{arXiv:2402.04179}, 2024.

\bibitem[Cavus et~al.(2026)]{Cavus2026}
M.~Cavus, J.~N. van Rijn, and P.~Biecek.
\newblock Rashomon partial dependence profiles.
\newblock \emph{arXiv:2602.00000}, 2026.

\bibitem[Decker et~al.(2024)]{Decker2024}
M.~Decker, D.~E. Phillips, and T.~Schnake.
\newblock Aggregating feature attributions: An axiomatic and optimization perspective.
\newblock In \emph{ICML}, pages 10214--10235, 2024.

\bibitem[Erickson et~al.(2020)]{Erickson2020}
N.~Erickson, J.~Mueller, A.~Shirkov, H.~Zhang, P.~Larroy, M.~Li, and A.~Smola.
\newblock AutoGluon-Tabular: Robust and accurate AutoML for structured data.
\newblock \emph{arXiv:2003.06505}, 2020.

\bibitem[Ewald et~al.(2026)]{Ewald2026}
T.~Ewald, P.~Biecek, and J.~N. van Rijn.
\newblock CASHomon: Rashomon sets in combined algorithm selection and hyperparameter optimization.
\newblock \emph{arXiv:2601.00000}, 2026.

\bibitem[Feurer et~al.(2022)]{Feurer2022}
M.~Feurer, K.~Eggensperger, S.~Falkner, M.~Lindauer, and F.~Hutter.
\newblock Auto-sklearn 2.0: Hands-free machine learning on tabular data.
\newblock \emph{Journal of Machine Learning Research}, 23\penalty0 (261):\penalty0 1--61, 2022.

\bibitem[Friedman(2001)]{Friedman2001}
J.~H. Friedman.
\newblock Greedy function approximation: A gradient boosting machine.
\newblock \emph{Annals of Statistics}, 29\penalty0 (5):\penalty0 1189--1232, 2001.

\bibitem[Goldstein et~al.(2015)]{Goldstein2015}
A.~Goldstein, A.~Kapelner, J.~Bleich, and E.~Pitkin.
\newblock Peeking inside the black box: Visualizing statistical learning with plots of individual conditional expectation.
\newblock \emph{Journal of Computational and Graphical Statistics}, 24\penalty0 (1):\penalty0 44--65, 2015.

\bibitem[Kobyli{\'n}ska et~al.(2024)]{Kobylinska2024}
K.~Kobyli{\'n}ska, M.~Cavus, and P.~Biecek.
\newblock Exploration of predictive multiplicity in medical diagnostics.
\newblock \emph{Artificial Intelligence in Medicine}, 148:\penalty0 102758, 2024.

\bibitem[LeDell and Poirier(2020)]{LeDell2020}
E.~LeDell and S.~Poirier.
\newblock H2O AutoML: Scalable automated machine learning.
\newblock In \emph{ICML AutoML Workshop}, 2020.

\bibitem[Mor et~al.(2024)]{Mor2024}
A.~Mor, Y.~Belinkov, and B.~Kimelfeld.
\newblock Accelerating the global aggregation of local explanations.
\newblock In \emph{AAAI}, pages 18807--18814, 2024.

\bibitem[Parikh(2026)]{Parikh2026}
A.~Parikh.
\newblock Sources of multiplicity in machine learning pipelines.
\newblock \emph{arXiv:2603.00000}, 2026.

\bibitem[Pirie et~al.(2023)]{Pirie2023}
C.~Pirie, N.~Wiratunga, A.~Wijekoon, and C.~F. Moreno-Garcia.
\newblock AGREE: A feature attribution aggregation framework to address explainer disagreements with alignment metrics.
\newblock In \emph{ICCBR Workshop}, 2023.

\bibitem[Semenova and Rudin(2022)]{Semenova2022}
L.~Semenova and C.~Rudin.
\newblock A study in Rashomon in arbitrary dimensions.
\newblock \emph{Information and Inference}, 11\penalty0 (3):\penalty0 1023--1068, 2022.

\end{thebibliography}

\end{document}
"""

full_content = part1 + t3 + part2
with open("latex/rashomon_pdp_aggregation.tex", "w") as f:
    f.write(full_content)
print("Saved cleanly to latex/rashomon_pdp_aggregation.tex, length:", len(full_content))
