<!-- mdlint off(HEADERS_TOO_MANY_H1) -->

# Version 0.52.0

## Major Features and Improvements

*   Adds official support for Python 3.12 and 3.13.
*   Drops support for Python 3.9.
*   Depends on `tensorflow>=2.21.0,<2.22.0`.
*   Depends on `protobuf>=6.31.1,<7.0.0`.
*   Depends on `pyarrow>14`.
*   Updates the minimum Bazel version required to build TFMA to 7.4.1.

## Bug fixes and other Changes

*   **Pickling Stability Architecture**:
    *   Refactored core test modules (`rouge_test.py`, `stats_test.py`, `model_util_test.py`, `confidence_intervals_util_test.py`, `metrics_plots_and_validations_evaluator_test.py`) to use a new **class-based matcher architecture**.
    *   Replaced nested closures with module-level classes (e.g., `CheckResult`, `CheckResultMean`) to ensure full serializability for `PrismRunner` on Python 3.13.
    *   Removed `self` (test instance) capture in Beam matchers to resolve `RuntimeError: Unable to pickle fn` during distributed execution.
    *   Enabled `--no_save_main_session` for all Beam pipelines in the test suite to prevent unintentional serialization of the main session and shared resources.
*   **NumPy 2.0 & Python 3.13 Compatibility**:
    *   Standardized on safe scalar extraction by replacing `float(ndarray)` with `.item()` or `metric_util.safe_to_scalar` in aggregation, attributions, calibration, flip metrics, and NDCG modules to resolve `TypeError` in Beam pipelines.
    *   Fixed a batching bug in `flip_metrics.py` to correctly process all examples in a Beam batch.
    *   Implemented robust, warning-free division in AUC, PR AUC, and confusion matrix calculations using `np.divide` with `where` clauses to prevent `RuntimeWarning`.
    *   Fixed `TypeError` in `poisson_bootstrap.py` by removing redundant size argument from `poisson(1, 1)` to return a scalar.
    *   Implemented `kind='stable'` sort in `metric_util.top_k_indices` for deterministic tie-breaking.
*   **Bug Fixes and Functional Corrections**:
    *   Fixed a critical regression in `metric_util.py` where `SubKey(k=k)` incorrectly selected the first prediction instead of the requested k-th largest prediction.
    *   Fixed `UnparsedFlagAccessError` in `ModelSignaturesDoFn` tests by removing direct `absl.flags` access in pickling-sensitive contexts.
    *   Removed obsolete `@unittest.expectedFailure` decorators from tests that are now passing in the stabilized environment.
    *   Fixed various indentation and syntax errors in utility tests.
    *   Improved virtual environment relocation strategy to resolve Bazel sandbox access issues for `numpy` and other C-extension headers.
    *   Fixed `false_omission_rate` in `binary_confusion_matrices.py` to return NaN when undefined, resolving proto mismatches in `confusion_matrix_plot_test.py` and `score_distribution_plot_test.py`.
    *   Fixed `NotFoundError` in `model_eval_lib_test.py` by ensuring temporary directories exist before writing files using `tf.io.gfile.makedirs`.
    *   Added missing `numpy` imports in Beam-based modules to fix `NameError` regressions.

## Breaking Changes
*   Python 3.9 is no longer supported. The minimum supported Python version is now 3.10.

## Deprecations
*   N/A
