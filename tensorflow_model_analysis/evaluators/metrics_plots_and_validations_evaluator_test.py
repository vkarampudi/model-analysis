# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Test for MetricsPlotsAndValidationsEvaluator with different metrics."""

import os

import apache_beam as beam
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.testing import util
from google.protobuf import text_format
from tensorflow_metadata.proto.v0 import schema_pb2
from tfx_bsl.tfxio import tensor_adapter, test_util

from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import model_eval_lib
from tensorflow_model_analysis.evaluators import metrics_plots_and_validations_evaluator
from tensorflow_model_analysis.extractors import (
    example_weights_extractor,
    features_extractor,
    labels_extractor,
    materialized_predictions_extractor,
    predictions_extractor,
    slice_key_extractor,
    unbatch_extractor,
)
from tensorflow_model_analysis.metrics import (
    attributions,
    binary_confusion_matrices,
    calibration,
    calibration_plot,
    confusion_matrix_plot,
    metric_specs,
    metric_types,
)
from tensorflow_model_analysis.proto import config_pb2, validation_result_pb2
from tensorflow_model_analysis.utils import test_util as testutil
from tensorflow_model_analysis.utils import util as tfma_util
from tensorflow_model_analysis.utils.keras_lib import tf_keras

_TF_MAJOR_VERSION = int(tf.version.VERSION.split(".")[0])


from tensorflow_model_analysis.api import types

def _is_close(actual, expected):
    if isinstance(actual, types.ValueWithTDistribution):
        actual = actual.unsampled_value
    if actual is None:
        return expected is None
    return np.isclose(actual, expected)


def _check_metrics_keras_diff(
    weighted_example_count_key, label_key, prediction_key, expected_prediction_value
):
    def check_metrics(got):
        if len(got) != 1:
            raise ValueError(f"Expected 1 result, got {len(got)}")
        got_slice_key, got_metrics = got[0]
        if got_slice_key != ():
            raise ValueError(f"Expected empty slice key, got {got_slice_key}")
        if not _is_close(got_metrics.get(weighted_example_count_key), 0):
            raise ValueError(
                f"Unexpected weighted_example_count: {got_metrics.get(weighted_example_count_key)}"
            )
        if not _is_close(got_metrics.get(label_key), 0):
            raise ValueError(f"Unexpected label_key: {got_metrics.get(label_key)}")
        if not _is_close(
            got_metrics.get(prediction_key), expected_prediction_value
        ):
            raise ValueError(
                f"Unexpected prediction_key: {got_metrics.get(prediction_key)}, expected {expected_prediction_value}"
            )

    return check_metrics


def _check_attributions(expected_attributions):
    def check_attributions(got):
        if len(got) != 1:
            raise ValueError(f"Expected 1 result, got {len(got)}")
        got_slice_key, got_attributions = got[0]
        if got_slice_key != ():
            raise ValueError(f"Expected empty slice key, got {got_slice_key}")
        total_attributions_key = metric_types.MetricKey(name="total_attributions")
        if total_attributions_key not in got_attributions:
            raise ValueError("total_attributions_key not in results")
        actual = got_attributions[total_attributions_key]
        for k, v in expected_attributions.items():
            if not np.isclose(actual[k], v):
                raise ValueError(f"Unexpected attribution for {k}: {actual[k]}, expected {v}")

    return check_attributions


def _check_metrics_keras_ingraph(
    example_count_key,
    weighted_example_count_key,
    label_key,
    label_unweighted_key,
    binary_accuracy_key,
    expected_values,
):
    def check_metrics(got):
        if len(got) != 1:
            raise ValueError(f"Expected 1 result, got {len(got)}")
        got_slice_key, got_metrics = got[0]
        if got_slice_key != ():
             raise ValueError(f"Expected empty slice key, got {got_slice_key}")
        if binary_accuracy_key not in got_metrics:
            raise ValueError(f"binary_accuracy_key {binary_accuracy_key} not in results")
        for k, v in expected_values.items():
            if not np.isclose(got_metrics[k], v):
                 raise ValueError(f"Unexpected value for {k}: {got_metrics[k]}, expected {v}")

    return check_metrics


def _check_cross_slice_keys(expected_slice_keys):
    def check_result(got_sliced_metrics):
        actual_slice_keys = [k for k, _ in got_sliced_metrics]
        if len(expected_slice_keys) != len(actual_slice_keys) or set(
            expected_slice_keys
        ) != set(actual_slice_keys):
            raise ValueError(
                f"Expected {expected_slice_keys}, got {actual_slice_keys}"
            )

    return check_result


class MetricsPlotsAndValidationsEvaluatorTest(
    testutil.TensorflowModelAnalysisTest, parameterized.TestCase
):
    def _getExportDir(self):
        return os.path.join(self._getTempDir(), "export_dir")

    def _getBaselineDir(self):
        return os.path.join(self._getTempDir(), "baseline_export_dir")

    def _build_keras_model(self, model_name, model_dir, mul):
        input_layer = tf_keras.layers.Input(shape=(1,), name="input_1")
        output_layer = tf_keras.layers.Lambda(
            lambda x, mul: x * mul, output_shape=(1,), arguments={"mul": mul}
        )(input_layer)
        model = tf_keras.models.Model([input_layer], output_layer)
        model.compile(
            optimizer=tf_keras.optimizers.Adam(lr=0.001),
            loss=tf_keras.losses.BinaryCrossentropy(name="loss"),
            metrics=["accuracy"],
        )
        model.save(model_dir, save_format="tf")
        return self.createTestEvalSharedModel(
            model_name=model_name, model_path=model_dir
        )

    def testFilterAndSeparateComputations(self):
        eval_config = config_pb2.EvalConfig(
            model_specs=[
                config_pb2.ModelSpec(name="candidate", label_key="tips"),
                config_pb2.ModelSpec(
                    name="baseline", label_key="tips", is_baseline=True
                ),
            ],
            cross_slicing_specs=[config_pb2.CrossSlicingSpec()],
        )
        metrics_specs = metric_specs.specs_from_metrics(
            [
                tf_keras.metrics.BinaryAccuracy(name="accuracy"),
                tf_keras.metrics.AUC(name="auc", num_thresholds=10000),
                tf_keras.metrics.AUC(
                    name="auc_precison_recall", curve="PR", num_thresholds=10000
                ),
                tf_keras.metrics.Precision(name="precision"),
                tf_keras.metrics.Recall(name="recall"),
                calibration.MeanLabel(name="mean_label"),
                calibration.MeanPrediction(name="mean_prediction"),
                calibration.Calibration(name="calibration"),
                confusion_matrix_plot.ConfusionMatrixPlot(name="confusion_matrix_plot"),
                calibration_plot.CalibrationPlot(name="calibration_plot"),
            ],
            model_names=["candidate", "baseline"],
            binarize=config_pb2.BinarizationOptions(class_ids={"values": [0, 5]}),
        )
        computations = metric_specs.to_computations(
            metrics_specs, eval_config=eval_config
        )
        non_derived, derived, _, ci_derived = (
            metrics_plots_and_validations_evaluator._filter_and_separate_computations(
                computations
            )
        )
        # 2 models x 2 classes x _binary_confusion_matrix_[0.5]_100,
        # 2 models x 2 classes x _CalibrationHistogramCombiner
        # 2 models x 2 classes x _calibration_historgram_27
        # 2 models x 2 classes x _CompilableMetricsCombiner,
        # 2 models x 2 classes x _WeightedLabelsPredictionsExamplesCombiner,
        # 4 models x _ExampleCountCombiner
        self.assertLen(non_derived, 16)
        # 2 models x 2 classes x _binary_confusion_matrices_[0.5],
        # 2 models x 2 classes x _binary_confusion_matrices_10000
        # 2 models x 2 classes x _binary_confusion_matrices_confusion_matrix_plot
        # 2 models x 2 classes x precision
        # 2 models x 2 classes x recall
        # 2 models x 2 classes x calibration
        # 2 models x 2 classes x auc_precision_recall
        # 2 models x 2 classes x mean_prediction
        # 2 models x 2 classes x mean_label
        # 2 models x 2 classes x confusion_matrix_plot
        # 2 models x 2 classes x calibration_plot
        # 2 models x 2 classes x auc
        # 2 models x 2 classes x accuracy
        self.assertLen(derived, 52)
        # None of the metric has CIDerivedMetricComputation.
        self.assertEmpty(ci_derived)

    def testFilterAndSeparateComputationsWithCIDerivedMetrics(self):
        def derived_metric_fn():
            pass

        def ci_derived_fn():
            pass

        computations = [
            metric_types.DerivedMetricComputation(
                [metric_types.MetricKey("key1")], derived_metric_fn
            ),
            metric_types.CIDerivedMetricComputation(
                [metric_types.MetricKey("key1")], ci_derived_fn
            ),
            metric_types.CIDerivedMetricComputation(
                [metric_types.MetricKey("key1")], ci_derived_fn
            ),
        ]
        _, derived, _, ci_derived = (
            metrics_plots_and_validations_evaluator._filter_and_separate_computations(
                computations
            )
        )

        self.assertLen(derived, 1)
        self.assertLen(ci_derived, 1)

    def testEvaluateWithKerasAndDiffMetrics(self):
        model_dir, baseline_dir = self._getExportDir(), self._getBaselineDir()
        eval_shared_model = self._build_keras_model("candidate", model_dir, mul=0)
        baseline_eval_shared_model = self._build_keras_model(
            "baseline", baseline_dir, mul=1
        )

        schema = text_format.Parse(
            """
        tensor_representation_group {
          key: ""
          value {
            tensor_representation {
              key: "input_1"
              value {
                dense_tensor {
                  column_name: "input_1"
                  shape { dim { size: 1 } }
                }
              }
            }
          }
        }
        feature {
          name: "input_1"
          type: FLOAT
        }
        feature {
          name: "label"
          type: FLOAT
        }
        feature {
          name: "example_weight"
          type: FLOAT
        }
        feature {
          name: "extra_feature"
          type: BYTES
        }
        """,
            schema_pb2.Schema(),
        )
        tfx_io = test_util.InMemoryTFExampleRecord(
            schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN
        )
        tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
            arrow_schema=tfx_io.ArrowSchema(),
            tensor_representations=tfx_io.TensorRepresentations(),
        )

        examples = [
            self._makeExample(
                input_1=0.0,
                label=1.0,
                example_weight=1.0,
                extra_feature="non_model_feature",
            ),
            self._makeExample(
                input_1=1.0,
                label=0.0,
                example_weight=0.5,
                extra_feature="non_model_feature",
            ),
        ]

        eval_config = config_pb2.EvalConfig(
            model_specs=[
                config_pb2.ModelSpec(
                    name="candidate",
                    label_key="label",
                    example_weight_key="example_weight",
                ),
                config_pb2.ModelSpec(
                    name="baseline",
                    label_key="label",
                    example_weight_key="example_weight",
                    is_baseline=True,
                ),
            ],
            slicing_specs=[config_pb2.SlicingSpec()],
            metrics_specs=metric_specs.specs_from_metrics(
                [
                    calibration.MeanLabel("mean_label"),
                    calibration.MeanPrediction("mean_prediction"),
                ],
                model_names=["candidate", "baseline"],
            ),
        )

        eval_shared_models = [eval_shared_model, baseline_eval_shared_model]
        extractors = [
            features_extractor.FeaturesExtractor(
                eval_config=eval_config,
                tensor_representations=tensor_adapter_config.tensor_representations,
            ),
            labels_extractor.LabelsExtractor(eval_config),
            example_weights_extractor.ExampleWeightsExtractor(eval_config),
            predictions_extractor.PredictionsExtractor(
                eval_shared_model=eval_shared_models, eval_config=eval_config
            ),
            unbatch_extractor.UnbatchExtractor(),
            slice_key_extractor.SliceKeyExtractor(eval_config=eval_config),
        ]
        evaluators = [
            metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
                eval_config=eval_config, eval_shared_model=eval_shared_models
            )
        ]

        options = PipelineOptions(flags=["--no_save_main_session"])
        with beam.Pipeline(options=options) as pipeline:
            # pylint: disable=no-value-for-parameter
            metrics = (
                pipeline
                | "Create" >> beam.Create([e.SerializeToString() for e in examples])
                | "BatchExamples" >> tfx_io.BeamSource()
                | "InputsToExtracts" >> model_eval_lib.BatchedInputsToExtracts()
                | "ExtractAndEvaluate"
                >> model_eval_lib.ExtractAndEvaluate(
                    extractors=extractors, evaluators=evaluators
                )
            )

            # pylint: enable=no-value-for-parameter

            weighted_example_count_key = metric_types.MetricKey(
                name="weighted_example_count",
                model_name="candidate",
                is_diff=True,
                example_weighted=True,
            )
            prediction_key = metric_types.MetricKey(
                name="mean_prediction",
                model_name="candidate",
                is_diff=True,
                example_weighted=True,
            )
            label_key = metric_types.MetricKey(
                name="mean_label",
                model_name="candidate",
                is_diff=True,
                example_weighted=True,
            )
            expected_prediction_value = 0 - (0 * 1 + 1 * 0.5) / (1 + 0.5)

            util.assert_that(
                metrics[constants.METRICS_KEY],
                _check_metrics_keras_diff(
                    weighted_example_count_key,
                    label_key,
                    prediction_key,
                    expected_prediction_value,
                ),
                label="metrics",
            )

    def testEvaluateWithAttributions(self):
        eval_config = config_pb2.EvalConfig(
            model_specs=[config_pb2.ModelSpec()],
            metrics_specs=[
                config_pb2.MetricsSpec(
                    metrics=[
                        config_pb2.MetricConfig(
                            class_name=attributions.TotalAttributions().__class__.__name__
                        )
                    ]
                )
            ],
            options=config_pb2.Options(
                disabled_outputs={"values": ["eval_config_pb2.json"]}
            ),
        )
        extractors = [slice_key_extractor.SliceKeyExtractor()]
        evaluators = [
            metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
                eval_config=eval_config
            )
        ]

        example1 = {
            "labels": None,
            "predictions": None,
            "example_weights": np.array(1.0),
            "features": {},
            "attributions": {"feature1": 1.1, "feature2": 1.2},
        }
        example2 = {
            "labels": None,
            "predictions": None,
            "example_weights": np.array(1.0),
            "features": {},
            "attributions": {"feature1": 2.1, "feature2": 2.2},
        }
        example3 = {
            "labels": None,
            "predictions": None,
            "example_weights": np.array(1.0),
            "features": {},
            "attributions": {
                "feature1": np.array([3.1]),
                "feature2": np.array([3.2]),
            },
        }

        options = PipelineOptions(flags=["--no_save_main_session"])
        with beam.Pipeline(options=options) as pipeline:
            # pylint: disable=no-value-for-parameter
            results = (
                pipeline
                | "Create" >> beam.Create([example1, example2, example3])
                | "ExtractEvaluate"
                >> model_eval_lib.ExtractAndEvaluate(
                    extractors=extractors, evaluators=evaluators
                )
            )

            # pylint: enable=no-value-for-parameter

            expected_attributions = {
                "feature1": 1.1 + 2.1 + 3.1,
                "feature2": 1.2 + 2.2 + 3.2,
            }

            util.assert_that(
                results[constants.ATTRIBUTIONS_KEY],
                _check_attributions(expected_attributions),
                label="attributions",
            )

    def testEvaluateWithJackknifeAndDiffMetrics(self):
        model_dir, baseline_dir = self._getExportDir(), self._getBaselineDir()
        eval_shared_model = self._build_keras_model("candidate", model_dir, mul=0)
        baseline_eval_shared_model = self._build_keras_model(
            "baseline", baseline_dir, mul=1
        )

        options = config_pb2.Options()
        options.compute_confidence_intervals.value = True
        options.confidence_intervals.method = (
            config_pb2.ConfidenceIntervalOptions.JACKKNIFE
        )

        eval_config = config_pb2.EvalConfig(
            model_specs=[
                config_pb2.ModelSpec(
                    name="candidate",
                    label_key="label",
                    example_weight_key="example_weight",
                ),
                config_pb2.ModelSpec(
                    name="baseline",
                    label_key="label",
                    example_weight_key="example_weight",
                    is_baseline=True,
                ),
            ],
            slicing_specs=[config_pb2.SlicingSpec()],
            metrics_specs=metric_specs.specs_from_metrics(
                [
                    calibration.MeanLabel("mean_label"),
                    calibration.MeanPrediction("mean_prediction"),
                ],
                model_names=["candidate", "baseline"],
            ),
            options=options,
        )

        eval_shared_models = {
            "candidate": eval_shared_model,
            "baseline": baseline_eval_shared_model,
        }

        schema = text_format.Parse(
            """
        tensor_representation_group {
          key: ""
          value {
            tensor_representation {
              key: "input_1"
              value {
                dense_tensor {
                  column_name: "input_1"
                  shape { dim { size: 1 } }
                }
              }
            }
          }
        }
        feature {
          name: "input_1"
          type: FLOAT
        }
        feature {
          name: "label"
          type: FLOAT
        }
        feature {
          name: "example_weight"
          type: FLOAT
        }
        feature {
          name: "extra_feature"
          type: BYTES
        }
        """,
            schema_pb2.Schema(),
        )
        tfx_io = test_util.InMemoryTFExampleRecord(
            schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN
        )
        tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
            arrow_schema=tfx_io.ArrowSchema(),
            tensor_representations=tfx_io.TensorRepresentations(),
        )

        examples = [
            self._makeExample(
                input_1=0.0,
                label=1.0,
                example_weight=1.0,
                extra_feature="non_model_feature",
            ),
            self._makeExample(
                input_1=1.0,
                label=0.0,
                example_weight=0.5,
                extra_feature="non_model_feature",
            ),
        ]

        extractors = [
            features_extractor.FeaturesExtractor(
                eval_config=eval_config,
                tensor_representations=tensor_adapter_config.tensor_representations,
            ),
            labels_extractor.LabelsExtractor(eval_config),
            example_weights_extractor.ExampleWeightsExtractor(eval_config),
            predictions_extractor.PredictionsExtractor(
                eval_shared_model=eval_shared_models, eval_config=eval_config
            ),
            unbatch_extractor.UnbatchExtractor(),
            slice_key_extractor.SliceKeyExtractor(eval_config=eval_config),
        ]
        evaluators = [
            metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
                eval_config=eval_config, eval_shared_model=eval_shared_models
            )
        ]

        pb_options = PipelineOptions(flags=["--no_save_main_session"])
        with beam.Pipeline(options=pb_options) as pipeline:
            # pylint: disable=no-value-for-parameter
            metrics = (
                pipeline
                | "Create"
                >> beam.Create([e.SerializeToString() for e in examples * 1000])
                | "BatchExamples" >> tfx_io.BeamSource()
                | "InputsToExtracts" >> model_eval_lib.BatchedInputsToExtracts()
                | "ExtractAndEvaluate"
                >> model_eval_lib.ExtractAndEvaluate(
                    extractors=extractors, evaluators=evaluators
                )
            )

            # pylint: enable=no-value-for-parameter

            weighted_example_count_key = metric_types.MetricKey(
                name="weighted_example_count",
                model_name="candidate",
                is_diff=True,
                example_weighted=True,
            )
            prediction_key = metric_types.MetricKey(
                name="mean_prediction",
                model_name="candidate",
                is_diff=True,
                example_weighted=True,
            )
            label_key = metric_types.MetricKey(
                name="mean_label",
                model_name="candidate",
                is_diff=True,
                example_weighted=True,
            )
            expected_prediction_value = 0 - (0 * 1 + 1 * 0.5) / (1 + 0.5)

            util.assert_that(
                metrics[constants.METRICS_KEY],
                _check_metrics_keras_diff(
                    weighted_example_count_key,
                    label_key,
                    prediction_key,
                    expected_prediction_value,
                ),
            )

    @parameterized.named_parameters(
        ("compiled_metrics", False),
        ("evaluate", True),
    )
    def testEvaluateWithKerasModelWithInGraphMetrics(self, add_custom_metrics):
        # Custom metrics not supported in TFv1
        if _TF_MAJOR_VERSION < 2:
            add_custom_metrics = False

        input1 = tf_keras.layers.Input(shape=(1,), name="input_1")
        input2 = tf_keras.layers.Input(shape=(1,), name="input_2")
        inputs = [input1, input2]
        input_layer = tf_keras.layers.concatenate(inputs)
        output_layer = tf_keras.layers.Dense(
            1, activation=tf.nn.sigmoid, name="output"
        )(input_layer)
        model = tf_keras.models.Model(inputs, output_layer)
        # The model.evaluate API is used when custom metrics are used. Otherwise
        # model.compiled_metrics is used.
        if add_custom_metrics:
            model.add_metric(tf.reduce_sum(input_layer), name="custom")
        model.compile(
            optimizer=tf_keras.optimizers.Adam(lr=0.001),
            loss=tf_keras.losses.BinaryCrossentropy(name="loss"),
            metrics=[tf_keras.metrics.BinaryAccuracy(name="binary_accuracy")],
        )

        export_dir = self._getExportDir()
        model.save(export_dir, save_format="tf")

        eval_config = config_pb2.EvalConfig(
            model_specs=[
                config_pb2.ModelSpec(
                    label_key="label", example_weight_key="example_weight"
                )
            ],
            slicing_specs=[config_pb2.SlicingSpec()],
            metrics_specs=metric_specs.specs_from_metrics(
                [calibration.MeanLabel("mean_label")],
                unweighted_metrics=[
                    tf_keras.metrics.BinaryAccuracy(name="binary_accuracy"),
                    calibration.MeanLabel("mean_label"),
                ],
            ),
        )
        eval_shared_model = self.createTestEvalSharedModel(model_path=export_dir)

        examples = [
            self._makeExample(
                input_1=0.0,
                input_2=1.0,
                label=1.0,
                example_weight=1.0,
                extra_feature="non_model_feature",
            ),
            self._makeExample(
                input_1=1.0,
                input_2=0.0,
                label=0.0,
                example_weight=0.5,
                extra_feature="non_model_feature",
            ),
        ]

        schema = text_format.Parse(
            """
        tensor_representation_group {
          key: ""
          value {
            tensor_representation {
              key: "input_1"
              value {
                dense_tensor {
                  column_name: "input_1"
                  shape { dim { size: 1 } }
                }
              }
            }
            tensor_representation {
              key: "input_2"
              value {
                dense_tensor {
                  column_name: "input_2"
                  shape { dim { size: 1 } }
                }
              }
            }
          }
        }
        feature {
          name: "input_1"
          type: FLOAT
        }
        feature {
          name: "input_2"
          type: FLOAT
        }
        feature {
          name: "label"
          type: FLOAT
        }
        feature {
          name: "example_weight"
          type: FLOAT
        }
        feature {
          name: "extra_feature"
          type: BYTES
        }
        """,
            schema_pb2.Schema(),
        )
        tfx_io = test_util.InMemoryTFExampleRecord(
            schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN
        )
        tensor_adapter_config = tensor_adapter.TensorAdapterConfig(
            arrow_schema=tfx_io.ArrowSchema(),
            tensor_representations=tfx_io.TensorRepresentations(),
        )
        extractors = [
            features_extractor.FeaturesExtractor(
                eval_config=eval_config,
                tensor_representations=tensor_adapter_config.tensor_representations,
            ),
            labels_extractor.LabelsExtractor(eval_config),
            example_weights_extractor.ExampleWeightsExtractor(eval_config),
            predictions_extractor.PredictionsExtractor(
                eval_shared_model=eval_shared_model, eval_config=eval_config
            ),
            unbatch_extractor.UnbatchExtractor(),
            slice_key_extractor.SliceKeyExtractor(eval_config=eval_config),
        ]
        evaluators = [
            metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
                eval_config=eval_config, eval_shared_model=eval_shared_model
            )
        ]
        options = PipelineOptions(flags=["--no_save_main_session"])
        with beam.Pipeline(options=options) as pipeline:
            # pylint: disable=no-value-for-parameter
            metrics = (
                pipeline
                | "Create" >> beam.Create([e.SerializeToString() for e in examples])
                | "BatchExamples" >> tfx_io.BeamSource()
                | "InputsToExtracts" >> model_eval_lib.BatchedInputsToExtracts()
                | "ExtractAndEvaluate"
                >> model_eval_lib.ExtractAndEvaluate(
                    extractors=extractors, evaluators=evaluators
                )
            )

            # pylint: enable=no-value-for-parameter

            example_count_key = metric_types.MetricKey(name="example_count")
            weighted_example_count_key = metric_types.MetricKey(
                name="weighted_example_count", example_weighted=True
            )
            label_key = metric_types.MetricKey(name="mean_label", example_weighted=True)
            label_unweighted_key = metric_types.MetricKey(
                name="mean_label", example_weighted=False
            )
            binary_accuracy_key = metric_types.MetricKey(
                name="binary_accuracy", example_weighted=False
            )
            expected_values = {
                example_count_key: 2,
                weighted_example_count_key: 1.0 + 0.5,
                label_key: (1.0 * 1.0 + 0.0 * 0.5) / (1.0 + 0.5),
                label_unweighted_key: (1.0 + 0.0) / (1.0 + 1.0),
            }

            util.assert_that(
                metrics[constants.METRICS_KEY],
                _check_metrics_keras_ingraph(
                    example_count_key,
                    weighted_example_count_key,
                    label_key,
                    label_unweighted_key,
                    binary_accuracy_key,
                    expected_values,
                ),
                label="metrics",
            )

    def testAddCrossSliceMetricsMatchAll(self):
        overall_slice_key = ()
        slice_key1 = (("feature", 1),)
        slice_key2 = (("feature", 2),)
        slice_key3 = (("feature", 3),)
        metrics_dict = {}
        sliced_metrics = [
            (overall_slice_key, metrics_dict),
            (slice_key1, metrics_dict),
            (slice_key2, metrics_dict),
            (slice_key3, metrics_dict),
        ]
        options = PipelineOptions(flags=["--no_save_main_session"])
        with beam.Pipeline(options=options) as pipeline:
            cross_sliced_metrics = (
                pipeline
                | "CreateSlicedMetrics" >> beam.Create(sliced_metrics)
                | "AddCrossSliceMetrics"
                >> metrics_plots_and_validations_evaluator._AddCrossSliceMetrics(
                    cross_slice_specs=[
                        config_pb2.CrossSlicingSpec(baseline_spec={}, slicing_specs=[])
                    ],
                    cross_slice_computations=[],
                )
            )

            expected_slice_keys = [
                # cross slice keys
                (overall_slice_key, slice_key1),
                (overall_slice_key, slice_key2),
                (overall_slice_key, slice_key3),
                # single slice keys
                overall_slice_key,
                slice_key1,
                slice_key2,
                slice_key3,
            ]

            util.assert_that(
                cross_sliced_metrics, _check_cross_slice_keys(expected_slice_keys)
            )

    @parameterized.named_parameters(
        ("IntIsDiffable", 1, True),
        ("FloatIsDiffable", 1.0, True),
        ("NumpyFloatDtypeIsDiffable", np.array([1.0], dtype=np.float64), True),
        ("NumpyIntDtypeIsDiffable", np.array([1], dtype=np.int64), True),
        ("MessageNotDiffable", validation_result_pb2.ValidationResult(), False),
        (
            "TupleNotDiffable",
            binary_confusion_matrices.Matrices(
                thresholds=[-1e-7, 0.5, 1.0 + 1e-7],
                tp=[2.0, 1.0, 0.0],
                fp=[2.0, 0.0, 0.0],
                tn=[0.0, 2.0, 2.0],
                fn=[0.0, 1.0, 2.0],
            ),
            True,
        ),
        ("BytesNotDiffable", b"some bytes", False),
        ("NumpyObjectDtypeNotDiffable", np.array(["obj"], dtype=object), False),
    )
    def testIsMetricDiffable(self, metric_value, expected_is_diffable):
        self.assertEqual(
            expected_is_diffable,
            metrics_plots_and_validations_evaluator._is_metric_diffable(metric_value),
        )

    def testMetricsSpecsCountersInModelAgnosticMode(self):
        schema = text_format.Parse(
            """
        feature {
          name: "label"
          type: FLOAT
        }
        feature {
          name: "prediction"
          type: FLOAT
        }
        """,
            schema_pb2.Schema(),
        )

        tfx_io = test_util.InMemoryTFExampleRecord(
            schema=schema, raw_record_column_name=constants.ARROW_INPUT_COLUMN
        )

        examples = [
            self._makeExample(label=1.0, prediction=0.7),
            self._makeExample(label=0.0, prediction=0.3),
        ]

        eval_config = config_pb2.EvalConfig(
            model_specs=[
                config_pb2.ModelSpec(prediction_key="prediction", label_key="label")
            ],
            metrics_specs=[
                config_pb2.MetricsSpec(
                    metrics=[config_pb2.MetricConfig(class_name="ExampleCount")]
                )
            ],
            slicing_specs=[config_pb2.SlicingSpec()],
        )

        extractors = [
            features_extractor.FeaturesExtractor(eval_config),
            labels_extractor.LabelsExtractor(eval_config),
            example_weights_extractor.ExampleWeightsExtractor(eval_config),
            materialized_predictions_extractor.MaterializedPredictionsExtractor(
                eval_config
            ),
            unbatch_extractor.UnbatchExtractor(),
            slice_key_extractor.SliceKeyExtractor(eval_config=eval_config),
        ]
        evaluators = [
            metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
                eval_config
            )
        ]

        options = PipelineOptions(flags=["--no_save_main_session"])
        with beam.Pipeline(options=options) as pipeline:
            # pylint: disable=no-value-for-parameter
            _ = (
                pipeline
                | "Create" >> beam.Create([e.SerializeToString() for e in examples])
                | "BatchExamples" >> tfx_io.BeamSource()
                | "InputsToExtracts" >> model_eval_lib.BatchedInputsToExtracts()
                | "ExtractEvaluate"
                >> model_eval_lib.ExtractAndEvaluate(
                    extractors=extractors, evaluators=evaluators
                )
            )
        result = pipeline.run()
        result.wait_until_finish()

        metric_filter = beam.metrics.metric.MetricsFilter().with_name(
            "metric_computed_ExampleCount_v2_" + constants.MODEL_AGNOSTIC
        )
        actual_metrics_count = (
            result.metrics().query(filter=metric_filter)["counters"][0].committed
        )
        self.assertEqual(actual_metrics_count, 1)


if __name__ == "__main__":
    tf.compat.v1.enable_v2_behavior()
    tf.test.main()
