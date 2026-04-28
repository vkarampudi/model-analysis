# Copyright 2023 Google LLC
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
"""Tests for stats metrics."""

import apache_beam as beam
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from apache_beam.testing import util
from google.protobuf import text_format

import tensorflow_model_analysis as tfma
from tensorflow_model_analysis.metrics import metric_types, metric_util, stats
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.utils import test_util


def _get_examples():
    example0 = {
        "labels": None,
        "predictions": None,
        "features": {
            "example_weights": [2.2],
            "age": [18],
            "income": [50000],
        },
    }
    example1 = {
        "labels": None,
        "predictions": None,
        "features": {
            "example_weights": [6.8],
            "age": [21],
            "income": [100000],
        },
    }
    example2 = {
        "labels": None,
        "predictions": None,
        "features": {
            "example_weights": [9.2],
            "age": [50],
            "income": [300000],
        },
    }
    example3 = {
        "labels": None,
        "predictions": None,
        "features": {
            "example_weights": [6.7],
            "age": [65],
            "income": [400000],
        },
    }
    return [example0, example1, example2, example3]


def _compute_mean_metric(pipeline, computation):
    return (
        pipeline
        | "Create" >> beam.Create(_get_examples())
        | "Process" >> beam.Map(metric_util.to_standard_metric_inputs)
        | "AddSlice" >> beam.Map(lambda x: ((), x))
        | "ComputeMeanMetric" >> beam.CombinePerKey(computation.combiner)
    )


def _check_got(got, computation):
    if len(got) != 1:
        raise ValueError(f"Expected 1, got {len(got)}")
    got_slice_key, got_metrics = got[0]
    if got_slice_key != ():
        raise ValueError(f"Expected (), got {got_slice_key}")
    if computation.keys[0] not in got_metrics:
        raise ValueError(f"Expected {computation.keys[0]} in metrics")
    return got_metrics


class CheckResultMean:
    def __init__(self, computation_key, expected_metric_key, expected_mean):
        self.computation_key = computation_key
        self.expected_metric_key = expected_metric_key
        self.expected_mean = expected_mean

    def __call__(self, got):
        if len(got) != 1:
            raise ValueError(f"Expected 1, got {len(got)}")
        got_slice_key, got_metrics = got[0]
        if got_slice_key != ():
            raise ValueError(f"Expected (), got {got_slice_key}")
        if self.computation_key not in got_metrics:
            raise ValueError(f"Expected {self.computation_key} in metrics")
        if self.expected_metric_key not in got_metrics:
            raise ValueError(f"Expected {self.expected_metric_key} in metrics")

        got_mean = got_metrics[self.expected_metric_key]
        if abs(got_mean - self.expected_mean) > 1e-5:
            raise ValueError(f"Expected mean {self.expected_mean}, got {got_mean}")


class CheckResultNan:
    def __init__(self, key):
        self.key = key

    def __call__(self, got):
        if len(got) != 1:
            raise ValueError(f"Expected 1, got {len(got)}")
        got_slice_key, got_metrics = got[0]
        if got_slice_key != ():
            raise ValueError(f"Expected (), got {got_slice_key}")
        if self.key not in got_metrics:
            raise ValueError(f"Expected {self.key}")
        if not np.isnan(got_metrics[self.key]):
            raise ValueError(f"Expected NaN, got {got_metrics[self.key]}")

class CheckResultMeanEnd2End:
    def __init__(self, expected_key_age, expected_key_income, expected_result_age, expected_result_income):
        self.expected_key_age = expected_key_age
        self.expected_key_income = expected_key_income
        self.expected_result_age = expected_result_age
        self.expected_result_income = expected_result_income

    def __call__(self, got):
        if len(got) != 1:
            raise ValueError(f"Expected 1 result, got {len(got)}")
        got_slice_key, got_metrics = got[0]
        if got_slice_key != ():
            raise ValueError(f"Expected (), got {got_slice_key}")
        if len(got_metrics) != 2:
            raise ValueError(f"Expected 2 metrics, got {len(got_metrics)}")
        if self.expected_key_age not in got_metrics:
            raise ValueError(f"Expected {self.expected_key_age}")
        if self.expected_key_income not in got_metrics:
            raise ValueError(f"Expected {self.expected_key_income}")
        if abs(self.expected_result_age - got_metrics[self.expected_key_age]) > 1e-5:
            raise ValueError("Age mismatch")
        if abs(self.expected_result_income - got_metrics[self.expected_key_income]) > 1e-5:
            raise ValueError("Income mismatch")


class MeanTestValidExamples(
    test_util.TensorflowModelAnalysisTest, parameterized.TestCase
):

    @parameterized.named_parameters(
        ("Age", ["features", "age"], "mean_features.age", 38.5),
        ("Income", ["features", "income"], "mean_features.income", 212500),
    )
    def testMeanUnweighted(
        self, feature_key_path, expected_metric_key_name, expected_mean
    ):
        mean_metric_key = metric_types.MetricKey(name=expected_metric_key_name)
        mean_metric_computation = stats.Mean(feature_key_path).computations()[0]

        with beam.Pipeline(
            options=beam.options.pipeline_options.PipelineOptions(
                flags=["--no_save_main_session"]
            )
        ) as pipeline:
            result = _compute_mean_metric(pipeline, mean_metric_computation)

            util.assert_that(
                result,
                CheckResultMean(
                    mean_metric_computation.keys[0], mean_metric_key, expected_mean
                ),
                label="result",
            )

    @parameterized.named_parameters(
        ("Age", ["features", "age"], "mean_features.age", 1077.9 / 24.9),
        (
            "Income",
            ["features", "income"],
            "mean_features.income",
            6230000 / 24.9,
        ),
    )
    def testMeanWeighted(
        self, feature_key_path, expected_metric_key_name, expected_mean
    ):
        mean_metric_key = metric_types.MetricKey(
            name=expected_metric_key_name, example_weighted=True
        )
        mean_metric_computation = stats.Mean(
            feature_key_path,
            example_weights_key_path=["features", "example_weights"],
        ).computations()[0]

        with beam.Pipeline(
            options=beam.options.pipeline_options.PipelineOptions(
                flags=["--no_save_main_session"]
            )
        ) as pipeline:
            result = _compute_mean_metric(pipeline, mean_metric_computation)

            util.assert_that(
                result,
                CheckResultMean(
                    mean_metric_computation.keys[0], mean_metric_key, expected_mean
                ),
                label="result",
            )

    def testMeanName(self):
        feature_key_path = ["features", "age"]
        name = "name_to_verify_123_!@#"
        expected_mean = 38.5
        mean_metric_key = metric_types.MetricKey(name=name)
        mean_metric_computation = stats.Mean(
            feature_key_path, name=name
        ).computations()[0]

        with beam.Pipeline(
            options=beam.options.pipeline_options.PipelineOptions(
                flags=["--no_save_main_session"]
            )
        ) as pipeline:
            result = _compute_mean_metric(pipeline, mean_metric_computation)

            util.assert_that(
                result,
                CheckResultMean(
                    mean_metric_computation.keys[0], mean_metric_key, expected_mean
                ),
                label="result",
            )


class MeanTestInvalidExamples(
    test_util.TensorflowModelAnalysisTest, parameterized.TestCase
):
    def testMeanNotOneFeatureValue(self):
        # This example should cause a ValueError because
        # the feature (age) contains more than one value.
        example = {
            "labels": None,
            "predictions": None,
            "features": {
                "example_weights": [2.2],
                "age": [18, 21],
            },
        }

        feature_key_path = ["features", "age"]
        example_weights_key_path = ["features", "example_weights"]

        mean_metric_computation = stats.Mean(
            feature_key_path, example_weights_key_path=example_weights_key_path
        ).computations()[0]

        with self.assertRaisesRegex(
            (AssertionError, RuntimeError),
            r"Mean\(\) is only supported for scalar features, but found features = "
            r"\[18, 21\]",
        ):
            with beam.Pipeline(
                options=beam.options.pipeline_options.PipelineOptions(
                    flags=["--no_save_main_session"]
                )
            ) as pipeline:
                _ = (
                    pipeline
                    | "Create" >> beam.Create([example])
                    | "Process" >> beam.Map(metric_util.to_standard_metric_inputs)
                    | "AddSlice" >> beam.Map(lambda x: ((), x))
                    | "ComputeMeanMetric"
                    >> beam.CombinePerKey(mean_metric_computation.combiner)
                )

    def testMeanNotOneExampleWeight(self):
        # This example should cause a ValueError
        # because it has multiple example weights.
        example = {
            "labels": None,
            "predictions": None,
            "features": {
                "example_weights": [4.6, 8.5],
                "age": [18],
            },
        }

        feature_key_path = ["features", "age"]
        example_weights_key_path = ["features", "example_weights"]

        mean_metric_computation = stats.Mean(
            feature_key_path, example_weights_key_path=example_weights_key_path
        ).computations()[0]

        with self.assertRaisesRegex(
            (AssertionError, RuntimeError),
            r"Expected 1 \(scalar\) example weight for each example, but found "
            r"example weight = \[4.6, 8.5\]",
        ):
            with beam.Pipeline(
                options=beam.options.pipeline_options.PipelineOptions(
                    flags=["--no_save_main_session"]
                )
            ) as pipeline:
                _ = (
                    pipeline
                    | "Create" >> beam.Create([example])
                    | "Process" >> beam.Map(metric_util.to_standard_metric_inputs)
                    | "AddSlice" >> beam.Map(lambda x: ((), x))
                    | "ComputeMeanMetric"
                    >> beam.CombinePerKey(mean_metric_computation.combiner)
                )

    def testMeanExampleCountIsZero(self):
        example = {
            "labels": None,
            "predictions": None,
            "features": {
                "example_weights": [0.0],
                "age": [18],
            },
        }

        feature_key_path = ["features", "age"]
        example_weights_key_path = ["features", "example_weights"]

        mean_metric_computation = stats.Mean(
            feature_key_path, example_weights_key_path=example_weights_key_path
        ).computations()[0]
        key = mean_metric_computation.keys[0]

        with beam.Pipeline(
            options=beam.options.pipeline_options.PipelineOptions(
                flags=["--no_save_main_session"]
            )
        ) as pipeline:
            result = (
                pipeline
                | "Create" >> beam.Create([example])
                | "Process" >> beam.Map(metric_util.to_standard_metric_inputs)
                | "AddSlice" >> beam.Map(lambda x: ((), x))
                | "ComputeMeanMetric"
                >> beam.CombinePerKey(mean_metric_computation.combiner)
            )

            util.assert_that(
                result, CheckResultNan(key), label="result"
            )


class MeanEnd2EndTest(parameterized.TestCase):
    def testMeanEnd2End(self):
        extracts = [
            {
                "features": {
                    "example_weights": np.array([0.5]),
                    "age": np.array([30]),
                    "income": np.array([150000]),
                },
            },
            {
                "features": {
                    "example_weights": np.array([0.3]),
                    "age": np.array([40]),
                    "income": np.array([200000]),
                },
            },
        ]

        eval_config = text_format.Parse(
            """
        metrics_specs {
          metrics {
            class_name: "Mean"
            config: '"feature_key_path":["features", "age"], '
            '"example_weights_key_path":["features", "example_weights"]'
          },
          metrics {
            class_name: "Mean"
            config: '"feature_key_path":["features", "income"], '
            '"example_weights_key_path":["features", "example_weights"]'
          }   ,
        }
        """,
            config_pb2.EvalConfig(),
        )

        extractors = tfma.default_extractors(eval_config=eval_config)
        evaluators = tfma.default_evaluators(eval_config=eval_config)

        expected_key_age = metric_types.MetricKey(
            name="mean_features.age", example_weighted=True
        )
        expected_key_income = metric_types.MetricKey(
            name="mean_features.income", example_weighted=True
        )

        expected_result_age = 33.75  # (30 * 0.5 + 40 * 0.3) / (0.5 + 0.3) = 33.75
        # (150k * 0.5 + 200k * 0.3) / (0.5 + 0.3) = 168,750
        expected_result_income = 168750

        with beam.Pipeline(
            options=beam.options.pipeline_options.PipelineOptions(
                flags=["--no_save_main_session"]
            )
        ) as pipeline:
            result = (
                pipeline
                | "LoadData" >> beam.Create(extracts)
                | "ExtractEval"
                >> tfma.ExtractAndEvaluate(extractors=extractors, evaluators=evaluators)
            )

            self.assertIn("metrics", result)
            util.assert_that(
                result["metrics"],
                CheckResultMeanEnd2End(
                    expected_key_age,
                    expected_key_income,
                    expected_result_age,
                    expected_result_income,
                ),
                label="result",
            )

    def testMeanEnd2EndWithoutExampleWeights(self):
        extracts = [
            {
                "features": {
                    "age": np.array([30]),
                    "income": np.array([150000]),
                },
            },
            {
                "features": {
                    "age": np.array([40]),
                    "income": np.array([200000]),
                },
            },
        ]

        eval_config = text_format.Parse(
            """
        metrics_specs {
          metrics {
            class_name: "Mean"
            config: '{"feature_key_path":["features", "age"]}'
          },
          metrics {
            class_name: "Mean"
            config: '{"feature_key_path":["features", "income"]}'
          }   ,
        }
        """,
            config_pb2.EvalConfig(),
        )

        extractors = tfma.default_extractors(eval_config=eval_config)
        evaluators = tfma.default_evaluators(eval_config=eval_config)

        expected_key_age = metric_types.MetricKey(
            name="mean_features.age", example_weighted=False
        )
        expected_key_income = metric_types.MetricKey(
            name="mean_features.income", example_weighted=False
        )

        expected_result_age = 35  # (30 + 40) / (1 + 1) = 35
        # (150k + 200k) / (1 + 1) = 175000
        expected_result_income = 175000

        with beam.Pipeline(
            options=beam.options.pipeline_options.PipelineOptions(
                flags=["--no_save_main_session"]
            )
        ) as pipeline:
            result = (
                pipeline
                | "LoadData" >> beam.Create(extracts)
                | "ExtractEval"
                >> tfma.ExtractAndEvaluate(extractors=extractors, evaluators=evaluators)
            )

            self.assertIn("metrics", result)
            util.assert_that(
                result["metrics"],
                CheckResultMeanEnd2End(
                    expected_key_age,
                    expected_key_income,
                    expected_result_age,
                    expected_result_income,
                ),
                label="result",
            )


if __name__ == "__main__":
    tf.test.main()
