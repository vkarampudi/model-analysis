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
"""Tests for ROUGE metrics."""

import statistics as stats

import apache_beam as beam
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from apache_beam.testing import util
from google.protobuf import text_format
from rouge_score import tokenizers

from tensorflow_model_analysis import constants
from tensorflow_model_analysis.evaluators import metrics_plots_and_validations_evaluator
from tensorflow_model_analysis.metrics import metric_types, metric_util, rouge
from tensorflow_model_analysis.proto import config_pb2
from tensorflow_model_analysis.slicer import slicer_lib as slicer
from tensorflow_model_analysis.utils import test_util


def _get_result(pipeline, examples, combiner):
    return (
        pipeline
        | "Create" >> beam.Create(examples)
        | "Process" >> beam.Map(metric_util.to_standard_metric_inputs)
        | "AddSlice" >> beam.Map(lambda x: ((), x))
        | "ComputeRouge" >> beam.CombinePerKey(combiner)
    )


class CheckResult:
    def __init__(self, expected_name, rouge_computation):
        self.expected_name = expected_name
        self.rouge_computation = rouge_computation

    def __call__(self, got):
        if len(got) != 1:
            raise ValueError(f"Expected 1 result, got {len(got)}")
        got_slice_key, got_metrics = got[0]
        if got_slice_key != ():
            raise ValueError(f"Expected slice_key to be (), got {got_slice_key}")
        if self.rouge_computation.keys[0] not in got_metrics:
            raise ValueError(
                f"Expected {self.rouge_computation.keys[0]} in got_metrics"
            )

        got_name = next(iter(got_metrics.keys())).name
        if got_name != self.expected_name:
            raise ValueError(f"Expected name {self.expected_name}, got {got_name}")

class CheckResultScores:
    def __init__(self, rouge_key, rouge_computation, expected_precision, expected_recall, expected_fmeasure, places=None):
        self.rouge_key = rouge_key
        self.rouge_computation = rouge_computation
        self.expected_precision = expected_precision
        self.expected_recall = expected_recall
        self.expected_fmeasure = expected_fmeasure
        self.places = places

    def __call__(self, got):
        if len(got) != 1:
            raise ValueError(f"Expected 1 result, got {len(got)}")
        got_slice_key, got_metrics = got[0]
        if got_slice_key != ():
            raise ValueError(f"Expected slice_key to be (), got {got_slice_key}")

        got_precision = got_metrics[self.rouge_key].precision
        got_recall = got_metrics[self.rouge_key].recall
        got_fmeasure = got_metrics[self.rouge_key].fmeasure

        delta = 10**-self.places if self.places else 1e-7
        if abs(got_precision - self.expected_precision) > delta:
            raise ValueError(f"Precision mismatch: expected {self.expected_precision}, got {got_precision}")
        if abs(got_recall - self.expected_recall) > delta:
            raise ValueError(f"Recall mismatch: expected {self.expected_recall}, got {got_recall}")
        if abs(got_fmeasure - self.expected_fmeasure) > delta:
            raise ValueError(f"F-measure mismatch: expected {self.expected_fmeasure}, got {got_fmeasure}")

class CheckResultNan:
    def __init__(self, rouge_key, rouge_computation):
        self.rouge_key = rouge_key
        self.rouge_computation = rouge_computation

    def __call__(self, got):
        if len(got) != 1:
            raise ValueError(f"Expected 1 result, got {len(got)}")
        got_slice_key, got_metrics = got[0]
        if got_slice_key != ():
            raise ValueError(f"Expected slice_key to be (), got {got_slice_key}")

        if not np.isnan(got_metrics[self.rouge_key].precision):
            raise ValueError("Expected NaN precision")
        if not np.isnan(got_metrics[self.rouge_key].recall):
            raise ValueError("Expected NaN recall")
        if not np.isnan(got_metrics[self.rouge_key].fmeasure):
            raise ValueError("Expected NaN fmeasure")

class CheckResultScoresE2E:
    def __init__(self, rouge_key, rouge_type, expected_unweighted_scores, example_weights):
        self.rouge_key = rouge_key
        self.rouge_type = rouge_type
        self.expected_unweighted_scores = expected_unweighted_scores
        self.example_weights = example_weights

    def __call__(self, got):
        if len(got) != 1:
            raise ValueError(f"Expected 1 result, got {len(got)}")
        got_slice_key, got_metrics = got[0]
        if got_slice_key != ():
            raise ValueError(f"Expected (), got {got_slice_key}")
        if self.rouge_key not in got_metrics:
            raise ValueError(f"Expected {self.rouge_key} in got_metrics")

        expected_precision = np.average(
            self.expected_unweighted_scores[self.rouge_type][0],
            weights=self.example_weights,
        )
        expected_recall = np.average(
            self.expected_unweighted_scores[self.rouge_type][1],
            weights=self.example_weights,
        )
        expected_fmeasure = np.average(
            self.expected_unweighted_scores[self.rouge_type][2],
            weights=self.example_weights,
        )

        if abs(got_metrics[self.rouge_key].precision - expected_precision) > 1e-7:
            raise ValueError("Precision mismatch")
        if abs(got_metrics[self.rouge_key].recall - expected_recall) > 1e-7:
            raise ValueError("Recall mismatch")
        if abs(got_metrics[self.rouge_key].fmeasure - expected_fmeasure) > 1e-7:
            raise ValueError("F-measure mismatch")

class RougeTest(test_util.TensorflowModelAnalysisTest, parameterized.TestCase):

    @parameterized.parameters(["rougen", "rouge0", "rouge10"])
    def testInvalidRougeTypes(self, rouge_type):
        target_text = "testing one two"
        prediction_text = "testing"
        example = {
            constants.LABELS_KEY: target_text,
            constants.PREDICTIONS_KEY: prediction_text,
        }
        rouge_computation = rouge.Rouge(rouge_type).computations()[0]
        with self.assertRaisesRegex(
            (ValueError, RuntimeError),
            "(Invalid rouge type|rougen requires positive n)",
        ):
            with beam.Pipeline() as pipeline:
                _get_result(
                    pipeline=pipeline,
                    examples=[example],
                    combiner=rouge_computation.combiner,
                )

    @parameterized.parameters(
        [
            "rouge1",
            "rouge2",
            "rouge3",
            "rouge4",
            "rouge5",
            "rouge6",
            "rouge7",
            "rouge8",
            "rouge9",
            "rougeL",
            "rougeLsum",
        ]
    )
    def testValidRogueTypes(self, rouge_type):
        target_text = "testing one two"
        prediction_text = "testing"
        example = {
            constants.LABELS_KEY: target_text,
            constants.PREDICTIONS_KEY: prediction_text,
        }
        rouge_computation = rouge.Rouge(rouge_type).computations()[0]
        with beam.Pipeline(
            options=beam.options.pipeline_options.PipelineOptions(
                flags=["--no_save_main_session"]
            )
        ) as pipeline:
            result = _get_result(
                pipeline=pipeline,
                examples=[example],
                combiner=rouge_computation.combiner,
            )

            util.assert_that(
                result, CheckResult(rouge_type, rouge_computation), label="result"
            )

    @parameterized.parameters(["rouge1", "rouge2", "rougeL", "rougeLsum"])
    def testNameOverride(self, rouge_type):
        target_text = "testing one two"
        prediction_text = "testing"
        expected_name = "override_default_name_with_this"
        example = {
            constants.LABELS_KEY: target_text,
            constants.PREDICTIONS_KEY: prediction_text,
        }
        rouge_computation = rouge.Rouge(rouge_type, name=expected_name).computations()[
            0
        ]
        with beam.Pipeline(
            options=beam.options.pipeline_options.PipelineOptions(
                flags=["--no_save_main_session"]
            )
        ) as pipeline:
            result = _get_result(
                pipeline=pipeline,
                examples=[example],
                combiner=rouge_computation.combiner,
            )

            util.assert_that(
                result,
                CheckResult(expected_name, rouge_computation),
                label="result",
            )

    @parameterized.named_parameters(
        (
            "rouge1",
            "rouge1",
            ["testing one two", "testing"],
            1,
            1 / 3,
            1 / 2,
        ),
        (
            "rouge2",
            "rouge2",
            ["testing one two", "testing one"],
            1,
            1 / 2,
            2 / 3,
        ),
        (
            "rougeL_consecutive",
            "rougeL",
            ["testing one two", "testing one"],
            1,
            2 / 3,
            4 / 5,
        ),
        (
            "rougeL_nonconsecutive",
            "rougeL",
            ["testing one two", "testing two"],
            1,
            2 / 3,
            4 / 5,
        ),
        (
            "rougeLsum",
            "rougeLsum",
            ["w1 w2 w3 w4 w5", "w1 w2 w6 w7 w8\nw1 w3 w8 w9 w5"],
            2 / 5,
            4 / 5,
            8 / 15,
        ),
    )
    def testRougeSingleExample(
        self,
        rouge_type,
        example_texts,
        expected_precision,
        expected_recall,
        expected_fmeasure,
    ):
        example = {
            constants.LABELS_KEY: example_texts[0],
            constants.PREDICTIONS_KEY: example_texts[1],
        }
        rouge_key = metric_types.MetricKey(name=rouge_type)
        rouge_computation = rouge.Rouge(rouge_type).computations()[0]
        with beam.Pipeline(
            options=beam.options.pipeline_options.PipelineOptions(
                flags=["--no_save_main_session"]
            )
        ) as pipeline:
            result = _get_result(
                pipeline=pipeline,
                examples=[example],
                combiner=rouge_computation.combiner,
            )

            util.assert_that(
                result,
                CheckResultScores(
                    rouge_key,
                    rouge_computation,
                    expected_precision,
                    expected_recall,
                    expected_fmeasure,
                ),
                label="result",
            )

    @parameterized.parameters("rouge1", "rouge2", "rougeL", "rougeLsum")
    def testRougeMultipleExampleWeights(self, rouge_type):
        example = {
            constants.LABELS_KEY: "testing one two",
            constants.PREDICTIONS_KEY: "testing",
            constants.EXAMPLE_WEIGHTS_KEY: [0.4, 0.6],
        }
        rouge_computation = rouge.Rouge(rouge_type).computations()[0]
        with self.assertRaisesRegex(
            (ValueError, RuntimeError),
            "if example_weight size > 0, the values must all be the same",
        ):
            with beam.Pipeline() as pipeline:
                _get_result(
                    pipeline=pipeline,
                    examples=[example],
                    combiner=rouge_computation.combiner,
                )

    @parameterized.named_parameters(
        [
            (
                "rouge1",
                "rouge1",
                ["testing one two", "This is a test"],
                "This is not a test",
                4 / 5,
                1,
                8 / 9,
            ),
            (
                "rouge2",
                "rouge2",
                ["testing one two", "This is a test"],
                "This is not a test",
                1 / 2,
                2 / 3,
                4 / 7,
            ),
            (
                "rougeL",
                "rougeL",
                ["testing one two", "This is a test"],
                "This is not a test",
                4 / 5,
                1,
                8 / 9,
            ),
            (
                "rougeLsum",
                "rougeLsum",
                ["testing one two", "This is a test"],
                "This is not a test",
                # ROUGE-L == ROUGE-L-Sum for these examples
                # because there is no sentence splitting
                4 / 5,
                1,
                8 / 9,
            ),
        ]
    )
    def testRougeMultipleTargetTexts(
        self,
        rouge_type,
        targets,
        prediction,
        expected_precision,
        expected_recall,
        expected_fmeasure,
    ):
        example = {
            constants.LABELS_KEY: targets,
            constants.PREDICTIONS_KEY: prediction,
        }
        rouge_key = metric_types.MetricKey(name=rouge_type)
        rouge_computation = rouge.Rouge(rouge_type).computations()[0]
        with beam.Pipeline() as pipeline:
            result = _get_result(
                pipeline=pipeline,
                examples=[example],
                combiner=rouge_computation.combiner,
            )

            util.assert_that(
                result,
                CheckResultScores(
                    rouge_key,
                    rouge_computation,
                    expected_precision,
                    expected_recall,
                    expected_fmeasure,
                ),
                label="result",
            )

    @parameterized.named_parameters(
        [
            (
                "rouge1",
                "rouge1",
                ["testing one two", "testing"],
                ["This is a test", "This is not a test"],
                stats.mean([1, 4 / 5]),
                stats.mean([1 / 3, 1]),
                stats.mean([1 / 2, 8 / 9]),
            ),
            (
                "rouge2",
                "rouge2",
                ["testing one two", "testing one"],
                ["This is a test", "This is not a test"],
                stats.mean([1, 1 / 2]),
                stats.mean([1 / 2, 2 / 3]),
                stats.mean([2 / 3, 4 / 7]),
            ),
            (
                "rougeL",
                "rougeL",
                ["testing one two", "testing one"],
                ["This is a test", "This is not a test"],
                stats.mean([1, 4 / 5]),
                stats.mean([2 / 3, 1]),
                stats.mean([4 / 5, 8 / 9]),
            ),
            (
                "rougeLsum",
                "rougeLsum",
                ["testing one two", "testing one"],
                ["This is a test", "This is not a test"],
                # ROUGE-L == ROUGE-L-Sum for these examples
                # because there is no sentence splitting
                stats.mean([1, 4 / 5]),
                stats.mean([2 / 3, 1]),
                stats.mean([4 / 5, 8 / 9]),
            ),
        ]
    )
    def testRougeMultipleExamplesUnweighted(
        self,
        rouge_type,
        example_1_texts,
        example_2_texts,
        expected_precision,
        expected_recall,
        expected_fmeasure,
    ):
        example1 = {
            constants.LABELS_KEY: example_1_texts[0],
            constants.PREDICTIONS_KEY: example_1_texts[1],
        }
        example2 = {
            constants.LABELS_KEY: example_2_texts[0],
            constants.PREDICTIONS_KEY: example_2_texts[1],
        }
        rouge_key = metric_types.MetricKey(name=rouge_type)
        rouge_computation = rouge.Rouge(rouge_type).computations()[0]
        with beam.Pipeline(
            options=beam.options.pipeline_options.PipelineOptions(
                flags=["--no_save_main_session"]
            )
        ) as pipeline:
            result = _get_result(
                pipeline=pipeline,
                examples=[example1, example2],
                combiner=rouge_computation.combiner,
            )

            util.assert_that(
                result,
                CheckResultScores(
                    rouge_key,
                    rouge_computation,
                    expected_precision,
                    expected_recall,
                    expected_fmeasure,
                    places=6,
                ),
                label="result",
            )

    example_weights = [0.5, 0.7]

    @parameterized.named_parameters(
        [
            (
                "rouge1",
                "rouge1",
                ["testing one two", "testing"],
                ["This is a test", "This is not a test"],
                np.average([1, 4 / 5], weights=example_weights),
                np.average([1 / 3, 1], weights=example_weights),
                np.average([1 / 2, 8 / 9], weights=example_weights),
            ),
            (
                "rouge2",
                "rouge2",
                ["testing one two", "testing one"],
                ["This is a test", "This is not a test"],
                np.average([1, 1 / 2], weights=example_weights),
                np.average([1 / 2, 2 / 3], weights=example_weights),
                np.average([2 / 3, 4 / 7], weights=example_weights),
            ),
            (
                "rougeL",
                "rougeL",
                ["testing one two", "testing one"],
                ["This is a test", "This is not a test"],
                np.average([1, 4 / 5], weights=example_weights),
                np.average([2 / 3, 1], weights=example_weights),
                np.average([4 / 5, 8 / 9], weights=example_weights),
            ),
            (
                "rougeLsum",
                "rougeLsum",
                ["testing one two", "testing one"],
                ["This is a test", "This is not a test"],
                # ROUGE-L == ROUGE-L-Sum for these examples
                # because there is no sentence splitting
                np.average([1, 4 / 5], weights=example_weights),
                np.average([2 / 3, 1], weights=example_weights),
                np.average([4 / 5, 8 / 9], weights=example_weights),
            ),
        ]
    )
    def testRougeMultipleExamplesWeighted(
        self,
        rouge_type,
        example_1_texts,
        example_2_texts,
        expected_precision,
        expected_recall,
        expected_fmeasure,
    ):
        example1 = {
            constants.LABELS_KEY: example_1_texts[0],
            constants.PREDICTIONS_KEY: example_1_texts[1],
            constants.EXAMPLE_WEIGHTS_KEY: self.example_weights[0],
        }
        example2 = {
            constants.LABELS_KEY: example_2_texts[0],
            constants.PREDICTIONS_KEY: example_2_texts[1],
            constants.EXAMPLE_WEIGHTS_KEY: self.example_weights[1],
        }
        rouge_key = metric_types.MetricKey(name=rouge_type)
        rouge_computation = rouge.Rouge(rouge_type).computations()[0]
        with beam.Pipeline(
            options=beam.options.pipeline_options.PipelineOptions(
                flags=["--no_save_main_session"]
            )
        ) as pipeline:
            result = _get_result(
                pipeline=pipeline,
                examples=[example1, example2],
                combiner=rouge_computation.combiner,
            )

            util.assert_that(
                result,
                CheckResultScores(
                    rouge_key,
                    rouge_computation,
                    expected_precision,
                    expected_recall,
                    expected_fmeasure,
                ),
                label="result",
            )

    @parameterized.parameters("rouge1", "rouge2", "rougeL", "rougeLsum")
    def testRougeWeightedCountIsZero(self, rouge_type):
        example = {
            constants.LABELS_KEY: "testing one two",
            constants.PREDICTIONS_KEY: "testing",
            constants.EXAMPLE_WEIGHTS_KEY: [0],
        }
        rouge_key = metric_types.MetricKey(name=rouge_type)
        rouge_computation = rouge.Rouge(rouge_type).computations()[0]
        with beam.Pipeline(
            options=beam.options.pipeline_options.PipelineOptions(
                flags=["--no_save_main_session"]
            )
        ) as pipeline:
            result = _get_result(
                pipeline=pipeline,
                examples=[example],
                combiner=rouge_computation.combiner,
            )

            util.assert_that(
                result,
                CheckResultNan(rouge_key, rouge_computation),
                label="result",
            )

    def testRougeLSumSentenceSplitting(self):
        rouge_type = "rougeLsum"
        rouge_key = metric_types.MetricKey(name=rouge_type)
        tokenizer_preparer_logging_message = (
            "INFO:absl:" + rouge._LOGGING_MESSAGE_TOKENIZER_PREPARER
        )
        rouge_computation = rouge.Rouge(rouge_type, use_stemmer=True).computations()[0]
        target_text = "First sentence.\nSecond Sentence."
        prediction_text = "Second sentence.\nFirst Sentence."
        example = {
            constants.LABELS_KEY: target_text,
            constants.PREDICTIONS_KEY: prediction_text,
        }
        with self.assertLogs(level="INFO") as cm:
            with beam.Pipeline(
                options=beam.options.pipeline_options.PipelineOptions(
                    flags=["--no_save_main_session"]
                )
            ) as pipeline:
                result = _get_result(
                    pipeline=pipeline,
                    examples=[example],
                    combiner=rouge_computation.combiner,
                )

                util.assert_that(
                    result,
                    CheckResultScores(
                        rouge_key,
                        rouge_computation,
                        1,
                        1,
                        1,
                    ),
                    label="result",
                )
        self.assertNotIn(tokenizer_preparer_logging_message, cm.output)

        # Without newlines, summaries are treated as single sentences.
        target_text = target_text.replace("\n", " ")
        prediction_text = prediction_text.replace("\n", " ")
        example = {
            constants.LABELS_KEY: target_text,
            constants.PREDICTIONS_KEY: prediction_text,
        }
        with self.assertLogs(level="INFO") as cm:
            with beam.Pipeline(
                options=beam.options.pipeline_options.PipelineOptions(
                    flags=["--no_save_main_session"]
                )
            ) as pipeline:
                result = _get_result(
                    pipeline=pipeline,
                    examples=[example],
                    combiner=rouge_computation.combiner,
                )

                util.assert_that(
                    result,
                    CheckResultScores(
                        rouge_key,
                        rouge_computation,
                        1 / 2,
                        1 / 2,
                        1 / 2,
                    ),
                    label="result",
                )
        self.assertNotIn(tokenizer_preparer_logging_message, cm.output)

        def check_split_summaries_result():
            with beam.Pipeline(
                options=beam.options.pipeline_options.PipelineOptions(
                    flags=["--no_save_main_session"]
                )
            ) as pipeline:
                result = _get_result(
                    pipeline=pipeline,
                    examples=[example],
                    combiner=rouge_computation.combiner,
                )

                util.assert_that(
                    result,
                    CheckResultScores(
                        rouge_key,
                        rouge_computation,
                        1,
                        1,
                        1,
                    ),
                    label="result",
                )

        # Split summaries into sentences using nltk
        rouge_computation = rouge.Rouge(
            rouge_type, use_stemmer=True, split_summaries=True
        ).computations()[0]
        check_split_summaries_result()

    def testRougeTokenizer(self):
        rouge_type = "rouge1"
        target_text = "testing one two"
        prediction_text = "testing"
        example = {
            constants.LABELS_KEY: target_text,
            constants.PREDICTIONS_KEY: prediction_text,
        }
        rouge_key = metric_types.MetricKey(name=rouge_type)
        rouge_computation = rouge.Rouge(
            rouge_type, tokenizer=tokenizers.DefaultTokenizer()
        ).computations()[0]
        with beam.Pipeline(
            options=beam.options.pipeline_options.PipelineOptions(
                flags=["--no_save_main_session"]
            )
        ) as pipeline:
            result = _get_result(
                pipeline=pipeline,
                examples=[example],
                combiner=rouge_computation.combiner,
            )

            util.assert_that(
                result,
                CheckResultScores(
                    rouge_key,
                    rouge_computation,
                    1,
                    1 / 3,
                    1 / 2,
                ),
                label="result",
            )


class RougeEnd2EndTest(parameterized.TestCase):
    def testRougeEnd2End(self):
        # Same tests as RougeTest.testRougeMultipleExamplesWeighted
        eval_config = text_format.Parse(
            """
        model_specs {
          label_key: "labels"
          prediction_key: "predictions"
        }
        metrics_specs {
          metrics {
            class_name: "Rouge"
            config: '"rouge_type":"rouge1"'
          }
          metrics {
            class_name: "Rouge"
            config: '"rouge_type":"rouge2"'
          }
          metrics {
            class_name: "Rouge"
            config: '"rouge_type":"rougeL"'
          }
          metrics {
            class_name: "Rouge"
            config: '"rouge_type":"rougeLsum"'
          }
        }
        """,
            config_pb2.EvalConfig(),
        )
        rouge_types = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        example_weights = [0.5, 0.7]
        extracts = [
            {
                constants.LABELS_KEY: np.array(["testing one two"]),
                constants.PREDICTIONS_KEY: np.array(["testing"]),
                constants.EXAMPLE_WEIGHTS_KEY: np.array([example_weights[0]]),
                constants.FEATURES_KEY: None,
                constants.SLICE_KEY_TYPES_KEY: slicer.slice_keys_to_numpy_array([()]),
            },
            {
                constants.LABELS_KEY: np.array(["This is a test"]),
                constants.PREDICTIONS_KEY: np.array(["This is not a test"]),
                constants.EXAMPLE_WEIGHTS_KEY: np.array([example_weights[1]]),
                constants.FEATURES_KEY: None,
                constants.SLICE_KEY_TYPES_KEY: slicer.slice_keys_to_numpy_array([()]),
            },
        ]

        # Values are [unweighed_score_for_example_1, unweighted_score_for_example_2]
        # where the scores are precision, recall, and fmeasure.
        expected_unweighted_scores = {
            "rouge1": ([1, 4 / 5], [1 / 3, 1], [1 / 2, 8 / 9]),
            "rouge2": ([0, 1 / 2], [0, 2 / 3], [0, 4 / 7]),
            "rougeL": ([1, 4 / 5], [1 / 3, 1], [1 / 2, 8 / 9]),
            "rougeLsum": ([1, 4 / 5], [1 / 3, 1], [1 / 2, 8 / 9]),
        }
        for rouge_type in rouge_types:
            rouge_key = metric_types.MetricKey(name=rouge_type)

            with beam.Pipeline(
                options=beam.options.pipeline_options.PipelineOptions(
                    flags=["--no_save_main_session"]
                )
            ) as pipeline:
                result = (
                    pipeline
                    | "LoadData" >> beam.Create(extracts)
                    | "ExtractEval"
                    >> metrics_plots_and_validations_evaluator.MetricsPlotsAndValidationsEvaluator(
                        eval_config=eval_config
                    ).ptransform
                )

                self.assertIn("metrics", result)
                util.assert_that(
                    result["metrics"],
                    CheckResultScoresE2E(
                        rouge_key,
                        rouge_type,
                        expected_unweighted_scores,
                        example_weights,
                    ),
                    label="result",
                )


if __name__ == "__main__":
    tf.test.main()
