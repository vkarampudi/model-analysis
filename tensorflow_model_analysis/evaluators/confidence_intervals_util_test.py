# Copyright 2021 Google LLC
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
"""Tests for confidence_intervals_util."""

import apache_beam as beam
import numpy as np
from absl.testing import absltest, parameterized
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.testing import util
from numpy import testing

from tensorflow_model_analysis.evaluators import confidence_intervals_util
from tensorflow_model_analysis.metrics import binary_confusion_matrices, metric_types

_FULL_SAMPLE_ID = -1


def _check_sample_combine_fn_no_input(slice_key):
    def check_result(got_pcoll):
        if len(got_pcoll) != 1:
            raise util.BeamAssertException(f"Expected 1 result, got {len(got_pcoll)}")
        accumulators_by_slice = dict(got_pcoll)
        if slice_key not in accumulators_by_slice:
            raise util.BeamAssertException(f"Expected {slice_key} in results")
        accumulator = accumulators_by_slice[slice_key]
        if accumulator.num_samples != 2:
            raise util.BeamAssertException(
                f"Expected 2 samples, got {accumulator.num_samples}"
            )
        if not isinstance(accumulator.point_estimates, dict):
            raise util.BeamAssertException("point_estimates should be a dict")
        if not isinstance(accumulator.metric_samples, dict):
            raise util.BeamAssertException("metric_samples should be a dict")

    return check_result


def _check_sample_combine_fn(
    slice_key1,
    slice_key2,
    metric_key,
    array_metric_key,
    non_numeric_metric_key,
    non_numeric_array_metric_key,
    mixed_type_array_metric_key,
    skipped_metric_key,
):
    def check_result(got_pcoll):
        if len(got_pcoll) != 2:
            raise util.BeamAssertException(f"Expected 2 results, got {len(got_pcoll)}")
        accumulators_by_slice = dict(got_pcoll)

        if slice_key1 not in accumulators_by_slice:
            raise util.BeamAssertException(f"Expected {slice_key1} in results")
        slice1_accumulator = accumulators_by_slice[slice_key1]
        if metric_key not in slice1_accumulator.point_estimates:
            raise util.BeamAssertException("metric_key not in point_estimates")
        if slice1_accumulator.point_estimates[metric_key] != 2.1:
            raise util.BeamAssertException("Unexpected point estimate for metric_key")
        if metric_key not in slice1_accumulator.metric_samples:
            raise util.BeamAssertException("metric_key not in metric_samples")
        if slice1_accumulator.metric_samples[metric_key] != [1, 2]:
            raise util.BeamAssertException("Unexpected samples for metric_key")
        if array_metric_key not in slice1_accumulator.metric_samples:
            raise util.BeamAssertException("array_metric_key not in metric_samples")
        array_metric_samples = slice1_accumulator.metric_samples[array_metric_key]
        if len(array_metric_samples) != 2:
            raise util.BeamAssertException("Expected 2 array metric samples")
        testing.assert_array_equal(np.array([2, 3]), array_metric_samples[0])
        testing.assert_array_equal(np.array([0, 1]), array_metric_samples[1])

        if non_numeric_metric_key not in slice1_accumulator.point_estimates:
            raise util.BeamAssertException(
                "non_numeric_metric_key not in point_estimates"
            )
        if non_numeric_metric_key in slice1_accumulator.metric_samples:
            raise util.BeamAssertException(
                "non_numeric_metric_key should not have samples"
            )
        if non_numeric_array_metric_key not in slice1_accumulator.point_estimates:
            raise util.BeamAssertException(
                "non_numeric_array_metric_key not in point_estimates"
            )
        if non_numeric_array_metric_key in slice1_accumulator.metric_samples:
            raise util.BeamAssertException(
                "non_numeric_array_metric_key should not have samples"
            )
        if mixed_type_array_metric_key not in slice1_accumulator.point_estimates:
            raise util.BeamAssertException(
                "mixed_type_array_metric_key not in point_estimates"
            )
        if mixed_type_array_metric_key in slice1_accumulator.metric_samples:
            raise util.BeamAssertException(
                "mixed_type_array_metric_key should not have samples"
            )

        error_key = metric_types.MetricKey("__ERROR__")
        if error_key not in slice1_accumulator.point_estimates:
            raise util.BeamAssertException("error_key not in point_estimates")
        if "CI not computed for" not in slice1_accumulator.point_estimates[error_key]:
            raise util.BeamAssertException("Unexpected error message for CI failure")
        if skipped_metric_key in slice1_accumulator.metric_samples:
            raise util.BeamAssertException("skipped_metric_key should not have samples")

        if slice_key2 not in accumulators_by_slice:
            raise util.BeamAssertException(f"Expected {slice_key2} in results")
        slice2_accumulator = accumulators_by_slice[slice_key2]
        if metric_key not in slice2_accumulator.point_estimates:
            raise util.BeamAssertException("metric_key not in slice2 point_estimates")
        if slice2_accumulator.point_estimates[metric_key] != 6.3:
            raise util.BeamAssertException("Unexpected point estimate for slice2")
        if error_key not in slice2_accumulator.point_estimates:
            raise util.BeamAssertException("error_key not in slice2 point_estimates")

    return check_result


class _ValidateSampleCombineFn(confidence_intervals_util.SampleCombineFn):
    def extract_output(
        self,
        accumulator: confidence_intervals_util.SampleCombineFn.SampleAccumulator,
    ) -> confidence_intervals_util.SampleCombineFn.SampleAccumulator:
        return self._validate_accumulator(accumulator)


class ConfidenceIntervalsUtilTest(parameterized.TestCase):
    @parameterized.named_parameters(
        {
            "testcase_name": "_ints",
            "values": [0, 1, 2],
            "ddof": 1,
            "expected_mean": 1,
            "expected_std": np.std([0, 1, 2], ddof=1),
        },
        {
            "testcase_name": "_ndarrays",
            "values": [np.array([0]), np.array([1]), np.array([2])],
            "ddof": 1,
            "expected_mean": np.array([1]),
            "expected_std": np.array([np.std([0, 1, 2], ddof=1)]),
        },
        {
            "testcase_name": "_confusion_matrices",
            "values": [
                binary_confusion_matrices.Matrices(
                    thresholds=[0.5], tp=[0], fp=[1], tn=[2], fn=[3]
                ),
                binary_confusion_matrices.Matrices(
                    thresholds=[0.5], tp=[4], fp=[5], tn=[6], fn=[7]
                ),
                binary_confusion_matrices.Matrices(
                    thresholds=[0.5], tp=[8], fp=[9], tn=[10], fn=[11]
                ),
            ],
            "ddof": 1,
            "expected_mean": binary_confusion_matrices.Matrices(
                thresholds=[0.5],
                tp=np.mean([0, 4, 8]),
                fp=np.mean([1, 5, 9]),
                tn=np.mean([2, 6, 10]),
                fn=np.mean([3, 7, 11]),
            ),
            "expected_std": binary_confusion_matrices.Matrices(
                thresholds=[0.5],
                tp=np.std([0, 4, 8], ddof=1),
                fp=np.std([1, 5, 9], ddof=1),
                tn=np.std([2, 6, 10], ddof=1),
                fn=np.std([3, 7, 11], ddof=1),
            ),
        },
    )
    def test_mean_and_std(self, values, ddof, expected_mean, expected_std):
        actual_mean, actual_std = confidence_intervals_util.mean_and_std(values, ddof)
        self.assertEqual(expected_mean, actual_mean)
        self.assertEqual(expected_std, actual_std)

    def test_sample_combine_fn(self):
        metric_key = metric_types.MetricKey("metric")
        array_metric_key = metric_types.MetricKey("array_metric")
        missing_sample_metric_key = metric_types.MetricKey("missing_metric")
        non_numeric_metric_key = metric_types.MetricKey("non_numeric_metric")
        non_numeric_array_metric_key = metric_types.MetricKey("non_numeric_array")
        mixed_type_array_metric_key = metric_types.MetricKey("mixed_type_array")
        skipped_metric_key = metric_types.MetricKey("skipped_metric")
        slice_key1 = (("slice_feature", 1),)
        slice_key2 = (("slice_feature", 2),)
        # the sample value is irrelevant for this test as we only verify counters.
        samples = [
            # unsampled value for slice 1
            (
                slice_key1,
                confidence_intervals_util.SampleMetrics(
                    sample_id=_FULL_SAMPLE_ID,
                    metrics={
                        metric_key: 2.1,
                        array_metric_key: np.array([1, 2]),
                        missing_sample_metric_key: 3,
                        non_numeric_metric_key: "a",
                        non_numeric_array_metric_key: np.array(["a", "aaa"]),
                        mixed_type_array_metric_key: np.array(["a"]),
                        skipped_metric_key: 16,
                    },
                ),
            ),
            # sample values for slice 1
            (
                slice_key1,
                confidence_intervals_util.SampleMetrics(
                    sample_id=0,
                    metrics={
                        metric_key: 1,
                        array_metric_key: np.array([2, 3]),
                        missing_sample_metric_key: 2,
                        non_numeric_metric_key: "b",
                        non_numeric_array_metric_key: np.array(["a", "aaa"]),
                        # one sample is an empty float array
                        mixed_type_array_metric_key: np.array([], dtype=float),
                        skipped_metric_key: 7,
                    },
                ),
            ),
            # sample values for slice 1 missing missing_sample_metric_key
            (
                slice_key1,
                confidence_intervals_util.SampleMetrics(
                    sample_id=1,
                    metrics={
                        metric_key: 2,
                        array_metric_key: np.array([0, 1]),
                        non_numeric_metric_key: "c",
                        non_numeric_array_metric_key: np.array(["a", "aaa"]),
                        # one sample is a unicode array
                        mixed_type_array_metric_key: np.array(["a"]),
                        skipped_metric_key: 8,
                    },
                ),
            ),
            # unsampled value for slice 2
            (
                slice_key2,
                confidence_intervals_util.SampleMetrics(
                    sample_id=_FULL_SAMPLE_ID,
                    metrics={
                        metric_key: 6.3,
                        array_metric_key: np.array([10, 20]),
                        missing_sample_metric_key: 6,
                        non_numeric_metric_key: "d",
                        non_numeric_array_metric_key: np.array(["a", "aaa"]),
                        mixed_type_array_metric_key: np.array(["a"]),
                        skipped_metric_key: 10000,
                    },
                ),
            ),
            # Only 1 sample value (missing sample ID 1) for slice 2
            (
                slice_key2,
                confidence_intervals_util.SampleMetrics(
                    sample_id=0,
                    metrics={
                        metric_key: 3,
                        array_metric_key: np.array([20, 30]),
                        missing_sample_metric_key: 12,
                        non_numeric_metric_key: "d",
                        non_numeric_array_metric_key: np.array(["a", "aaa"]),
                        mixed_type_array_metric_key: np.array(["a"]),
                        skipped_metric_key: 5000,
                    },
                ),
            ),
        ]

        options = PipelineOptions(flags=["--no_save_main_session"])
        with beam.Pipeline(options=options) as pipeline:
            result = (
                pipeline
                | "Create" >> beam.Create(samples, reshuffle=False)
                | "CombineSamplesPerKey"
                >> beam.CombinePerKey(
                    _ValidateSampleCombineFn(
                        num_samples=2,
                        full_sample_id=_FULL_SAMPLE_ID,
                        skip_ci_metric_keys=[skipped_metric_key],
                    )
                )
            )

            util.assert_that(
                result,
                _check_sample_combine_fn(
                    slice_key1,
                    slice_key2,
                    metric_key,
                    array_metric_key,
                    non_numeric_metric_key,
                    non_numeric_array_metric_key,
                    mixed_type_array_metric_key,
                    skipped_metric_key,
                ),
            )

            runner_result = pipeline.run()
            runner_result.wait_until_finish()
            # we expect one missing samples counter increment for slice2, since we
            # expected 2 samples, but only saw 1.
            metric_filter = beam.metrics.metric.MetricsFilter().with_name(
                "num_slices_missing_samples"
            )
            counters = runner_result.metrics().query(filter=metric_filter)["counters"]
            self.assertLen(counters, 1)
            self.assertEqual(1, counters[0].committed)

            # verify total slice counter
            metric_filter = beam.metrics.metric.MetricsFilter().with_name("num_slices")
            counters = runner_result.metrics().query(filter=metric_filter)["counters"]
            self.assertLen(counters, 1)
            self.assertEqual(2, counters[0].committed)

    def test_sample_combine_fn_no_input(self):
        slice_key = (("slice_feature", 1),)
        samples = [
            (
                slice_key,
                confidence_intervals_util.SampleMetrics(
                    sample_id=_FULL_SAMPLE_ID, metrics={}
                ),
            ),
            (
                slice_key,
                confidence_intervals_util.SampleMetrics(sample_id=0, metrics={}),
            ),
            (
                slice_key,
                confidence_intervals_util.SampleMetrics(sample_id=1, metrics={}),
            ),
        ]

        options = PipelineOptions(flags=["--no_save_main_session"])
        with beam.Pipeline(options=options) as pipeline:
            result = (
                pipeline
                | "Create" >> beam.Create(samples)
                | "CombineSamplesPerKey"
                >> beam.CombinePerKey(
                    _ValidateSampleCombineFn(
                        num_samples=2, full_sample_id=_FULL_SAMPLE_ID
                    )
                )
            )

            util.assert_that(result, _check_sample_combine_fn_no_input(slice_key))
            pipeline.run().wait_until_finish()


if __name__ == "__main__":
    absltest.main()
