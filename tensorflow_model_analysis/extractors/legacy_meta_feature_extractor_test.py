# Copyright 2018 Google LLC
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
"""Test for using the MetaFeatureExtractor as part of TFMA."""

import apache_beam as beam
import numpy as np
import tensorflow as tf
from apache_beam.testing import util

from tensorflow_model_analysis import constants
from tensorflow_model_analysis.api import types
from tensorflow_model_analysis.extractors import (
    legacy_meta_feature_extractor as meta_feature_extractor,
)
from tensorflow_model_analysis.extractors import slice_key_extractor
from tensorflow_model_analysis.slicer import slicer_lib as slicer
from tensorflow_model_analysis.utils import test_util


class CheckMetaFeaturesResult:
    def __call__(self, got):
        if len(got) != 2:
            raise ValueError("Expected 2 results, got %s" % got)
        for res in got:
            if (
                "num_interests"
                not in res[constants.FEATURES_PREDICTIONS_LABELS_KEY].features
            ):
                raise ValueError("Expected num_interests in features")
            expected = len(
                meta_feature_extractor.get_feature_value(
                    res[constants.FEATURES_PREDICTIONS_LABELS_KEY], "interest"
                )
            )
            actual = meta_feature_extractor.get_feature_value(
                res[constants.FEATURES_PREDICTIONS_LABELS_KEY], "num_interests"
            )
            if expected != actual:
                raise ValueError("Expected %s, got %s" % (expected, actual))


class CheckSliceOnMetaFeatureResult:
    def __call__(self, got):
        if len(got) != 4:
            raise ValueError("Expected 4 results, got %s" % got)
        expected_slice_keys = [
            (),
            (),
            (("num_interests", 1),),
            (("num_interests", 2),),
        ]
        actual_slice_keys = sorted(slice_key for slice_key, _ in got)
        if actual_slice_keys != sorted(expected_slice_keys):
            raise ValueError(
                "Expected slice keys %s, got %s"
                % (expected_slice_keys, actual_slice_keys)
            )


def make_features_dict(features_dict):
    result = {}
    for key, value in features_dict.items():
        result[key] = {"node": np.array(value)}
    return result


def create_fpls():
    """Create test FPL dicts that can be used for verification."""
    fpl1 = types.FeaturesPredictionsLabels(
        input_ref=0,
        features=make_features_dict(
            {"gender": ["f"], "age": [13], "interest": ["cars"]}
        ),
        predictions=make_features_dict(
            {
                "kb": [1],
            }
        ),
        labels=make_features_dict({"ad_risk_score": [0]}),
    )
    fpl2 = types.FeaturesPredictionsLabels(
        input_ref=1,
        features=make_features_dict(
            {"gender": ["m"], "age": [10], "interest": ["cars", "movies"]}
        ),
        predictions=make_features_dict(
            {
                "kb": [1],
            }
        ),
        labels=make_features_dict({"ad_risk_score": [0]}),
    )
    return [fpl1, fpl2]


def wrap_fpl(fpl):
    return {
        constants.INPUT_KEY: "xyz",
        constants.FEATURES_PREDICTIONS_LABELS_KEY: fpl,
    }


def get_num_interests(fpl):
    interests = meta_feature_extractor.get_feature_value(fpl, "interest")
    new_features = {"num_interests": len(interests)}
    return new_features


class MetaFeatureExtractorTest(test_util.TensorflowModelAnalysisTest):
    def testMetaFeatures(self):
        with beam.Pipeline() as pipeline:
            fpls = create_fpls()

            metrics = (
                pipeline
                | "CreateTestInput" >> beam.Create(fpls)
                | "WrapFpls" >> beam.Map(wrap_fpl)
                | "ExtractInterestsNum"
                >> meta_feature_extractor.ExtractMetaFeature(get_num_interests)
            )

            util.assert_that(metrics, CheckMetaFeaturesResult())

    def testNoModificationOfExistingKeys(self):
        def bad_meta_feature_fn(_):
            return {"interest": ["bad", "key"]}

        with self.assertRaisesRegex(
            Exception, "Modification of existing keys is not allowed"
        ):
            with beam.Pipeline() as pipeline:
                fpls = create_fpls()

                _ = (
                    pipeline
                    | "CreateTestInput" >> beam.Create(fpls)
                    | "WrapFpls" >> beam.Map(wrap_fpl)
                    | "ExtractInterestsNum"
                    >> meta_feature_extractor.ExtractMetaFeature(bad_meta_feature_fn)
                )

    def testSliceOnMetaFeature(self):
        # We want to make sure that slicing on the newly added feature works, so
        # pulling in slice here.
        with beam.Pipeline() as pipeline:
            fpls = create_fpls()
            metrics = (
                pipeline
                | "CreateTestInput" >> beam.Create(fpls)
                | "WrapFpls" >> beam.Map(wrap_fpl)
                | "ExtractInterestsNum"
                >> meta_feature_extractor.ExtractMetaFeature(get_num_interests)
                | "ExtractSlices"
                >> slice_key_extractor.ExtractSliceKeys(
                    [
                        slicer.SingleSliceSpec(),
                        slicer.SingleSliceSpec(columns=["num_interests"]),
                    ]
                )
                | "FanoutSlices" >> slicer.FanoutSlices()
            )

            util.assert_that(metrics, CheckSliceOnMetaFeatureResult())

    def testGetSparseTensorValue(self):
        sparse_tensor_value = tf.compat.v1.SparseTensorValue(
            indices=[[0, 0, 0], [0, 1, 0], [0, 1, 1]],
            values=["", "one", "two"],
            dense_shape=[1, 2, 2],
        )
        fpl_with_sparse_tensor = types.FeaturesPredictionsLabels(
            input_ref=0, features={}, predictions={}, labels={}
        )

        meta_feature_extractor._set_feature_value(
            fpl_with_sparse_tensor.features, "sparse", sparse_tensor_value
        )
        self.assertEqual(
            ["", "one", "two"],
            meta_feature_extractor.get_feature_value(fpl_with_sparse_tensor, "sparse"),
        )


if __name__ == "__main__":
    tf.test.main()
