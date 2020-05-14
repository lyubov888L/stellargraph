# -*- coding: utf-8 -*-
#
# Copyright 2020 Data61, CSIRO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = [
    "SlidingFeaturesNodeGenerator",
    "SlidingFeaturesNodeSequence",
]

import numpy as np
from . import Generator
from tensorflow.keras.utils import Sequence

from ..core.validation import require_integer_in_range


class SlidingFeaturesNodeGenerator(Generator):
    def __init__(
        self, G, window_size,
    ):
        require_integer_in_range(window_size, "window_size", min_val=1)

        self.graph = G

        node_type = G.unique_node_type(
            "G: expected a graph with a single node type, found a graph with node types: %(found)s"
        )
        self._features = G.node_features(node_type=node_type)

        self._window_size = window_size

    def num_batch_dims(self):
        return 1

    def flow(self, iloc_range, batch_size, target_distance=None):
        if target_distance is not None:
            require_integer_in_range(target_distance, "target_distance")

        if batch_size is not None:
            require_integer_in_range(batch_size, "batch_size")

        if not isinstance(iloc_range, range):
            raise ValueError("todo")

        return SlidingFeaturesNodeSequence(
            self._features, iloc_range, self._window_size, target_distance, batch_size
        )


class SlidingFeaturesNodeSequence(Sequence):
    def __init__(self, features, iloc_range, window_size, target_distance, batch_size):
        self._num_nodes = features.shape[0]
        self._num_sequence_samples = len(iloc_range)
        self._num_sequence_variates = features.shape[2:]

        # have the first dimension be the slicing dimension
        self._features = np.moveaxis(features, 1, 0)[iloc_range, ...]

        self._times = times
        self._window_size = window_size
        self._target_distance = target_distance
        self._batch_size = batch_size

        query_length = window_size + (0 if target_distance is None else target_distance)
        self._num_windows = self._num_sequence_samples - query_length + 1

    def __len__(self):
        return int(np.ceil(self._num_windows / self._batch_size))

    def __getitem__(self, batch_num):
        first_start = batch_num * self._batch_size
        last_start = min((batch_num + 1) * self._batch_size, self._num_windows)

        has_targets = self._target_distance is not None

        arrays = []
        targets = [] if has_targets else None
        for start in range(first_start, last_start):
            end = start + self._window_size
            arrays.append(self._features[start:end])
            if has_targets:
                targets.append(self._features[end + self._target_distance - 1])

        this_batch_size = last_start - first_start

        batch_feats = np.moveaxis(np.stack(arrays), 1, 2)
        assert (
            batch_feats.shape
            == (this_batch_size, self._num_nodes, self._window_size)
            + self._num_sequence_variates
        )

        if has_targets:
            batch_targets = np.stack(targets)
            assert (
                batch_targets.shape
                == (this_batch_size, self._num_nodes) + self._num_sequence_variates
            )
        else:
            batch_targets = None

        return [batch_feats], batch_targets
