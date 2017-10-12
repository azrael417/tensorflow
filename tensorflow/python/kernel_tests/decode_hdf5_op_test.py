# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for DecodeHDF5 op from parsing_ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.ops import parsing_ops
from tensorflow.python.platform import test


class DecodeHDF5OpTest(test.TestCase):

  def _test(self, args, expected_out=None, expected_err_re=None):
    with self.test_session() as sess:
      decode = parsing_ops.decode_hdf5(**args)

      if expected_err_re is None:
        out = sess.run(decode)

        for i, field in enumerate(out):
          if field.dtype == np.float32 or field.dtype == np.float64:
            self.assertAllClose(field, expected_out[i])
          else:
            self.assertAllEqual(field, expected_out[i])

      else:
        with self.assertRaisesOpError(expected_err_re):
          sess.run(decode)

  def testSimple(self):
    args = {
        "records": "(3)[1,2,3]",
        "output": [tf.int32]
    }

    expected_out = [[1, 2, 3]]

    self._test(args, expected_out)


if __name__ == "__main__":
  test.main()
