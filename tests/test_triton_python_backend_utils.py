# Copyright 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import importlib.util
import struct
import unittest
from pathlib import Path

import numpy as np


_UTILS_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "resources"
    / "triton_python_backend_utils.py"
)
_SPEC = importlib.util.spec_from_file_location("triton_python_backend_utils", _UTILS_PATH)
pb_utils = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(pb_utils)


class DeserializeBytesTensorTest(unittest.TestCase):
    def test_deserialize_valid_bytes_tensor(self):
        encoded = struct.pack("<I", 3) + b"foo" + struct.pack("<I", 3) + b"bar"

        result = pb_utils.deserialize_bytes_tensor(encoded)

        self.assertEqual(result.dtype, np.object_)
        self.assertEqual(result.tolist(), [b"foo", b"bar"])

    def test_deserialize_rejects_incomplete_length_field(self):
        with self.assertRaisesRegex(ValueError, "incomplete length field"):
            pb_utils.deserialize_bytes_tensor(b"\x03\x00")

    def test_deserialize_rejects_string_extending_beyond_buffer(self):
        encoded = struct.pack("<I", 5) + b"foo"

        with self.assertRaisesRegex(ValueError, "string extends beyond buffer"):
            pb_utils.deserialize_bytes_tensor(encoded)


if __name__ == "__main__":
    unittest.main()
