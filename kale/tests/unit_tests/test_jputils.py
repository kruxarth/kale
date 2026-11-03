# Copyright 2026 The Kubeflow Authors.
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

import contextlib

import pytest
from testfixtures import mock

from kale.common import jputils as ju


def _output_display(data):
    # `data` must be a list
    return [{"output_type": "display_data", "data": data}]


@pytest.mark.parametrize(
    "outputs,target",
    [
        ([], ""),
        # ---
        (_output_display({"image/png": "bytes"}), ju.IMAGE_HTML_TEMPLATE.format("", "bytes")),
        # ---
        (_output_display({"text/html": "bytes"}), "bytes"),
        # ---
        (_output_display({"text/plain": "bytes"}), ju.TEXT_HTML_TEMPLATE.format("bytes")),
        # ---
        (
            _output_display({"application/javascript": "bytes"}),
            ju.JAVASCRIPT_HTML_TEMPLATE.format("bytes"),
        ),
    ],
)
def test_generate_html_output(outputs, target):
    """Tests html artifact generation from cell outputs."""
    assert target == ju.generate_html_output(outputs)


@mock.patch("kale.common.jputils.process_outputs", new=lambda x: x)
def test_run_code():
    """Test that Python code runs inside a jupyter kernel successfully."""
    # test standard code
    code = ("a = 3\nprint(a)",)
    ju.run_code(code)

    # test magic command
    code = ("%%time\nprint('Some dull code')",)
    ju.run_code(code)


@mock.patch("kale.common.jputils.process_outputs", new=lambda x: x)
def test_run_code_streams_output(capfd):
    """Test that stdout and stderr of the user code reach the process logs."""
    code = ("import sys\nprint('to stdout')\nprint('to stderr', file=sys.stderr)",)
    ju.run_code(code)

    out, err = capfd.readouterr()
    assert "to stdout" in out
    assert "to stderr" in err


@mock.patch("kale.common.jputils.process_outputs", new=lambda x: x)
def test_run_code_stops_at_first_error(tmp_path):
    """Test that a failing cell stops the run and exits with an error."""
    marker = tmp_path / "ran"
    code = ("raise ValueError('boom')", f"open({str(marker)!r}, 'w').close()")

    with pytest.raises(SystemExit) as exc_info:
        ju.run_code(code)

    assert exc_info.value.code == -1
    assert not marker.exists()


@pytest.mark.parametrize("code", [("print('ok')",), ("raise ValueError('boom')",)])
@mock.patch("kale.common.jputils.process_outputs", new=lambda x: x)
def test_run_code_shuts_down_kernel(code):
    """Test that the kernel is shut down whether the code succeeds or fails."""
    preprocessors = []

    class RecordingPreprocessor(ju.StreamingExecutePreprocessor):
        def preprocess(self, *args, **kwargs):
            preprocessors.append(self)
            return super().preprocess(*args, **kwargs)

    with (
        mock.patch("kale.common.jputils.StreamingExecutePreprocessor", RecordingPreprocessor),
        contextlib.suppress(SystemExit),
    ):
        ju.run_code(code)

    assert len(preprocessors) == 1
    # nbclient drops its kernel manager and client once the kernel is shut down
    assert preprocessors[0].km is None
    assert preprocessors[0].kc is None
