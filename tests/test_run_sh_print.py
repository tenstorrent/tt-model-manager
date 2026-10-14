# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""run.sh's TT_MODEL_PRINT=1 path prints a command that pastes back as the same argv."""

import os
import shlex
import subprocess
import sys

from tt_kernel import packaging
from tt_kernel.manifest import Mesh, Resources, WeightsRef

TRICKY = ["--chat-template", "a b; echo pwned $HOME 'q' \"dq\" *"]


def test_print_quotes_args_so_they_round_trip(tmp_path):
    model_py = tmp_path / "model.py"
    model_py.write_text("class C: pass\n")
    b = tmp_path / "b"
    packaging.stage_thin_package(
        b, name="m", arch="wormhole_b0", model_py=model_py, tt_kernel_version="0",
        vllm_metadata={"arch": "A", "main_class": "model:C"}, weights=WeightsRef(repo="o/w"),
        mesh=Mesh(devices=1, topology="N150"),
        resources=Resources(max_num_seqs=1, block_size=64, extra_args=TRICKY),
    )
    (b / "venv" / "bin").mkdir(parents=True)
    py = b / "venv" / "bin" / "python"
    py.write_text(f'#!/bin/bash\nexec {sys.executable} "$@"\n')
    py.chmod(0o755)
    ttnn = tmp_path / "site" / "ttnn"
    (ttnn / "build" / "lib").mkdir(parents=True)
    (ttnn / "__init__.py").write_text("")
    (ttnn / "build" / "lib" / "_ttnncpp.so").write_text("")
    # run.sh's SFPI gate runs before the print path and a MISSING SFPI is fatal even under
    # TT_MODEL_PRINT=1, so give the fake ttnn a local SFPI (no sfpi-version => no version check).
    sfpi = ttnn / "runtime" / "sfpi"
    (sfpi / "compiler" / "bin").mkdir(parents=True)
    (sfpi / "include").mkdir()
    (sfpi / "include" / "sfpi_lib.h").write_text("// test\n")
    gxx = sfpi / "compiler" / "bin" / "riscv-tt-elf-g++"
    gxx.write_text("#!/bin/sh\necho 'riscv-tt-elf-g++ (tenstorrent/sfpi:7.78.0[935]) 15.1.0'\n")
    gxx.chmod(0o700)

    env = {**os.environ, "PYTHONPATH": str(tmp_path / "site"), "TT_MODEL_PRINT": "1"}
    r = subprocess.run(["bash", str(b / "run.sh"), "--extra", "x y"], env=env,
                       capture_output=True, text=True, check=True)
    argv = shlex.split(r.stdout.splitlines()[-1])
    i = argv.index(TRICKY[0])
    assert argv[i:i + 2] == TRICKY
    assert argv[-2:] == ["--extra", "x y"]
