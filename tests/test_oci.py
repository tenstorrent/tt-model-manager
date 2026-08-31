# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""OCI layout export/import guards. The tar plumbing runs against a fake `docker`
on PATH, so no daemon is needed; a real round-trip happens in the on-hardware
acceptance run."""

import os
import stat
import tarfile
from pathlib import Path

import pytest

from tt_kernel import oci


def _fake_oci_dir(root: Path) -> Path:
    d = root / "image"
    (d / "blobs" / "sha256").mkdir(parents=True)
    (d / "oci-layout").write_text('{"imageLayoutVersion": "1.0.0"}')
    (d / "index.json").write_text("{}")
    blob = d / "blobs" / "sha256" / ("aa" * 8)
    blob.write_text("layer-bytes")
    return d


def _fake_docker(tmp_path, monkeypatch, script: str) -> Path:
    """Put a fake `docker` first on PATH and hide skopeo."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    exe = bindir / "docker"
    exe.write_text("#!/bin/bash\n" + script)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    monkeypatch.setattr(oci, "_skopeo", lambda: None)
    return bindir


def test_load_rejects_a_non_oci_dir(tmp_path):
    with pytest.raises(oci.OciError, match="oci-layout"):
        oci.load(tmp_path)


def test_save_refuses_non_empty_dest(tmp_path):
    dest = tmp_path / "out"
    dest.mkdir()
    (dest / "junk").write_text("x")
    with pytest.raises(oci.OciError, match="non-empty"):
        oci.save("img", dest)


def test_load_streams_a_dereferenced_tar(tmp_path, monkeypatch):
    """HF-cache snapshots are symlink farms into blobs/ — docker load needs the bytes,
    so the tar must dereference. The fake docker captures what it was piped."""
    src = _fake_oci_dir(tmp_path)
    # replace the blob with a symlink to elsewhere, as snapshot_download lays out
    blob = src / "blobs" / "sha256" / ("aa" * 8)
    real = tmp_path / "store" / "blob0"
    real.parent.mkdir()
    real.write_text("layer-bytes")
    blob.unlink()
    blob.symlink_to(real)

    captured = tmp_path / "captured.tar"
    _fake_docker(tmp_path, monkeypatch, f'[ "$1" = load ] && cat > {captured}\nexit 0\n')

    oci.load(src)

    with tarfile.open(captured) as tar:
        member = tar.getmember("blobs/sha256/" + "aa" * 8)
        assert member.isfile() and not member.issym()      # dereferenced
        assert tar.extractfile(member).read() == b"layer-bytes"
        names = tar.getnames()
        assert "oci-layout" in names and "index.json" in names


def test_save_untars_docker_save_output(tmp_path, monkeypatch):
    """docker save emits an OCI-layout tar; save() must stream-extract it."""
    payload = tmp_path / "payload"
    inner = _fake_oci_dir(payload)
    tarball = tmp_path / "img.tar"
    with tarfile.open(tarball, "w") as tar:
        for p in sorted(inner.rglob("*")):
            tar.add(p, arcname=str(p.relative_to(inner)), recursive=False)
    _fake_docker(tmp_path, monkeypatch, f'[ "$1" = save ] && cat {tarball}\nexit 0\n')

    dest = tmp_path / "exported"
    oci.save("tt-model/x:dev", dest)
    assert (dest / "oci-layout").exists()
    assert (dest / "blobs" / "sha256" / ("aa" * 8)).read_text() == "layer-bytes"


def test_save_cleans_up_when_docker_fails(tmp_path, monkeypatch):
    _fake_docker(tmp_path, monkeypatch, 'exit 1\n')
    dest = tmp_path / "exported"
    with pytest.raises(oci.OciError):
        oci.save("tt-model/x:dev", dest)
    assert not dest.exists()


# Narrowed by MESSAGE, not module: a warning filter's module field matches where the
# warning is ISSUED (the caller — tt_kernel.oci), not the module that raises it, so
# "ignore::DeprecationWarning:tarfile" does not match. Verified: it does not suppress.
@pytest.mark.filterwarnings(
    "ignore:Python 3.14 will, by default, filter extracted tar archives:DeprecationWarning"
)
def test_extraction_works_without_the_data_filter(tmp_path, monkeypatch):
    """`extractall(filter="data")` only exists from Python 3.11.4, and this package
    declares requires-python >=3.9 — there it is a TypeError. The fallback must extract
    the same content."""
    import tarfile

    from tt_kernel import oci as oci_mod

    payload = tmp_path / "payload"
    inner = _fake_oci_dir(payload)
    tarball = tmp_path / "img.tar"
    with tarfile.open(tarball, "w") as tar:
        for p in sorted(inner.rglob("*")):
            tar.add(p, arcname=str(p.relative_to(inner)), recursive=False)

    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    dest = tmp_path / "out"
    dest.mkdir()
    with tarfile.open(tarball) as tar:
        oci_mod._extract_safely(tar, dest)
    assert (dest / "oci-layout").is_file()
    assert (dest / "blobs" / "sha256" / ("aa" * 8)).read_text() == "layer-bytes"


@pytest.mark.parametrize("evil", ["../escape", "/etc/passwd"])
# Narrowed by MESSAGE, not module: a warning filter's module field matches where the
# warning is ISSUED (the caller — tt_kernel.oci), not the module that raises it, so
# "ignore::DeprecationWarning:tarfile" does not match. Verified: it does not suppress.
@pytest.mark.filterwarnings(
    "ignore:Python 3.14 will, by default, filter extracted tar archives:DeprecationWarning"
)
def test_the_fallback_refuses_members_that_escape_the_destination(tmp_path, monkeypatch, evil):
    """"It comes from the local docker daemon" is not a reason to extract ../ paths."""
    import tarfile

    from tt_kernel import oci as oci_mod

    # Built with an explicit TarInfo: tar.add() normalises a leading slash away, so
    # arcname alone cannot produce the absolute-path case this guards against.
    import io

    tarball = tmp_path / "evil.tar"
    data = b"x"
    with tarfile.open(tarball, "w") as tar:
        info = tarfile.TarInfo(name=evil)
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))

    monkeypatch.delattr(tarfile, "data_filter", raising=False)
    with tarfile.open(tarball) as tar:
        with pytest.raises(oci_mod.OciError, match="unsafe tar member"):
            oci_mod._extract_safely(tar, tmp_path / "out")


def test_skopeo_load_names_the_same_reference_save_wrote(tmp_path, monkeypatch):
    """save() writes oci:<dest>:latest; load() must read the same reference back."""
    calls = []
    monkeypatch.setattr(oci, "_skopeo", lambda: "/usr/bin/skopeo")
    monkeypatch.setattr(oci.subprocess, "run",
                        lambda argv, **k: calls.append(argv) or None)
    src = _fake_oci_dir(tmp_path)
    oci.load(src, expect_tag="tt-model/x:abc")
    assert calls[0][2] == f"oci:{src}:latest"
