# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.

"""A vLLM executor whose worker runs in the co-host device host.

Launch a member with ``--distributed-executor-backend tt_kernel.cohost.vllm_remote.RemoteExecutor``
and ``TT_COHOST_SOCKET`` set. The API server, engine core and scheduler stay in this
process. Each worker call goes over the socket to a real TTWorker in the device host, which
is the only process that opens the chips.
"""

from __future__ import annotations

import os
from concurrent.futures import Future
from typing import Any, Callable

import cloudpickle
from vllm.platforms import current_platform
from vllm.v1.executor.uniproc_executor import UniProcExecutor

from . import rpc

ASYNC_REF = "__cohost_async_ref__"


class _RemoteOutputFuture(Future):
    """Resolved on first ``result()`` by asking the host for the deferred output (as UniProc does)."""

    def __init__(self, client: rpc.Client, ref: int, single_value: bool):
        super().__init__()
        self._client, self._ref, self._single = client, ref, single_value

    def result(self, timeout=None):
        if not super().done():
            try:
                out = self._client.call("get_output", ref=self._ref)
                self.set_result(out if self._single else [out])
            except Exception as e:  # noqa: BLE001 - surfaced through the future
                self.set_exception(e)
        return super().result()


class RemoteExecutor(UniProcExecutor):
    def _init_executor(self) -> None:
        self.driver_worker = None
        self._client = rpc.Client(os.environ["TT_COHOST_SOCKET"],
                                  timeout_s=float(os.environ.get("TT_COHOST_CONNECT_TIMEOUT_S", "600")))
        distributed_init_method, rank, local_rank = self._distributed_args()
        self._client.call("init_worker", kwargs=dict(
            vllm_config=self.vllm_config, local_rank=local_rank, rank=rank,
            distributed_init_method=distributed_init_method, is_driver_worker=True))
        self._client.call("run", method="init_device")
        self._client.call("run", method="load_model")
        current_platform.update_block_size_for_backend(self.vllm_config)

    def collective_rpc(  # type: ignore[override]
        self,
        method: str | Callable,
        timeout: float | None = None,
        args: tuple = (),
        kwargs: dict | None = None,
        non_block: bool = False,
        single_value: bool = False,
    ) -> Any:
        if not isinstance(method, str):
            method = cloudpickle.dumps(method)
        try:
            result = self._client.call("run", method=method, args=args, kwargs=kwargs or {})
        except Exception as e:  # noqa: BLE001 - a non-blocking caller expects a failed future
            if not non_block:
                raise
            fut: Future = Future()
            fut.set_exception(e)
            return fut
        if isinstance(result, dict) and ASYNC_REF in result:
            fut = _RemoteOutputFuture(self._client, result[ASYNC_REF], single_value)
            return fut if non_block else fut.result()
        if not non_block:
            return result if single_value else [result]
        fut = Future()
        fut.set_result(result if single_value else [result])
        return fut

    def check_health(self) -> None:
        self._client.call("ping")

    def shutdown(self) -> None:
        client = getattr(self, "_client", None)
        if client is not None:
            client.close()
            self._client = None
