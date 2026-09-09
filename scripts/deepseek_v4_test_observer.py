"""Cooperative teardown for the V4 benchmark's status-query observer."""

import asyncio


async def drain_observer(observer, *, timeout=60):
    """Let an in-flight RPC finish; canceling it can poison Engine's channel.

    Call after the observed request tasks have finished or been canceled, so
    the observer's next loop-condition check exits naturally. A timeout fails
    the test and its owner must shut down the Engine/server, not reuse it.
    """
    await asyncio.wait_for(asyncio.shield(observer), timeout=timeout)
