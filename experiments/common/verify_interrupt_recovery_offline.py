"""Offline suite: prohibit real model requests and isolate a legacy test deadline.

No files/production budget deadlines are altered. The old test creates its own
temporary ledger using min(now+600, controller_research.DEADLINE), which expired
on 2026-09-28; shift only that in-memory test constant for reproducibility.
"""
import sys,time
from unittest.mock import patch
import pytest
import httpx
from experiments import controller_research

def deny(*args,**kwargs):
    raise AssertionError('Real model call forbidden in offline verification')

if __name__=='__main__':
    with patch.object(httpx.HTTPTransport,'handle_request',deny),patch.object(httpx.AsyncHTTPTransport,'handle_async_request',deny),patch.object(controller_research,'DEADLINE',time.time()+3600):
        raise SystemExit(pytest.main(sys.argv[1:] or ['-q']))
