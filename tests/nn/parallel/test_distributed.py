# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for distributed module."""

import unittest
import torch

from msamp.common.dtype import Dtypes
from msamp.common.tensor import ScalingTensor
from msamp.nn import LinearReplacer
from msamp.optim import LBAdamW
from tests.helper import decorator


class DistributedTestCase(unittest.TestCase):
    """Test Distributed Module."""
    def setUp(self):
        """Hook method for setting up the test fixture before exercising it."""
        torch.manual_seed(1000)

    def tearDown(self):
        """Hook method for deconstructing the test fixture after testing it."""
        pass

    @decorator.cuda_test
    def test_ddp(self):
        """Test DistributedDataParallel."""
        x = torch.randn(4, 4).cuda()
        linear = torch.nn.Linear(4, 8).cuda()
        model = LinearReplacer.replace(linear, Dtypes.kfloat16)
        model = torch.distributed.DistributedDataParallel(model)
        opt = LBAdamW(model.parameters())

        losses = []
        iters = 10
        for _ in range(iters):
            opt.zero_grad()
            output = model(x)
            loss = output.sum()
            loss.backward()
            losses.append(loss.item())
            opt.step()

        for i in range(1, iters):
            assert losses[i] < losses[i - 1]
