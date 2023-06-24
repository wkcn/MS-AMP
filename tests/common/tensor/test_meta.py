# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for ScalingMeta."""

import torch
import pickle
import unittest

from msamp.common.dtype import Dtypes
from msamp.common.dtype import Floating
from msamp.common.tensor import ScalingMeta
from tests.helper import decorator


class ScalingMetaTestCase(unittest.TestCase):
    """Test ScalingMeta."""
    @decorator.cuda_test
    def test_compute_scaling_factor(self):
        """Test compute_scaling_factor in ScalingMeta."""
        amax = torch.zeros([], device='cuda')
        scale = torch.ones((), device='cuda')
        fp_max = Floating.qfp_max[Dtypes.kfloat8_e4m3]
        margin = 0
        scale.copy_(ScalingMeta.compute_scaling_factor(amax, scale, fp_max, margin))

        assert scale.item() == 1.0

        # 2^(floor(log2(448.0/10)))=32
        amax = torch.tensor(10, device='cuda')
        scale = torch.ones((), device='cuda')
        scale.copy_(ScalingMeta.compute_scaling_factor(amax, scale, fp_max, margin))
        assert scale.item() == 32

        # 1/(2^abs(floor(log2(448.0/10000))))
        amax = torch.tensor(10000, device='cuda')
        scale = torch.ones((), device='cuda')
        scale.copy_(ScalingMeta.compute_scaling_factor(amax, scale, fp_max, margin))
        assert scale.item() == 1.0 / 32

    def test_iswarmup_intime(self):
        """Test is_warmup and is_in_time_scaling i ScalingMeta."""
        meta = ScalingMeta(Dtypes.kfloat8_e4m3)
        assert meta.is_warmup()
        assert meta.is_in_time_scaling()
        meta.amax_counter += 1
        assert not meta.is_warmup()
        assert meta.is_in_time_scaling()

        meta = ScalingMeta(Dtypes.kfloat8_e4m3, window_size=2)
        meta.amax_counter += 1
        meta.amax_counter += 1
        assert not meta.is_warmup()
        assert not meta.is_in_time_scaling()

    def test_disable_in_time_scaling(self):
        """Test disable in time scaling in ScalingMeta."""
        bak = ScalingMeta.in_time_scaling
        ScalingMeta.in_time_scaling = False
        meta = ScalingMeta(Dtypes.kfloat8_e4m3)
        self.assertFalse(meta.is_in_time_scaling())
        ScalingMeta.in_time_scaling = bak

    @decorator.cuda_test
    def test_meta_pickle(self):
        """Test pickle and unpickle of ScalingMeta."""
        meta = ScalingMeta(Dtypes.kfloat8_e4m3)
        value = torch.randn((3, 4), device='cuda')
        fp8_value = value.cast(meta.qtype, meta=meta)

        meta2 = pickle.loads(pickle.dumps(meta))

        self.assertEqual(meta.qtype, meta2.qtype)
        self.assertEqual(meta.amax_counter, meta2.amax_counter)
        self.assertEqual(meta.window_size, meta2.window_size)
        self.assertTrue(torch.equal(meta.scale, meta2.scale))
        self.assertTrue(torch.equal(meta.scale_inv, meta2.scale_inv))
        self.assertTrue(torch.equal(meta.amax, meta2.amax))
