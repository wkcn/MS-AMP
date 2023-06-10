import math
import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import torch.distributed as dist
from msamp.common.tensor import ScalingTensor, ScalingMeta
from msamp.common.dtype import Dtypes, Floating
from msamp.common.utils import TransformerEngineWrapper
from msamp.operators.dist_op import DistOp


class ScalingTensorReducer:
    def __init__(self, parameters, process_group, bucket_bytes_cap):
        parameters = list(parameters)
        if not all(isinstance(p, ScalingTensor) for p in parameters):
            raise ValueError("All parameters must be ScalingTensor")
        # check the devices of parameters are the same
        if not all(p.device == parameters[0].device for p in parameters):
            raise ValueError("All parameters must be on the same device")
        self.device = parameters[0].device
        self.parameters = parameters
        self.param_to_id = {p: i for i, p in enumerate(parameters)}
        self.buffer = self._create_buffer(parameters)
        self.bucket_bytes_cap = bucket_bytes_cap
        self.process_group = process_group
        self._build_buckets(parameters)
        self._register_backward_hooks()
        self.bucket_unreduced_param_ids = dict()
        self.reduction_stream = torch.cuda.Stream()
        self.dist_handles = []

    def get_param_id(self, param):
        return self.param_to_id[param]

    def reset_buckets(self):
        self.wait()
        if len(self.bucket_unreduced_param_ids) > 0:
            raise RuntimeError("some gradients are not reduced: {}".format(list(self.bucket_unreduced_param_ids.keys())))
        self.bucket_unreduced_param_ids = {k: set(v) for k, v in self.bucket_to_param_ids.items()}

    def wait(self):
        for handle in self.dist_handles:
            handle.wait()
        self.dist_handles = []
        torch.cuda.current_stream().wait_stream(self.reduction_stream)

    def _create_buffer(self, parameters):
        buffer_size = sum(p.numel() for p in parameters)
        return torch.empty((buffer_size, ), dtype=torch.uint8, device=self.device)

    def _build_buckets(self, parameters):
        bucket_bytes = 0
        total_bytes = 0
        bucket_id = 0
        bucket_offset = 0
        param_id_to_bucket_id = {}
        bucket_to_param_ids = {}
        bucket_to_range = {}
        param_id_to_range = {}
        for p in parameters[::-1]:
            param_id = self.get_param_id(p)
            nbytes = p.numel()
            param_id_to_bucket_id[param_id] = bucket_id
            bucket_to_param_ids.setdefault(bucket_id, []).append(param_id)
            bucket_bytes += nbytes
            total_bytes += nbytes
            if bucket_bytes >= self.bucket_bytes_cap:
                bucket_to_range[bucket_id] = (bucket_offset, bucket_offset + bucket_bytes)
                bucket_id += 1
                bucket_bytes = 0

        # the last bucket, if not empty
        if bucket_bytes > 0:
            bucket_to_range[bucket_id] = (bucket_offset, bucket_offset + bucket_bytes)
        self.param_id_to_bucket_id = param_id_to_bucket_id
        self.bucket_to_param_ids = bucket_to_param_ids
        self.bucket_to_range = bucket_to_range

    def _register_backward_hooks(self):
        for p in self.parameters:
            p.register_backward_post_hook(self._get_backward_hook(p))

    def _get_backward_hook(self, param):
        param_id = self.get_param_id(param)
        bucket_id = self.param_id_to_bucket_id[param_id]
        def hook_fn(*args, **kwargs):
            unreduced_param_ids = self.bucket_unreduced_param_ids[bucket_id]
            try:
                unreduced_param_ids.remove(param_id)
            except KeyError:
                raise RuntimeError("gradient is already reduced")
            if len(unreduced_param_ids) == 0:
                # the bucket is full, reduce it
                self._reduce_bucket(bucket_id)
                self.bucket_unreduced_param_ids.pop(bucket_id)
        return hook_fn

    def _reduce_bucket(self, bucket_id):
        self.reduction_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.reduction_stream):
            # step 1: collect the gradients
            param_ids = self.bucket_to_param_ids[bucket_id]
            params = [self.parameters[i] for i in param_ids]
            grads = [p.grad for p in params]
            metas = [g.meta for g in grads]

            # step 2: synchronize the amax
            # shape of amaxs: (n, )
            amaxs = torch.stack([g.max().float() for g in grads])
            scales = torch.stack([meta.scale for meta in metas])
            # convert NAN to INF since NCCL-ReduceMax ignores NAN
            # notice: nan and posinf must be INF
            amaxs.nan_to_num_(nan=torch.inf, posinf=torch.inf)
            dist.all_reduce(amaxs, op=dist.ReduceOp.MAX, group=self.process_group)

            # step 3: update scaling factor
            wgrad_qtype = Dtypes.kfloat8_e4m3
            fp_max = Floating.qfp_max[wgrad_qtype]
            world_size = dist.get_world_size(self.process_group)
            pre_scale = 1.0 / math.sqrt(world_size)
            # shape of sf: (n, )
            sf = ScalingMeta.compute_scaling_factor(amaxs, scales, fp_max, margin=0)
            sf.mul_(pre_scale)

            # update meta.amax[0] to global amax
            for meta, amax in zip(metas, amaxs):
                meta.amax[0] = amax

            # step 4: quantize the gradients to FP8
            dummy_amax = torch.empty((1, ), dtype=torch.float32, device=self.device)
            bucket_range = self.bucket_to_range[bucket_id]
            bucket_start, bucket_end = bucket_range
            bucket_offset = bucket_start
            for i, (grad, meta) in enumerate(zip(grads, metas)):
                fp8_grad = TransformerEngineWrapper.cast_to_fp8(
                    grad.view(1, -1),
                    meta.scale,
                    dummy_amax,
                    meta.scale_inv,
                    meta.qtype,
                )
                meta.scale_inv.data.copy_(torch.reciprocal(meta.scale))
                grads[i] = None
                # copy fp8_grad to buffer
                grad_numel = grad.numel()
                buf = self.buffer.narrow(0, bucket_offset, grad_numel).view_as(fp8_grad)
                buf.copy_(fp8_grad)
                params[i].grad = ScalingTensor(buf, meta)
                params[i].grad.div_(world_size)
                bucket_offset += grad_numel

            # step 5: allreduce the gradients
            flat_fp8_grads = self.buffer.narrow(0, bucket_start, bucket_end - bucket_start)
            if True:
                # [TODO] support native distributed API with FP8 support
                handle = dist.all_reduce(flat_fp8_grads, op=dist.ReduceOp.SUM, group=self.process_group, async_op=True)
                self.dist_handles.append(handle)
            else:
                if dist.get_world_size() == dist.get_world_size(self.process_group):
                    DistOp.all_reduce(flat_fp8_grads, qtype=wgrad_qtype, op=dist.ReduceOp.SUM)
                else:
                    raise RuntimeError("msamp.nn.parallel.DistributedDataParallel only supports `self.process_group is None`")


class _DDPSink(torch.autograd.Function):
    @staticmethod
    def forward(ctx, reducer, *inputs):
        ctx.set_materialize_grads(False)
        ctx.reducer = reducer
        reducer.reset_buckets()
        return inputs
    @staticmethod
    def backward(ctx, *grad_outputs):
        ctx.reducer.wait()
        return (None, *grad_outputs)


class DistributedDataParallel(torch.nn.parallel.DistributedDataParallel):
    def __init__(self, module, **kwargs):
        super().__init__(module, **kwargs)
        scaling_params = [p for p in self.parameters() if p.requires_grad and isinstance(p, ScalingTensor)]
        self.scaling_tensor_reducer = ScalingTensorReducer(scaling_params, self.process_group, self.bucket_bytes_cap)
    def forward(self, *inputs, **kwargs):
        if torch.is_grad_enabled():
            inputs = _DDPSink.apply(self.scaling_tensor_reducer, *inputs)
        out = super().forward(*inputs, **kwargs)
        return out


torch.nn.parallel.DistributedDataParallel = DistributedDataParallel
