import torch
import msamp
from msamp.common.dtype import Dtypes
from msamp.common.tensor import ScalingTensor
from msamp.nn import ScalingModule, ScalingParameter
import transformer_engine.pytorch as te
import transformer_engine_extensions as tex
tec = te.cpp_extensions
try:
    from transformer_engine.pytorch.module.base import TransformerEngineBaseModule
except:
    from transformer_engine.pytorch.module import TransformerEngineBaseModule


def set_activation_dtype(self, inp: torch.Tensor) -> None:
    """Get activation data type for AMP."""
    # Native AMP (`torch.autocast`) gets highest priority
    if torch.is_autocast_enabled():
        self.activation_dtype = torch.get_autocast_gpu_dtype()
        return

    # All checks after this have already been performed once, thus skip
    # We assume that user doesn't change input types across iterations
    if hasattr(self, "activation_dtype"):
        return

    assert all(
        (
            (inp.dtype == param.dtype) if param is not None and not isinstance(param, ScalingTensor) else True
            for param in self.parameters()
        )
    ), (
        "Data type for activations and weights must "
        "match when outside of autocasted region"
    )
    assert all(
        (
            (inp.dtype == buf.dtype) if buf is not None else True
            for buf in self.buffers()
        )
    ), (
        "Data type for activations and buffers must "
        "match when outside of autocasted region"
    )
    self.activation_dtype = inp.dtype

TransformerEngineBaseModule.set_activation_dtype = set_activation_dtype


TE2MSAMP_DTYPE = {
    tex.DType.kFloat8E4M3: Dtypes.kfloat8_e4m3,
    tex.DType.kFloat8E5M2: Dtypes.kfloat8_e5m2,
    tex.DType.kBFloat16: Dtypes.kbfloat16,
    tex.DType.kFloat16: Dtypes.kfloat16,
    tex.DType.kFloat32: Dtypes.kfloat32,
}

# === lower functions ===
old_fused_cast_transpose = tex.fused_cast_transpose
@torch.no_grad()
def fused_cast_transpose(input, scale, amax, scale_inv, input_cast, input_transpose, otype):
    if isinstance(input, ScalingTensor):
        qtype = TE2MSAMP_DTYPE[otype]
        if input_transpose is not None:
            sv = input.cast(qtype)
            # data should be contiguous, and TE does not check it.
            st = sv.t().contiguous()
            v, t = sv.value, st.value
            input_transpose.data = t
        else:
            sv = input.cast(qtype)
            v = sv.value

        if input_cast is not None:
            input_cast.data = v
        scale_inv.copy_(sv.meta.scale_inv)
    else:
        old_fused_cast_transpose(input, scale, amax, scale_inv, input_cast, input_transpose, otype)

tex.fused_cast_transpose = fused_cast_transpose


# === upper functions ===

old_fp8_cast_transpose_fused = te.cpp_extensions.fp8_cast_transpose_fused
@torch.no_grad()
def fp8_cast_transpose_fused(inp, fp8_meta_tensor, fp8_tensor, dtype, cast_out=None, transpose_out=None):
    if isinstance(inp, ScalingTensor):
        qtype = TE2MSAMP_DTYPE[dtype]
        if transpose_out is not None:
            sv = inp.cast(qtype)
            # data should be contiguous, and TE does not check it.
            st = sv.t().contiguous()
            v, t = sv.value, st.value
            transpose_out.data = t
        else:
            sv = inp.cast(qtype)
            v = sv.value

        if cast_out is not None:
            cast_out.data = v
        fp8_meta_tensor.scale_inv[fp8_tensor].copy_(sv.meta.scale_inv)
        return v, t

    return old_fp8_cast_transpose_fused(inp, fp8_meta_tensor, fp8_tensor, dtype, cast_out, transpose_out)

old_cast_to_fp8 = te.cpp_extensions.cast_to_fp8
@torch.no_grad()
def cast_to_fp8(inp, fp8_meta_tensor, fp8_tensor, otype, out=None):
    if isinstance(inp, ScalingTensor):
        qtype = TE2MSAMP_DTYPE[otype]
        sv = inp.cast(qtype)
        v = sv.value
        if out is not None:
            out.data = v
        fp8_meta_tensor.scale_inv[fp8_tensor].copy_(sv.meta.scale_inv)
        return v

    if out is None:
        return old_cast_to_fp8(inp, fp8_meta_tensor, fp8_tensor, otype)
    return old_cast_to_fp8(inp, fp8_meta_tensor, fp8_tensor, otype, out)
te.cpp_extensions.cast_to_fp8 = cast_to_fp8 


mods = [te.module]
mnames = ['linear', 'layernorm_linear', 'layernorm_mlp']
for mname in mnames:
    if hasattr(te.module, mname):
        mods.append(getattr(te.module, mname))
for m in mods:
    for name in ['fp8_cast_transpose_fused', 'cast_to_fp8']:
        setattr(m, name, globals()[name])

class CtxWrapper:
    def __init__(self, ctx):
        self.__dict__['ctx'] = ctx
    def __getattr__(self, name):
        return self.__dict__.get(name, getattr(self.__dict__['ctx'], name))
    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            setattr(self.ctx, name, value)
    def save_for_backward(self, *args):
        torch_args = []
        msamp_args = []
        for a in args:
            if isinstance(a, ScalingTensor):
                msamp_args.append(a)
                torch_args.append(None)
            else:
                torch_args.append(a)
                msamp_args.append(None)
        self.ctx.save_for_backward(*torch_args)
        self.ctx.msamp_args = msamp_args
    @property
    def saved_tensors(self):
        tensors = list(self.ctx.saved_tensors)
        for i, v in enumerate(self.ctx.msamp_args):
            if v is not None:
                tensors[i] = v
        return tensors


def override_func(mod, func_name):
    old_func = getattr(mod, func_name)
    assert issubclass(old_func, torch.autograd.Function), (func_name, old_func)

    class Func(torch.autograd.Function):
        @staticmethod
        def forward(ctx, place_holder, *args):
            scaling_tensors = []
            for i, a in enumerate(args):
                if isinstance(a, ScalingTensor):
                    scaling_tensors.append((i, a))
            if ctx is not None:
                ctx.scaling_tensors = scaling_tensors
                ctx = CtxWrapper(ctx)
            return old_func.forward(ctx, *args)

        @staticmethod
        def backward(ctx, *args):
            ctx = CtxWrapper(ctx)
            grads = list(old_func.backward(ctx, *args))
            for i, v in ctx.scaling_tensors:
                if not v.requires_grad:
                    continue
                assert grads[i] is not None
                if v.grad is None:
                    v.grad = grads[i]
                else:
                    v.grad += grads[i]
                v.backward_grad_update(v.grad)
                grads[i] = None
            return (None,) + tuple(grads)

    class Wrapper:
        EMPTY_TENSOR = torch.tensor([], requires_grad=True)

        @staticmethod
        def forward(ctx, *args):
            return Func.forward(ctx, Wrapper.EMPTY_TENSOR.detach(), *args)

        @staticmethod
        def apply(*args):
            return Func.apply(Wrapper.EMPTY_TENSOR, *args)

    setattr(mod, func_name, Wrapper)

class ScalingModuleBack(ScalingModule):
    def extra_repr(self):
        s = super().extra_repr()
        s += '('
        for k, v in self.named_parameters():
            s += f'{k}: ({v.qtype.name}, {tuple(v.shape)}), '
        s += ')'
        return s

IS_MSAMP_MODULE_STR = '_is_msamp_module'
class ScalingModuleFront:
    def _msamp_weight_cache(self, fn, args):
        if not getattr(self, IS_MSAMP_MODULE_STR, False):
            rtn = fn(*args)
        else:
            # MS-AMP
            old_fp8_weight_shapes = self.fp8_weight_shapes
            self.fp8_weight_shapes = [(0, 0)] * len(old_fp8_weight_shapes)
            # create empty tensor as placeholder
            rtn = fn(*args)
            self.fp8_weight_shapes = old_fp8_weight_shapes
        return rtn

    def set_fp8_weights(self):
        # when is_first_microbatch is not None
        # call every microbatch
        # cache weight_fp8, weight_t_fp8 for gradient accumulation
        # set_fp8_weights will clean up the cache
        if not getattr(self, IS_MSAMP_MODULE_STR, False):
            TransformerEngineBaseModule.set_fp8_weights(self)
        else:
            for i, shape in enumerate(self.fp8_weight_shapes, start=1):
                weight_cast_attr = f"weight{i}_fp8"
                weight_transpose_attr = f"weight{i}_t_fp8"

                if (
                    hasattr(self, weight_cast_attr)
                    and getattr(self, weight_cast_attr).shape == shape
                ):
                    return

                setattr(
                    self,
                    weight_cast_attr,
                    torch.empty(
                        (0, 0),
                        device=torch.cuda.current_device(),
                        dtype=torch.uint8,
                    ),
                )
                setattr(
                    self,
                    weight_transpose_attr,
                    torch.empty(
                        (0, 0),
                        device=torch.cuda.current_device(),
                        dtype=torch.uint8,
                    ),
                )


    def get_fp8_weights_empty_tensors(self, is_first_microbatch):
        # when is_first_microbatch is None, create empty tensors
        return self._msamp_weight_cache(
            TransformerEngineBaseModule.get_fp8_weights_empty_tensors,
            (self, is_first_microbatch))

MODULE_NAMES = ['linear', 'layernorm_linear', 'layernorm_mlp']
CLS_NAMES = ['Linear', 'LayerNormLinear', 'LayerNormMLP']

for mod_name, name in zip(MODULE_NAMES, CLS_NAMES):
    func_name = '_' + name
    mod = getattr(te.module, mod_name, te.module)
    override_func(mod, func_name)
    msamp_name = 'MSAMP' + name
    te_module = getattr(te, name)
    assert issubclass(te_module, TransformerEngineBaseModule), te_module
    cls = type(msamp_name, (ScalingModuleFront, te_module, ScalingModuleBack), {})
    setattr(te, msamp_name, cls)

MSAMP_MODULES_WEIGHT_NAMES = {
    te.MSAMPLinear: ['weight'],
    te.MSAMPLayerNormLinear: ['weight'],
    te.MSAMPLayerNormMLP: ['fc1_weight', 'fc2_weight'],
}

te.Linear = te.MSAMPLinear
te.transformer.Linear = te.MSAMPLinear
te.transformer.LayerNormLinear = te.MSAMPLayerNormLinear
te.transformer.LayerNormMLP = te.MSAMPLayerNormMLP
if hasattr(te, 'attention'):
    te.attention.Linear = te.MSAMPLinear
    te.attention.LayerNormLinear = te.MSAMPLayerNormLinear

def replace_msamp_te_net(model):
    def _replace(model):
        for mod in MSAMP_MODULES_WEIGHT_NAMES:
            if isinstance(model, mod):
                setattr(mod, IS_MSAMP_MODULE_STR, True)
                weight_names = MSAMP_MODULES_WEIGHT_NAMES[mod]
                for wname in weight_names:
                    if not hasattr(model, wname):
                        continue
                    # assert hasattr(model, wname), [k for k, v in model.named_parameters()]
                    weight = getattr(model, wname)
                    dtype = weight.dtype
                    requires_grad = weight.requires_grad
                    sp = ScalingParameter(weight.data.cast(Dtypes.kfloat16), requires_grad=requires_grad)
                    # release the old weight
                    weight.data = torch.tensor([])
                    setattr(model, wname, sp)
        else:
            for child_name, child in list(model.named_children()):
                setattr(model, child_name, _replace(child))
        return model
    model = _replace(model)

    fp8_named_weights = [(k, p) for k, p in model.named_parameters() if isinstance(p, ScalingParameter)]
    fp8_names = [k for k, _ in fp8_named_weights]
    torch.nn.parallel.DistributedDataParallel._set_params_and_buffers_to_ignore_for_model(model, fp8_names)
    # empty cache
    torch.cuda.empty_cache()
    return model
