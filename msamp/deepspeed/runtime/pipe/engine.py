import deepspeed
from deepspeed.runtime.pipe.engine import *
from ..engine import *
from deepspeed import comm as dist
from msamp.nn import model_state

_origin_DeepSpeedPipelineEngine = deepspeed.runtime.pipe.engine.PipelineEngine


class MSAMPPipelineEngine(MSAMPDeepSpeedEngine, _origin_DeepSpeedPipelineEngine):
    """ A training engine hybrid pipeline, data, and model parallel training.

    This engine is created by ``deepspeed.initialize()`` when a :class:`PipelineModule`
    is provided.
    """

    def _exec_reduce_tied_grads(self):
        # We need to run this first to write to self.averaged_gradients;
        # since this class turns `enable_backward_allreduce` off,
        # `self.overlapping_partition_gradients_reduce_epilogue()` defined in the DeepSpeedEngine
        # never actually runs. I suspect this is because of efficiency problems; get_flat_partition in
        # stage2.py might do something expensive; someone will have to look into that later. But
        # in the meantime, this fixes ZeRO2 + Pipelining enough to run a demo. Further profiling
        # needed to decide if it actually breaks everything.
        # (see https://github.com/EleutherAI/gpt-neox/issues/62#issuecomment-761471944)
        if self.zero_optimization_partition_gradients():
            self.optimizer.overlapping_partition_gradients_reduce_epilogue()

        weight_group_list = self.module.get_tied_weights_and_groups()
        for weight, group in weight_group_list:
            grad = weight._hp_grad if self.bfloat16_opt_enabled() else weight.grad
            dist.all_reduce(grad, group=group)

    def _exec_reduce_grads(self):
        self._force_grad_boundary = True
        if self.pipeline_enable_backward_allreduce:
            if self.bfloat16_opt_enabled():
                if self.zero_optimization_stage() == 0:
                    self._bf16_reduce_grads()
                else:
                    assert self.zero_optimization_stage() == 1, "only bf16 + z1 are supported"
                    raise NotImplementedError()
            else:
                self.allreduce_gradients(bucket_size=MEMORY_OPT_ALLREDUCE_SIZE)
        self._force_grad_boundary = False

    def _exec_backward_pass(self, buffer_id):
        assert self.optimizer is not None, "must provide optimizer during " \
                                           "init in order to use backward"

        self.mem_status('BEFORE BWD', reset_max=True)

        # The last stage just runs backward on the loss using DeepSpeed's typical
        # mechanisms.
        if self.is_last_stage():
            super().backward(self.loss)
            self.mem_status('AFTER BWD')
            return

        outputs = self.pipe_buffers['outputs'][buffer_id]

        if self.wall_clock_breakdown():
            self.timers('backward_microstep').start()
            self.timers('backward').start()
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner').start()

        # Reconstruct if we previously partitioned the output. We must be
        # careful to also restore the computational graph of the tensors we partitioned.
        if self.is_pipe_partitioned:
            if self.is_grad_partitioned:
                part_output = PartitionedTensor.from_meta(meta=outputs[0],
                                                          local_part=outputs[1],
                                                          group=self.grid.get_slice_parallel_group())
                self.pipe_buffers['output_tensors'][buffer_id].data = part_output.full()
                outputs = (self.pipe_buffers['output_tensors'][buffer_id], *outputs[2:])
            else:
                # Already restored from partition
                self.pipe_buffers['output_tensors'][buffer_id].data = outputs[0]
                outputs = (self.pipe_buffers['output_tensors'][buffer_id], *outputs[1:])

        grad_tensors = self.grad_layer
        if self.is_grad_partitioned:
            #print(f'RANK={self.global_rank} BEFORE-BWD restoring grad={self.grad_layer[0].size()} {self.grad_layer[1].size()}')
            part_grad = PartitionedTensor.from_meta(meta=self.grad_layer[0],
                                                    local_part=self.grad_layer[1],
                                                    group=self.grid.get_slice_parallel_group())
            grad_tensors = (part_grad.full(), *grad_tensors[2:])
            part_grad = None
            #print(f'RANK={self.global_rank} BEFORE-BWD restored grad={self.grad_layer[0].size()} {self.grad_layer[1].size()}')

        if self.bfloat16_opt_enabled() and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            self.optimizer.clear_lp_grads()

        # This handles either a single tensor or tuple of tensors.
        if isinstance(outputs, tuple):
            out_tensors = [t for t in outputs if t.is_floating_point()]
            assert len(out_tensors) == len(grad_tensors)
            torch.autograd.backward(tensors=out_tensors, grad_tensors=grad_tensors)
        else:
            torch.autograd.backward(tensors=(outputs, ), grad_tensors=(grad_tensors, ))

        if self.bfloat16_opt_enabled() and not self.is_last_stage():
            # manually call because we don't call optimizer.backward()
            self.optimizer.update_hp_grads(clear_lp_grads=False)

        # Free up the memory from the output of forward()
        self.pipe_buffers['output_tensors'][buffer_id] = None
        self.pipe_buffers['outputs'][buffer_id] = None
        grad_tensors = None

        if self.wall_clock_breakdown():
            self.timers('backward_inner').stop()
            self.timers('backward_inner_microstep').stop()
            self.timers('backward').stop()
            self.timers('backward_microstep').stop()

        self.mem_status('AFTER BWD')

    # A map of PipeInstruction types to methods. Each method will be executed with the
    # kwargs provided to the PipeInstruction from the scheduler.
    O = _origin_DeepSpeedPipelineEngine
    _INSTRUCTION_MAP = {
        schedule.OptimizerStep: O._exec_optimizer_step,
        schedule.ReduceGrads: _exec_reduce_grads,
        schedule.ReduceTiedGrads: _exec_reduce_tied_grads,
        schedule.LoadMicroBatch: O._exec_load_micro_batch,
        schedule.ForwardPass: O._exec_forward_pass,
        schedule.BackwardPass: _exec_backward_pass,
        schedule.SendActivation: O._exec_send_activations,
        schedule.RecvActivation: O._exec_recv_activations,
        schedule.SendGrad: O._exec_send_grads,
        schedule.RecvGrad: O._exec_recv_grads,
    }
    del O

    @instrument_w_nvtx
    def allreduce_gradients(self, bucket_size=MEMORY_OPT_ALLREDUCE_SIZE):
        assert not (self.bfloat16_opt_enabled() and self.pipeline_parallelism), \
            f'allreduce_gradients() is not valid when bfloat+pipeline_parallelism is enabled'

        model_state.ready_to_all_reduce_grads = False
        # Pass (PP) gas boundary flag to optimizer (required for zero)
        self.optimizer.is_gradient_accumulation_boundary = self.is_gradient_accumulation_boundary()
        # ZeRO stage >= 2 communicates during non gradient accumulation boundaries as well
        if self.zero_optimization_partition_gradients():
            self.optimizer.overlapping_partition_gradients_reduce_epilogue()

        # Communicate only at gradient accumulation boundaries
        elif self.is_gradient_accumulation_boundary():
            if self.zero_optimization_stage() == ZeroStageEnum.optimizer_states and hasattr(
                    self.optimizer, 'reduce_gradients'):
                self.optimizer.reduce_gradients(pipeline_parallel=self.pipeline_parallelism)
            else:
                self.buffered_allreduce_fallback(elements_per_buffer=bucket_size)

    def bfloat16_opt_enabled(self):
        return self.bfloat16_enabled() and not self.zero_optimization()
