import functools
from copy import copy

from . import cutlass_utils
from ..arch import get_cuda_arch
from ..common import jinja2_env
from ...ir import IRNode
from ...utils import IndentedBuffer

from third_party.cutlass.python.library import script as cutlass_lib
from torch.backends.cuda import matmul_settings

class CUDATemplate:
    index_counter = itertools.count()
    all_templates = dict()

    @staticmethod
    def _template_from_string(source):
        env = jinja2_env()
        if env is not None:
            return env.from_string(source)
        return None

    def __init__(self, name: str, source: str, debug=False):
        super().__init__()
        self.name = name
        self.template = self._template_from_string(source)
        assert name not in self.all_templates, "duplicate template name"
        self.all_templates[name] = self
        self.debug = debug

    def header(self) -> IndentedBuffer:
        return IndentedBuffer().splice(
            """
                #include <iostream>
                #include <memory>
                #include <random>
                #include <vector>
            """
        )

    def globals(self) -> IndentedBuffer:
        return IndentedBuffer().splice(
            """
                using bfloat16 = nv_bfloat16;
            """
        )

    def maybe_append_choice(
        self,
        choices,
        input_nodes,
        layout,
        prefix_args=0,
        suffix_args=0,
        epilogue_fn=identity,
        **kwargs,
    ):
        try:
            choices.append(
                self.generate(
                    input_nodes=input_nodes,
                    layout=layout,
                    prefix_args=prefix_args,
                    suffix_args=suffix_args,
                    epilogue_fn=epilogue_fn,
                    **kwargs,
                )
            )
        except NotImplementedError:
            pass

    def generate(
        self,
        input_nodes,
        layout,
        prefix_args=0,
        suffix_args=0,
        epilogue_fn=identity,
        **kwargs,
    ):
        assert self.template, "requires jinja2"
        defines = StringIO()
        for name, val in kwargs.items():
            defines.write(f"    {name} : tl.constexpr = {val}\n")
        defines = defines.getvalue()

        fake_out = ir.Buffer("buf_out", layout)
        kernel_name = f"triton_{self.name}"

        numel = sympy_product(layout.size)
        buffers = itertools.chain(input_nodes, (fake_out,))
        if not TritonScheduling.can_use_32bit_indexing(numel, buffers):
            raise NotImplementedError(
                "64-bit indexing is not yet implemented for triton templates"
            )

        kernel_options = dict(
            input_nodes=input_nodes,
            defines=defines,
            meta=kwargs,
            call_sizes=layout.size,
            prefix_args=prefix_args,
            suffix_args=suffix_args,
            epilogue_fn=epilogue_fn,
            index_dtype="tl.int32",
        )
        with patch.object(
            V.graph, "get_dtype", self.fake_get_dtype(fake_out)
        ), CUDATemplateKernel(
            kernel_name=kernel_name,
            output_node=fake_out,
            use_jit=True,
            **kernel_options,
        ) as kernel:
            # need to do call render twice to get all the needed args right
            try:
                self.template.render(
                    **kernel.template_env(),
                    **kwargs,
                )
                code = self.template.render(
                    **kernel.template_env(),
                    **kwargs,
                )
            except ZeroDivisionError:
                # TODO(nmacchioni): fix sympy division by zero
                return None
            if self.debug:
                print("Generated Code:\n", code)
            extra = (
                "-".join(
                    [
                        *[
                            f"{kwarg}={repr(kwargs[kwarg])}"
                            for kwarg in sorted(kwargs.keys())
                        ],
                        f"num_stages={num_stages}",
                        f"num_warps={num_warps}",
                    ]
                )
                + "-"
            )
            mod = PyCodeCache.load(code, extra)
            _, call_args, _ = kernel.args.python_argdefs()

        expected_args = [x.get_name() for x in input_nodes] + [fake_out.get_name()]
        # TODO(nmacchioni) fix bug here in CI tests
        # assert list(call_args) == expected_args, (call_args, expected_args)
        if list(call_args) != expected_args:
            return None
        extra_args = V.graph.sizevars.size_hints(
            map(sympy.expand, call_args[len(expected_args) :])
        )
        assert not extra_args, "TODO: dynamic shapes"

        kernel_hash_name = f"triton_{self.name}_{next(self.index_counter)}"

        def make_kernel_render(out_node):
            kernel = CUDATemplateKernel(
                kernel_name="KERNEL_NAME",
                output_node=out_node,
                use_jit=False,
                **kernel_options,
            )
            render = functools.partial(
                self.template.render,
                **kernel.template_env(),
                **kwargs,
            )
            return kernel, render

        # create the BenchmarkRequest
        grid = self.grid(*V.graph.sizevars.size_hints(layout.size), kwargs)
        bmreq = BenchmarkRequest(
            module_path=mod.__file__,
            module_cache_key=mod.key,
            kernel_name=kernel_name,
            grid=grid,
            extra_args=extra_args,
            num_stages=num_stages,
            num_warps=num_warps,
            input_tensors=TensorMeta.from_irnodes(input_nodes),
            output_tensor=TensorMeta.from_irnodes(layout),
        )

        return CUDATemplateCaller(
            kernel_hash_name,
            input_nodes,
            layout,
            make_kernel_render,
            extra.strip("-").replace("-", ", "),
            bmreq,
        )

    @staticmethod
    def fake_get_dtype(fake_out):
        _get_dtype_real = V.graph.get_dtype

        def get_dtype(name):
            if name == fake_out.get_name():
                return fake_out.get_dtype()
            return _get_dtype_real(name)

        return get_dtype


class CutlassTemplate(CUDATemplate):
    def header(self) -> IndentedBuffer:
        return super().header().splice(
            """
                #include "cutlass/cutlass.h"
                #include "cutlass/epilogue/collective/collective_builder.hpp"
                #include "cutlass/numeric_types.h"
                #include "cutlass/util/host_tensor.h"
                #include "cutlass/util/reference/host/tensor_fill.h"
                #include "cutlass/util/reference/device/tensor_fill.h"
                #include "cutlass/util/device_memory.h"
            """
        )

    def globals(self) -> IndentedBuffer:
        return super().globals().splice(
            """
                #define CUTLASS_CHECK(status)                                                         \
                {                                                                                   \
                  cutlass::Status error = status;                                                   \
                  if (error != cutlass::Status::kSuccess) {                                         \
                    auto msg = std::string("[") + __FILE__ + "] Got cutlass error: " +              \
                        cutlassGetStatusString(error) + " at: " + std::to_string(__LINE__);         \
                    std::cerr << msg << std::endl;                                                  \
                    throw std::runtime_error(msg);                                                  \
                  }                                                                                 \
                }
            """
        )


class CutlassGemmTemplate(CutlassTemplate):
    def header(self) -> IndentedBuffer:
        return super().header().splice(
            """
                #include "cutlass/gemm/gemm.h"
                #include "cutlass/gemm/collective/collective_builder.hpp"
                #include "cutlass/gemm/device/gemm_universal.h"
                #include "cutlass/gemm/device/gemm_universal_adapter.h"
                #include "cutlass/gemm/kernel/gemm_universal.hpp"
            """
        )


    @classmethod
    def cutlass_layout(cls, torch_layout) -> Optional[cutlass_lib.library.LayoutType]:
        if torch_layout.stride[-1] == 1:
            return cutlass_lib.library.LayoutType.RowMajor
        elif torch_layout.stride[-2] == 1:
            return cutlass_lib.library.LayoutType.ColumnMajor
        else:
            return None


    @classmethod
    def layout_match(cls, torch_layout, cutlass_layout) -> bool:
        return self.cutlass_layout(torch_layout) == cutlass_layout


    @classmethod
    def has_tma_epilogue(cls, op) -> bool:
        result = False
        if op.gemm_kind == cutlass_lib.library.GemmKind.Universal3x:
            epilogue_schedule_str = str(op.epilogue_schedule).split(".")[-1]
            result = epilogue_schedule_str.lower().startswith("tma")
        return result


    @classmethod
    def filter_op(
        cls,
        op: cutlass_layout.gemm_operation.GemmOperation,
        input_nodes: List[IRNode],
        layout: Layout
    ) -> cutlass_lib.gemm_operation.GemmOperation:
        ret = []

        # skip simt kernels
        if (
            op.tile_description.math_instruction.opcode_class
            == cutlass_lib.library.OpcodeClass.Simt
        ):
            return ret

        # Filter ops by dtypes.
        X = input_nodes[0]
        W = input_nodes[1]
        if not (
            cutlass_utils.dtype_match(X.get_dtype(), op.A.element) and
            cutlass_utils.dtype_match(W.get_dtype(), op.B.element) and
            cutlass_utils.dtype_match(layout.dtype, op.C.element)
        ):
            return ret

        # Filter ops by accumulation type.
        if matmul_settings.allow_fp16_reduced_precision_reduction:
            if not cutlass_utils.dtype_match(torch.float16, op.accumulator_type):
                return ret

        # Filter ops by input layouts.
        if not (
            cls.layout_match(X.get_layout(), op.A.layout) and
            cls.layout_match(W.get_layout(), op.B.layout)
        ):
            return ret

        # Update op.
        op = copy.deepcopy(op)

        # Set output layout and alignment.
        def set_output_alignment(torch_layout, op_element, has_tma_epilogue, name) -> bool:
            cutlass_layout = cls.cutlass_layout(torch_layout)
            if cutlass_layout is None:
                raise RuntimeError(f"Unsupported {name} layout in cutlass: {torch_layout}!")
            op_element.layout = outout_layput
            op_element.alignment = cutlass_utils.get_alignment(torch_layout)
            if has_tma_epilogue:
                max_alignment = cutlass_utils.get_alignment(layout.dtype)
                if op_element.alignment != max_alignment:
                    return False
            return True

        has_tma_epilogue = cls.has_tma_epilogue(op)
        if set_output_alignment(layout, op.D, has_tma_epilogue, "output"):
            return ret

        # Set epilogue.
        # TODO: update epilogue functor according to epilogues.
        op.element_epilogue = op.accumulator_type

        # Set bias layout and alignment.
        if len(input_nodes) == 3:
            Bias = input_nodes[2]
            if set_output_alignment(Bias.get_layout(), op.C, has_tma_epilogue, "bias"):
                return ret

        return ret


    def gen_ops(
        cls,
        input_nodes: List[IRNode],
        layout: Layout
    ) -> List[cutlass_lib.gemm_operation.GemmOperation]:
        ops = cutlass_utils.gen_ops()[cutlass_lib.library.OperationKind.Gemm].values()
        ops = [op for op in ops if cls.filter_op(op, input_nodes, layout)]
        return ops


    def should_swap_XW(
        cls,
        op: cutlass_lib.gemm_operation.GemmOperation,
        has_bias: bool,
    ) -> bool:
        has_tma_epilogue = cls.has_tma_epilogue(op)
        is_output_row_major = (op.D.layout == cutlass_lib.library.LayoutType.RowMajor)

        return has_tma_epilogue and has_bias and is_output_row_major


    def flip_cutlass_layout(
        cls,
        cutlass_layout: cutlass_lib.library.LayoutType,
    ) -> cutlass_lib.library.LayoutType:
        if cutlass_layout == cutlass_lib.library.LayoutType.RowMajor:
            return cutlass_lib.library.LayoutType.ColumnMajor
        else:
            return cutlass_lib.library.LayoutType.RowMajor


    def define_gemm_instance(
        cls,
        op: cutlass_lib.gemm_operation.GemmOperation,
        bias: IRNode,
    ) -> str:
        if op.gemm_kind == cutlass_lib.library.GemmKind.Universal3x:
            emitter = cutlass_lib.gemm_operation.EmitGemmUniversal3xInstance()
            has_bias = bias is not None
            if cls.should_swap_XW(op, has_bias):
                op.A.layout = cls.flip_cutlass_layout(op.A.layout)
                op.B.layout = cls.flip_cutlass_layout(op.B.layout)
                op.A, op.B = op.B, op.A
                op.C.layout = cls.flip_cutlass_layout(op.C.layout)
                op.D.layout = cls.flip_cutlass_layout(op.D.layout)
            op_def = emitter.emit(op)
        else:
            emitter = cutlass_lib.gemm_operation.EmitGemmInstance()
            op_def = emitter.emit(op)
            op_def = op_def.replace(
                "cutlass::gemm::device::Gemm", "cutlass::gemm::device::GemmUniversal"
            )
            op_def= op_def.replace("false,", "")
        return op_def


# Only supports alpha * A@B + beta * C now.
# TODO: Support arbitrary epilogue after epilogue visitor is released in cutlass 3.2.
gemm_template = CutlassGemmTempalte(
    name = "gemm",
    source = r"""
{{self.header().get_value()}}

{{self.globals().get_value()}}

{{self.define_gemm_instance(op, Bias)}}

template<typename GemmInstance>
{{define_kernel(X, W, Y, Bias, names="X, W, Y, Bias, GemmInstance gemm_op, uint8_t* workspace, cudaStream_t stream")}} {
  int64_t B = {{batch_size(X)}}
  int64_t M = {{size(X, -2)}}
  int64_t K = {{size(X, -1)}}
  int64_t N = {{size(W, -1)}}

  int64_t batch_stride_x = {{stride(X, -3)}};
  int64_t stride_x0 = {{stride(X, -2)}};
  int64_t stride_x1 = {{stride(X, -1)}};
  int64_t row_stride_x = {{max(stride(X, -2), stride(X, -1))}};
  int64_t offset_x = {{offset(X)}};

  int64_t batch_stride_w = {{stride(W, -3)}};
  int64_t stride_w0 = {{stride(W, -2)}};
  int64_t stride_w1 = {{stride(W, -1)}};
  int64_t row_stride_w = {{max(stride(W, -2), stride(W, -1))}};
  int64_t offset_w = {{offset(W)}}

  int64_t batch_stride_bias = {{stride(Bias, -3)}};
  int64_t stride_bias0 = {{stride(Bias, -2)}};
  int64_t stride_bias1 = {{stride(Bias, -1)}};
  int64_t row_stride_bias = {{max(stride(Bias, -2), stride(Bias, -1))}};
  int64_t offset_bias = {{offset(Bias)}};

  int64_t batch_stride_y = {{stride(Y, -3)}};
  int64_t stride_y0 = {{stride(Y, -2)}};
  int64_t stride_y1 = {{stride(Y, -1)}};
  int64_t row_stride_y = {{max(stride(Y, -2), stride(Y, -1))}};
  int64_t offset_y = {{offset(Y)}}

  {{check_not_null(X)}}
  {{check_not_null(W)}}
  {{check_not_null(Bias)}}
  {{check_not_null(Y)}}

  using ElementComputeEpilogue = GemmInstance::ElementAccumulator;
  using coord_t = cutlass::gemm::GemmCoord::Index;
  GemmInstance::Arguments arguments;

{% if op.prefix == "3x" %}
  // Initialize GemmUniversal3xInstance arguments.
  arguments = {
    {{self.gemm_mode(Y)}},  // GemmUniversalMode mode
{% if self.has_tma_epilogue(op) %}
  {% if rank("bias_ptr") == 1 %}
    {
      static_cast<coord_t>(N),
      static_cast<coord_t>(M),
      static_cast<coord_t>(K),
      static_cast<coord_t>(B)
    }, // ProblemShape problem_shape
    {
      ({{type("w_ptr")}}*)(w_ptr) + offset_w,  // ElementA const* ptr_A
      {stride_w0, stride_w1, batch_stride_w},  // StrideA dA
      ({{type("x_ptr")}}*)(x_ptr) + offset_x,  // ElementB const* ptr_B
      {stride_x0, stride_x1, batch_stride_x},  // StrideB dB
      },  // StrideB dB
    },  // MainloopArguments mainloop
    {
      {ElementComputeEpilogue({{alpha}}), ElementComputeEpilogue({{beta}})},  // typename ThreadEpilogueOp::Params thread
      nullptr,  // ElementC const* ptr_C
      {cute::Int<1>{}, cute::Int<0>{}, cute::Int<0>{}},  // StrideC dC
      ({{type("y_ptr")}}*)(y_ptr) + offset_y,  // ElementD const* ptr_D
      {stride_y1, stride_y0, batch_stride_y},  // StrideD dD
      ({{type("bias_ptr")}}*)(bias_ptr + offset_bias),  // ElementBias const* ptr_Bias
    },  // EpilogueArguments epilogue
  {% else %}
    {
      static_cast<coord_t>(M),
      static_cast<coord_t>(N),
      static_cast<coord_t>(K),
      static_cast<coord_t>(B)
    }, // ProblemShape problem_shape
    {
      ({{type("x_ptr")}}*)(x_ptr) + offset_x,  // ElementB const* ptr_A
      {stride_x0, stride_x1, batch_stride_x},  // StrideB dA
      ({{type("w_ptr")}}*)(w_ptr) + offset_w,  // ElementA const* ptr_B
      {stride_w0, stride_w1, batch_stride_w},  // StrideA dB
      },  // StrideB dB
    },  // MainloopArguments mainloop
    {
      {ElementComputeEpilogue({{alpha}}), ElementComputeEpilogue({{beta}})},  // typename ThreadEpilogueOp::Params thread
      ({{type("bias_ptr")}}*)(bias_ptr + offset_bias),  // ElementC const* ptr_C
      {stride_bias0, stride_bias1, batch_stride_bias},  // StrideC dC
      ({{type("y_ptr")}}*)(y_ptr) + offset_y,  // ElementD const* ptr_D
      {stride_y0, stride_y1, batch_stride_y},  // StrideD dD
      nullptr,  // ElementBias const* ptr_Bias
    },  // EpilogueArguments epilogue
  {% endif %}
{% else %}
    {
        static_cast<coord_t>(M),
        static_cast<coord_t>(N),
        static_cast<coord_t>(K),
        static_cast<coord_t>(B)
    },  // ProblemShape problem_shape
    {
      ({{type("x_ptr")}}*)(x_ptr) + offset_x,  // ElementA const* ptr_A
      {stride_x0, stride_x1, batch_stride_x},  // StrideA dA
      ({{type("w_ptr")}}*)(w_ptr) + offset_w,  // ElementB const* ptr_B
      {stride_w0, stride_w1, batch_stride_w},  // StrideB dB
    },  // MainloopArguments mainloop
    {
      {ElementComputeEpilogue({{alpha}}), ElementComputeEpilogue({{beta}})},  // typename ThreadEpilogueOp::Params thread
      ({{type("bias_ptr")}}*)(bias_ptr + offset_bias),  // ElementBias const* ptr_Bias
      {stride_bias0, stride_bias1, batch_stride_bias},  // StrideC dC
      ({{type("y_ptr")}}*)(y_ptr) + offset_y,  // ElementD const* ptr_D
      {stride_y0, stride_y1, batch_stride_y},  // StrideD dD
    },  // EpilogueArguments epilogue
{% endif %}
  };

{% else %}
  // Initialize GemmUniversalInstance arguments.
  arguments = {
    {{self.gemm_mode(has_batch)}},  // GemmUniversalMode mode
    {
      static_cast<coord_t>(M),
      static_cast<coord_t>(N),
      static_cast<coord_t>(K)
    },  // GemmCoord problem_size
    {{"B" if has_batch else split_k}},  // int batch_count
    {ElementComputeEpilogue({{alpha}}), ElementComputeEpilogue({{beta}})},  // typename EpilogueOutputOp::Params epilogue
    x_ptr + offset_x,  // void const * ptr_A
    w_ptr + offset_w,  // void const * ptr_B
    bias_ptr + offset_bias,  // void const * ptr_C
    y_ptr + offset_y,  // void * ptr_D
    batch_stride_x,  // int64_t batch_stride_A
    batch_stride_w,  // int64_t batch_stride_B
    batch_stride_bias,  // int64_t batch_stride_C
    batch_stride_y,  // int64_t batch_stride_D
    row_stride_x,  // typename LayoutA::Stride::LongIndex lda
    row_stride_w,  // typename LayoutB::Stride::LongIndex ldb
    row_stride_bias,  // typename LayoutC::Stride::LongIndex ldc
    row_stride_y,  // typename LayoutC::Stride::LongIndex ldd
  };
{% endif %}
}
""",
)
