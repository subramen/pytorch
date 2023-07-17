from .cuda_env import get_cuda_arch, get_cuda_version
from third_party.cutlass.python.library import script as cutlass_lib

class Args:
    def __init__(self, cuda_arch, cuda_version):
        self.operations = "all"
        self.build_dir = ""
        self.curr_build_dir = ""
        self.generator_target = ""
        self.architectures = cuda_arch
        self.kernels = "all"
        self.ignore_kernels = ""
        self.cuda_version = cuda_version
        self.kernel_filter_file = None
        self.selected_kernel_list = None
        self.interface_dir = None
        self.filter_by_cc = True
        self.disable_full_archs_compilation = False


@functools.cache
def gen_ops() -> List[Any]:
    arch = get_cuda_arch()
    version = get_cuda_version()
    args = Args(arch, version)
    manifest = cutlass_lib.manifest.Manifest(args)

    if arch == "90":
        cutlass_lib.generator.GenerateSM90(manifest, args.cuda_version)
        cutlass_lib.generator.GenerateSM80(manifest, args.cuda_version)
    else:
        try:
            func = getattr(cutlass_lib.generator, "GenerateSM" + arch)
            func(manifest, args.cuda_version)
        except AttributeError as e:
            raise NotImplementedError(
                "Arch " + arch + " is not supported by current cutlass lib."
            ) from e

    return manifest.operations


def dtype_match(torch_dtype, cutlass_dtype) -> bool:
    if torch_dtype == torch.float or torch_dtype == torch.float32:
        return cutlass_dtype == cutlass_lib.library.DataType.f32 or cutlass_dtype == cutlass_lib.library.DataType.tf32
    elif torch_dtype == torch.float16 or torch_dtype == torch.half:
        return cutlass_dtype == cutlass_lib.library.DataType.f16
    elif torch_dtype == torch.bfloat16:
        return cutlass_dtype == cutlass_lib.library.DataType.bf16
    else:
        return False


def get_alignments(torch_dtype) -> List[int]:
    if dtype in (torch.float16, torch.half, torch.bfloat16):
        return [8, 4, 2, 1]
    elif dtype in (torch.float, torch.float32):
        return [4, 2, 1]
    else:
        raise NotImplementedError(f"unsupported {dtype=} for alignments")


def get_alignment(torch_layout) -> int:
    dtype = torch_layout.dtype
    size = torch_layout.size
    offset = torch_layout.offset

    def is_static_int(number):
        return isinstance(number, int) or isinstance(number, sympy.Integer)

    if is_static_int(size[-1]) and is_static_int(offset):
        alignments = CutlassUtils.get_alignments(dtype)
        for alignment in alignments:
            if int(size[-1]) % alignment == 0 and int(offset) % alignment == 0:
                return alignment

    return 1
