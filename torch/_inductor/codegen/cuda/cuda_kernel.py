from ..common import Kernel
from ...utils import sympy_product

class CUDAKernel(Kernel):
    pass


class CUDATemplateKernel(CUDAKernel):
    def __init__(
        self,
        kernel_name,
    ):
        super().__init__()
        self.kernel_name = kernel_name
        self.named_nodes = {}


    def def_kernel(self, *nodes: IRNode, names_str: str = ""):
        """
        Hook called from template code to generate function def and
        needed args.
        """

        names = [x.strip() for x in names_str.strip().split(",")]
        if len(nodes) > len(names):
            raise RuntimeError(f"{len(nodes)=} > {len(names)=}, {nodes=}, {names=}")

        for name, node in zip(names[len(nodes)], nodes):
            self.named_nodes[name] = node
            self.args.input_buffers[node.get_name()] = name

        arg_defs, *_ = self.args.cpp_argdefs()
        arg_defs = arg_defs + names[len(nodes):]
        return f"void {self.kernel_name} ({', '.join(arg_defs)})"


    def size(self, node: IRNode, index: int) -> str:
        """
        Hook called from template code to get the size of an arg.
        Will add needed args to pass it in if it is dynamic.
        """

        if node is None:
            return "0"
        val = node.get_size()[index]
        return texpr(self.rename_indexing(val))


    def stride(self, node: IRNode, index: int) -> str:
        """
        Hook called from template code to get the stride of an arg.
        Will add needed args to pass it in if it is dynamic.
        """

        if node is None:
            return "0"

        val = sympy_product(node.get_stride()[index:])
        return texpr(self.rename_indexing(val))
