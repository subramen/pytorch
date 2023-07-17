import func_tools

from .. import config
from subprocess import PIPE, Popen

class Env:
    @staticmethod
    def _detect_cuda_arch_with_nvidia_smi() -> str:
        try:
            proc = Popen(
                ["nvidia-smi", "--query-gpu=gpu_name", "--format=csv"],
                stdout=PIPE,
                stderr=PIPE,
            )
            stdout, stderr = proc.communicate()
            stdout = stdout.decode("utf-8")
            sm_names = {
                "70": ["V100"],
                "75": ["T4", "Quadro T2000"],
                "80": ["PG509", "A100", "A10G", "RTX 30", "A30", "RTX 40"],
                "90": ["H100"],
            }
            for sm, names in sm_names.items():
                if any(name in stdout for name in names):
                    return sm
            return None
        except Exception:
            return None


    @staticmethod
    def _detect_cuda_version_with_nvidia_smi() -> str:
        try:
            proc = Popen(
                ["nvidia-smi -q | grep -i 'CUDA Version'"],
                stdout=PIPE,
                stderr=PIPE,
                shell=True,
            )
            stdout, stderr = proc.communicate()
            stdout = stdout.decode("utf-8")
            version_str = stdout.split(":")[1].strip()
            return version_str
        except Exception:
            return None


    @staticmethod
    def _assert_cuda(res):
        if res[0].value != 0:
            raise RuntimeError(f"CUDA error code={res[0].value}")
        return res[1:]


    @staticmethod
    def _detect_cuda_arch() -> str:
        try:
            from cuda import cuda

            _assert_cuda(cuda.cuInit(0))
            # Get Compute Capability of the first Visible device
            major, minor = _assert_cuda(cuda.cuDeviceComputeCapability(0))
            comp_cap = major * 10 + minor
            if comp_cap >= 90:
                return "90"
            elif comp_cap >= 80:
                return "80"
            elif comp_cap >= 75:
                return "75"
            elif comp_cap >= 70:
                return "70"
            else:
                return None
        except ImportError:
            # go back to old way to detect the CUDA arch
            return _detect_cuda_arch_with_nvidia_smi()
        except Exception:
            return None


    @staticmethod
    def _detect_cuda_version() -> str:
        try:
            from cuda import cuda
            _assert_cuda(cuda.cuInit(0))
            # Get Compute Capability of the first Visible device
            cuda_version = _assert_cuda(cuda.cuDriverGetVersion())
            major_version = cuda_version / 1000
            minor_version = (cuda_version - 1000 * major_version) / 10
            return f"{major_version}.{minor_version}"
        except ImportError:
            # go back to old way to detect the CUDA arch
            return _detect_cuda_version_with_nvidia_smi()
        except Exception:
            return None


    @staticmethod
    @property
    @functools.cache
    def get_cuda_arch() -> str:
        cuda_arch = config.cuda.arch
        if cuda_arch is None:
            cuda_arch = _detect_cuda_arch()
        return cuda_arch


    @staticmethod
    @property
    @functools.cache
    def get_cuda_version() -> str:
        cuda_version = config.cuda.version
        if cuda_version is None:
            cuda_version = _detect_cuda_version()
        return cuda_version
