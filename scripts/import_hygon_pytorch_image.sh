#!/usr/bin/env bash
set -euo pipefail

tag="${1:-harbor.sourcefind.cn:5443/dcu/admin/base/pytorch:2.7.1-ubuntu22.04-dtk26.04-py3.11}"

import_stream() {
  docker import \
  --change 'ENV PATH=/opt/ucx/bin:/opt/dtk/bin:/opt/dtk/llvm/bin:/opt/dtk/hip/bin:/opt/dtk/hip/bin/hipify:/opt/hyhal/bin:/opt/dtk/opencl/bin:/opt/mpi/bin:/opt/hwloc/bin/:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin' \
  --change 'ENV TZ=Asia/Shanghai' \
  --change 'ENV SHELL=/bin/bash' \
  --change 'ENV LC_ALL=en_US.UTF-8' \
  --change 'ENV LANG=en_US.UTF-8' \
  --change 'ENV LANGUAGE=en_US.UTF-8' \
  --change 'ENV LD_LIBRARY_PATH=/opt/ucx/lib:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/mpi/lib:/opt/hwloc/lib:/usr/local/lib/:/usr/local/lib64/:' \
  --change 'ENV PYTHONPATH=/usr/local/:' \
  --change 'ENV MANPATH=/opt/mpi/share/man:' \
  --change 'ENV HYHAL_PATH=/opt/hyhal' \
  --change 'ENV DTKROOT=/opt/dtk' \
  --change 'ENV AMDGPU_TARGETS=gfx906;gfx926;gfx928;gfx936;gfx938' \
  --change 'ENV ROCM_PATH=/opt/dtk' \
  --change 'ENV HIP_PATH=/opt/dtk/hip' \
  --change 'ENV MIOPEN_FIND_MODE=3' \
  --change 'ENV CMAKE_PREFIX_PATH=/opt/hyhal:' \
  --change 'ENV LIBRARY_PATH=/opt/hyhal/lib:/opt/dtk/opencl/lib:/opt/dtk/dushmem/lib:' \
  --change 'ENV C_INCLUDE_PATH=/opt/dtk/dcc/gcvm/include:/opt/dtk/include:/opt/hyhal/include:/opt/dtk/llvm/include:/opt/dtk/opencl/include:' \
  --change 'ENV CPLUS_INCLUDE_PATH=/opt/dtk/dcc/gcvm/include:/opt/dtk/include:/opt/hyhal/include:/opt/dtk/llvm/include:/opt/dtk/dushmem/include:' \
  --change 'ENV MOFED_VERSION=23.10-4.0.9.1' \
  --change 'ENV OS_VERSION=ubuntu22.04' \
  --change 'ENV PLATFORM=x86_64' \
  --change 'ENV UCX_HOME=/opt/ucx' \
  --change 'ENTRYPOINT ["docker-entrypoint.sh"]' \
  --change 'CMD ["/bin/bash"]' \
  --change 'WORKDIR /workspace' \
  --change 'LABEL dtk-version=dtk-26.04 maintainer=chenwch1@sugon.com org.opencontainers.image.ref.name=ubuntu org.opencontainers.image.version=22.04' \
    - "$tag"
}

if [[ "${DECOMPRESS_INPUT:-0}" == 1 ]]; then
  zstd -d -c | import_stream
else
  import_stream
fi
