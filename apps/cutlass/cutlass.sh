set -e

WORKSPACE_DIR=/root/Projects
CUTLASS_DIR=${WORKSPACE_DIR}/cutlass
TVM_DIR=${WORKSPACE_DIR}/tvm-dev
DMLC_DIR=${WORKSPACE_DIR}/tvm-dev/3rdparty/dmlc-core/
DLPACK_DIR=${WORKSPACE_DIR}/tvm-dev/3rdparty/dlpack/
SRC_FILE=./apps/cutlass/cutlass.cu
OBJ_FILE=/tmp/main.o
PY_FILE=./apps/cutlass/cutlass.py

/usr/local/cuda/bin/nvcc                              \
  -DCUTLASS_ENABLE_CUBLAS=1                           \
  -DCUTLASS_NAMESPACE=cutlass                         \
  -I ${CUTLASS_DIR}/include                           \
  -I ${TVM_DIR}/include                               \
  -I ${DMLC_DIR}/include                              \
  -I ${DLPACK_DIR}/include                            \
  -O3 -DNDEBUG -Xcompiler=-fPIC                       \
  -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1                  \
  -Xcompiler=-fno-strict-aliasing                     \
  -gencode=arch=compute_86,code=\[sm_86,compute_86\]  \
  -std=c++17 -x cu                                    \
  -c ${SRC_FILE} -o ${OBJ_FILE}

nm -gC ${OBJ_FILE} | grep -v "std::"
python3 ${PY_FILE}
ldd ${OBJ_FILE}.so
