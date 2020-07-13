#include <functional>

#include "caffe2/core/context_gpu.h"
#include "caffe2/operators/operator_fallback_gpu.h"
#include "instance_mining_op.h"

namespace caffe2 {

namespace {}  // namespace

REGISTER_CUDA_OPERATOR(InstanceMining, GPUFallbackOp);

}  // namespace caffe2
