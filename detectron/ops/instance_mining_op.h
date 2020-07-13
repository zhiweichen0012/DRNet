#ifndef CAFFE2_OPERATORS_INSTANCE_MINING_OP_H_
#define CAFFE2_OPERATORS_INSTANCE_MINING_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/net.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/conversions.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class InstanceMiningOp final : public Operator<Context> {
 public:
  USE_OPERATOR_CONTEXT_FUNCTIONS;
  InstanceMiningOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws),
        cur_iter_(0),
        acc_num_rois_(0),
        acc_num_rois_mining_(0),
        display_(this->template GetSingleArgument<int32_t>("display", 1280)),
        debug_info_(
            this->template GetSingleArgument<bool>("debug_info", false)) {}
  ~InstanceMiningOp() {}

  bool RunOnDevice() override;

 protected:
  bool debug_info_;

  int display_;
  int cur_iter_;
  int acc_num_rois_;
  int acc_num_rois_mining_;
};

}  // namespace caffe2

#endif  // CAFFE2_OPERATORS_INSTANCE_MINING_OP_H_
