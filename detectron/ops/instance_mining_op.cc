#include <functional>

#include "instance_mining_op.h"

namespace caffe2 {

template <typename T>
vector<size_t> sort_indexes(const vector<T>& v) {
  // initialize original index locations
  vector<size_t> idx(v.size());
  iota(idx.begin(), idx.end(), 0);

  // sort indexes based on comparing values in v
  sort(idx.begin(), idx.end(),
       [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });

  return idx;
}

std::set<int> mining_all_class(const float* Sdata, const float* Ydata,
                               const float* Tdata, const int num_rois,
                               const int num_classes) {
  std::set<int> rois_idx;

  std::vector<float> SSdata;
  SSdata.clear();

  for (int n = 0; n < num_rois; n++) {
    for (int c = 0; c < num_classes; c++) {
      SSdata.push_back(Sdata[n * num_classes + c]);
    }
  }
  vector<size_t> sort_idx = sort_indexes(SSdata);

  int fg_num = 50;
  int bg_num = 50;
  int fg_num_cur = 0;
  int bg_num_cur = 0;

  for (int i = 0; i < num_rois * num_classes; i++) {
    int c = sort_idx[i] % num_classes;
    int n = sort_idx[i] / num_classes;

    if (Tdata[c] == 0 && bg_num_cur < bg_num) {
      rois_idx.insert(n);
      bg_num_cur++;
    }
    if (Tdata[c] > 0 && fg_num_cur < fg_num) {
      rois_idx.insert(n);
      fg_num_cur++;
    }
    // if (Tdata[8]>0 || Tdata[18]>0){
    //   rois_idx.insert(n);
    // }
  }

  if (false) {
    for (int i = 0; i < num_rois * num_classes; i++) {
      printf("%f ", SSdata[sort_idx[i]]);
    }
    printf("\n");
  }
  if (false) {
    printf("fg_num: %d/%d bg_num: %d/%d \n", fg_num_cur, fg_num, bg_num_cur, bg_num);
  }

  return rois_idx;
}

std::set<int> mining_by_class(const float* Sdata, const float* Ydata,
                              const float* Tdata, const int num_rois,
                              const int num_classes) {
  std::set<int> rois_idx;

  Tensor SS = Tensor(caffe2::CPU);
  SS.Resize(num_rois);
  float* SSdata = SS.mutable_data<float>();
  // std::vector<float> SSdata;

  for (int c = 0; c < num_classes; c++) {
    // SSdata.clear();
    for (int n = 0; n < num_rois; n++) {
      SSdata[n] = Sdata[n * num_classes + c];
      // SSdata.push_back(Sdata[n * num_classes + c]);
    }
    std::sort(SSdata, SSdata + num_rois, std::greater<float>());
    // std::sort(SSdata.begin(), SSdata.end(), std::greater<float>());

    int p;
    if (Tdata[c] == 0) {
      // p = 20;
      p = 5;
    } else {
      // p = Ydata[c] * num_rois / 2;
      if (num_rois >100){
        p = Ydata[c] * num_rois / 10;
      }else if (num_rois >80){
        p = Ydata[c] * num_rois / 8;
      }else if (num_rois >60){
        p = Ydata[c] * num_rois / 6;
      }else if (num_rois >40){
        p = Ydata[c] * num_rois / 4;
      }else if (num_rois >10){
        p = Ydata[c] * num_rois / 2;
      }else{
        p = Ydata[c] * num_rois;
      }
    }
    float thres = SSdata[p];

    if (false) {
      printf("p: %d thres: %f 0: %f, 1: %f\n", p, thres, SSdata[0],
             SSdata[num_rois - 1]);
      for (int n = 0; n < num_rois; n++) {
        printf("%f ", SSdata[n]);
      }
      printf("\n");
    }

    for (int n = 0; n < num_rois; n++) {
      if (Sdata[n * num_classes + c] >= thres) {
        rois_idx.insert(n);
      }
    }
  }

  return rois_idx;
}

template <>
bool InstanceMiningOp<float, CPUContext>::RunOnDevice() {
  const auto& S = Input(0);
  const auto& Y = Input(1);
  const auto& T = Input(2);
  const auto& R = Input(3);
  const auto& O = Input(4);

  CAFFE_ENFORCE_EQ(S.ndim(), 2);
  CAFFE_ENFORCE_EQ(Y.ndim(), 2);
  CAFFE_ENFORCE_EQ(T.ndim(), 2);
  CAFFE_ENFORCE_EQ(R.ndim(), 2);
  CAFFE_ENFORCE_EQ(O.ndim(), 2);
  CAFFE_ENFORCE_EQ(S.dim32(1), Y.dim32(1));
  CAFFE_ENFORCE_EQ(S.dim32(1), T.dim32(1));
  CAFFE_ENFORCE_EQ(Y.dim32(0), T.dim32(0));
  CAFFE_ENFORCE_EQ(S.dim32(0), R.dim32(0));
  CAFFE_ENFORCE_EQ(S.dim32(0), O.dim32(0));

  const int num_rois = S.dim32(0);
  const int num_classes = S.dim32(1);
  const int channels_R = R.dim32(1);
  const int channels_O = O.dim32(1);

  const float* Sdata = S.data<float>();
  const float* Ydata = Y.data<float>();
  const float* Tdata = T.data<float>();
  const float* Rdata = R.data<float>();
  const float* Odata = O.data<float>();

  std::set<int> rois_idx =
  mining_by_class(Sdata, Ydata, Tdata, num_rois, num_classes);
  // std::set<int> rois_idx =
  //     mining_all_class(Sdata, Ydata, Tdata, num_rois, num_classes);

  auto* RM = Output(0);
  RM->Resize(rois_idx.size(), channels_R);
  float* RMdata = RM->mutable_data<float>();

  auto* OM = Output(1);
  OM->Resize(rois_idx.size(), channels_O);
  float* OMdata = OM->mutable_data<float>();

  std::set<int>::iterator it;
  int is;
  for (it = rois_idx.begin(), is = 0; it != rois_idx.end(); it++, is++) {
    int n = *it;
    for (int c = 0; c < channels_R; c++) {
      RMdata[is * channels_R + c] = Rdata[n * channels_R + c];
    }
    for (int c = 0; c < channels_O; c++) {
      OMdata[is * channels_O + c] = Odata[n * channels_O + c];
    }
  }

  cur_iter_++;
  acc_num_rois_ += num_rois;
  acc_num_rois_mining_ += rois_idx.size();
  if (cur_iter_ % display_ == 0) {
    printf("InstanceMining acc_num_rois: %d \n", acc_num_rois_ / display_);
    printf("InstanceMining acc_num_rois_mining: %d \n",
           acc_num_rois_mining_ / display_);
    acc_num_rois_ = 0;
    acc_num_rois_mining_ = 0;
  }

  return true;
}

REGISTER_CPU_OPERATOR(InstanceMining, InstanceMiningOp<float, CPUContext>);

namespace {}  // namespace

using namespace std::placeholders;

OPERATOR_SCHEMA(InstanceMining)
    .NumInputs(5)
    .NumOutputs(2)
    .SetDoc(R"DOC(
)DOC")
    .Arg("debug_info", "(bool) default to false")
    .Input(0, "scores", "input tensor of size (n*c)")
    .Input(1, "prediction", "input tensor of size (1*c)")
    .Input(2, "label", "input tensor of size (1*c)")
    .Input(3, "rois", "input tensor of size (n*5)")
    .Input(4, "obn_scores", "input tensor of size (n*1)")
    .Output(0, "mining_rois", "output tensor of size (n*5)")
    .Output(1, "mining_obn_scores", "output tensor of size (n*1)");

namespace {
NO_GRADIENT(InstanceMining);

}  // namespace

}  // namespace caffe2
