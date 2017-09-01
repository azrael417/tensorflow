/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifdef TENSORFLOW_USE_HDF5

// See docs in ../ops/parsing_ops.cc.
#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"

namespace tensorflow {

class DecodeHDF5Op : public OpKernel {
 public:
  explicit DecodeHDF5Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
    
    std::cout << "INSIDE CONSTRUCTOR" << std::endl;
    
    OP_REQUIRES_OK(ctx, ctx->GetAttr("OUT_TYPE", &out_type_));
    OP_REQUIRES(ctx, out_type_.size() < std::numeric_limits<int>::max(),
                errors::InvalidArgument("Out type too large"));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor* records;
    OpInputList record_defaults;
    
    //get records
    OP_REQUIRES_OK(ctx, ctx->input("records", &records));
    
    //make them flat out:
    auto records_t = records->flat<string>();
    int64 records_size = records_t.size();

    OpOutputList output;
    OP_REQUIRES_OK(ctx, ctx->output_list("output", &output));

    for (int i = 0; i < static_cast<int>(out_type_.size()); ++i) {
      Tensor* out = nullptr;
      OP_REQUIRES_OK(ctx, output.allocate(i, records->shape(), &out));
    }
    
    std::cout << "SIZE " << records_size << std::endl;
    
    for (int64 i = 0; i < records_size; ++i) {
      std::cout << i << std::endl;
    }
  }

 private:
  std::vector<DataType> out_type_;
};

REGISTER_KERNEL_BUILDER(Name("DecodeHDF5").Device(DEVICE_CPU), DecodeHDF5Op);

}  // namespace tensorflow

#endif