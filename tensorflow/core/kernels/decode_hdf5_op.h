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

#ifndef TENSORFLOW_DECODE_HDF5
#define TENSORFLOW_DECODE_HDF5
#endif

// See docs in ../ops/parsing_ops.cc.
#include <vector>
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/numbers.h"

//hdf5-specific stuff
#include "third_party/hdf5/hdf5.h"

//annoyingly we need this file because we need templating for the buffer read routine:
namespace tensorflow {

  class DecodeHDF5Op : public OpKernel {
  public:
    explicit DecodeHDF5Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
        
      OP_REQUIRES_OK(ctx, ctx->GetAttr("OUT_TYPE", &out_type_));
      OP_REQUIRES(ctx, out_type_.size() < std::numeric_limits<int>::max(),
      errors::InvalidArgument("Out type too large"));
    }
    
    void Compute(OpKernelContext* ctx) override;

  private:
    std::vector<DataType> out_type_;
    
    //binary parsing stuff
    template<typename T>
    Status ExtractValuesBinary(Tensor& tensor, const HDF5File::DatasetInfo& record){
      auto tensor_t = tensor.flat<T>();
      if(H5Tequal(record.type, H5T_NATIVE_FLOAT)){
        for (unsigned int i = 0; i < record.dset_size; ++i) {
          tensor_t(i) = static_cast<T>(*(reinterpret_cast<float*>(&record.buff[i*record.type_size])));
        }
      }
      else if(H5Tequal(record.type, H5T_NATIVE_DOUBLE)){
        for (unsigned int i = 0; i < record.dset_size; ++i) {
          tensor_t(i) = static_cast<T>(*(reinterpret_cast<double*>(&record.buff[i*record.type_size])));
        }
      }
      else if(H5Tequal(record.type, H5T_NATIVE_INT)){
        for (unsigned int i = 0; i < record.dset_size; ++i) {
          tensor_t(i) = static_cast<T>(*(reinterpret_cast<int*>(&record.buff[i*record.type_size])));
        }
      }
      else if(H5Tequal(record.type, H5T_NATIVE_LONG)){
        for (unsigned int i = 0; i < record.dset_size; ++i) {
          tensor_t(i) = static_cast<T>(*(reinterpret_cast<long*>(&record.buff[i*record.type_size])));
        }
      }
      return Status::OK();
    }
    std::vector<HDF5File::DatasetInfo> ParseRecord(const string& record);
  };

}  // namespace tensorflow

#endif