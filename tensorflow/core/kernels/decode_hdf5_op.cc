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

//hdf5-specific stuff
#include "third_party/hdf5/hdf5.h"


namespace tensorflow {

  class DecodeHDF5Op : public OpKernel {
  public:
    explicit DecodeHDF5Op(OpKernelConstruction* ctx) : OpKernel(ctx) {
        
      OP_REQUIRES_OK(ctx, ctx->GetAttr("OUT_TYPE", &out_type_));
      OP_REQUIRES(ctx, out_type_.size() < std::numeric_limits<int>::max(),
      errors::InvalidArgument("Out type too large"));
    }

    void Compute(OpKernelContext* ctx) override {
      ComputeASCII(ctx);
    }

  private:
    std::vector<DataType> out_type_;
    
    void ComputeBinary(OpKernelContext* ctx) {
      const Tensor* records;
      OpInputList record_defaults;
    
      //get records
      OP_REQUIRES_OK(ctx, ctx->input("records", &records));
    
      //make them flat out:
      auto records_t = records->flat<string>();
      int64 records_size = records_t.size();

      std::vector<HDF5File::DatasetInfo> recs;
      for (int64 i = 0; i < records_size; ++i) {
        std::vector<HDF5File::DatasetInfo> tmprecs = SplitRecord(records_t(i));
        recs.insert(recs.end(), tmprecs.begin(), tmprecs.end());
      }
      OP_REQUIRES(ctx, recs.size()==out_type_.size(), errors::InvalidArgument("Error, number of tensors in recordstring does not match the record format specified."));
    
      OpOutputList output;
      OP_REQUIRES_OK(ctx, ctx->output_list("output", &output));
    
      for (int i = 0; i < static_cast<int>(out_type_.size()); ++i) {
        Tensor* out = nullptr;
      
        ////get the tensor shapes:
        //TensorShape shape;
        //OP_REQUIRES_OK(ctx, ExtractShapeBinary(shape, recs[i]));
        //OP_REQUIRES_OK(ctx, output.allocate(i, shape, &out));
        //
        ////fill values:
        //OP_REQUIRES_OK(ctx, ExtractValuesBinary((*output[i]), out_type_[i], recs[i]));
      }
    }
    
    //the format is:
    // rec = num_buffs|buffs(num_buffs)
    // where each buff has the following data layout:
    // buff = type_buff_size|type_buff(buff_size)|num_dims|dims(num_dims)|data(prod_i dims(i))
    // this routine should just split into individual buffers so that we can parse them one by one
    // later. We are going to use strings because we cannot access the buffer directly.
    // it would be better if we could use StringPiece here instead of string but I will
    // leave that for later optimization
    std::vector<HDF5File::DatasetInfo> SplitRecord(const string& record){
      //create pointer to object for easy iteration
      //string* recpt = &record;
      ////first, check how many buffers are in that record:
      //size_t num_buffs = *reinterpret_cast<hsize_t>(recpt);
      //recpt+=sizeof(hsize_t);
      //
      ////allocate result vector
      //std::vector<string> result(num_buffs);
      
      //
      
    }
    
    void ComputeASCII(OpKernelContext* ctx) {
      const Tensor* records;
      OpInputList record_defaults;
    
      //get records
      OP_REQUIRES_OK(ctx, ctx->input("records", &records));
    
      //make them flat out:
      auto records_t = records->flat<string>();
      int64 records_size = records_t.size();

      std::vector<string> recs;
      for (int64 i = 0; i < records_size; ++i) {
        std::vector<string> tmprecs = str_util::Split(records_t(i), ":");
        recs.insert(recs.end(), tmprecs.begin(), tmprecs.end());
      }
      OP_REQUIRES(ctx, recs.size()==out_type_.size(), errors::InvalidArgument("Error, number of tensors in recordstring does not match the record format specified."));
    
      OpOutputList output;
      OP_REQUIRES_OK(ctx, ctx->output_list("output", &output));
    
      for (int i = 0; i < static_cast<int>(out_type_.size()); ++i) {
        Tensor* out = nullptr;
      
        //get the tensor shapes:
        TensorShape shape;
        OP_REQUIRES_OK(ctx, ExtractShapeASCII(shape, recs[i]));
        OP_REQUIRES_OK(ctx, output.allocate(i, shape, &out));
      
        //fill values:
        OP_REQUIRES_OK(ctx, ExtractValuesASCII((*output[i]), out_type_[i], recs[i]));
      }
    }
    
    //extract shape from record string
    Status ExtractShapeASCII(TensorShape& shape, const string& record){
      auto recordvec = str_util::Split(record, "[");
      CHECK(recordvec.size()==2);
    
      //clear shape
      shape.Clear();
      //infer dims
      string shapestring = recordvec[0];
      shapestring = str_util::StringReplace(shapestring, "(", "", true);
      shapestring = str_util::StringReplace(shapestring, ")", "", true);
      auto dims = str_util::Split(shapestring, ",");
      for(unsigned int d=0; d<dims.size(); d++){
        long long dimsize;
        CHECK(strings::safe_strto64(dims[d], &dimsize));
        shape.AddDim(dimsize);
      }
      return Status::OK();
    }
  
    //extract the content
    Status ExtractValuesASCII(Tensor& tensor, const DataType& dtype, const string& record){
      auto recordvec = str_util::Split(record, ")");
      //parse values
      string valuestring = recordvec[1];
      valuestring = str_util::StringReplace(valuestring, "[", "", true);
      valuestring = str_util::StringReplace(valuestring, "]", "", true);
      auto values = str_util::Split(valuestring, ",");
    
      CHECK(values.size()==tensor.shape().num_elements());
      
      //switch datatypes
      switch (dtype) {
        case DT_INT32: {
          auto tensor_t = tensor.flat<int32>();
          for (unsigned int i = 0; i < values.size(); ++i) {
            int32 val;
            CHECK(strings::safe_strto32(values[i].c_str(), &val));
            tensor_t(i) = val;
          }
          break;
        }
        case DT_INT64: {
          auto tensor_t = tensor.flat<int64>();
          for (unsigned int i = 0; i < values.size(); ++i) {
            int64 val;
            CHECK(strings::safe_strto64(values[i].c_str(), &val));
            tensor_t(i) = val;
          }
          break;
        }
        case DT_FLOAT: {
          auto tensor_t = tensor.flat<float>();
          for (unsigned int i = 0; i < values.size(); ++i) {
            float val;
            CHECK(strings::safe_strtof(values[i].c_str(), &val));
            tensor_t(i) = val;
          }
          break;
        }
        case DT_DOUBLE: {
          auto tensor_t = tensor.flat<double>();
          for (unsigned int i = 0; i < values.size(); ++i) {
            double val;
            CHECK(strings::safe_strtod(values[i].c_str(), &val));
            tensor_t(i) = val;
          }
          break;
        }
        default:
          return errors::InvalidArgument("hdf5: data type ", dtype, " not supported.");
      }
      return Status::OK();
    }
  };

  REGISTER_KERNEL_BUILDER(Name("DecodeHDF5").Device(DEVICE_CPU), DecodeHDF5Op);

}  // namespace tensorflow

#endif