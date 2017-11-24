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
#include "tensorflow/core/kernels/decode_hdf5_op.h"


namespace tensorflow {

    
  void DecodeHDF5Op::Compute(OpKernelContext* ctx) {
    const Tensor* records;
    OpInputList record_defaults;
  
    //get records
    OP_REQUIRES_OK(ctx, ctx->input("records", &records));
  
    //make them flat out:
    auto records_t = records->flat<string>();
    int64 records_size = records_t.size();

    std::vector<HDF5File::DatasetInfo> recs;
    for (int64 i = 0; i < records_size; ++i) {
      std::vector<HDF5File::DatasetInfo> tmprecs = ParseRecord(records_t(i));
      recs.insert(recs.end(), tmprecs.begin(), tmprecs.end());
    }
    OP_REQUIRES(ctx, recs.size()==out_type_.size(), errors::InvalidArgument("Error, number of tensors in recordstring does not match the record format specified."));
  
    OpOutputList output;
    OP_REQUIRES_OK(ctx, ctx->output_list("output", &output));
  
    for (int i = 0; i < static_cast<int>(out_type_.size()); ++i) {
      Tensor* out = nullptr;
    
      //get the tensor shapes:
      TensorShape shape;
      for(unsigned int d=0; d<recs[i].ndims; d++){
        shape.AddDim(recs[i].dims[d]);
      }
      OP_REQUIRES_OK(ctx, output.allocate(i, shape, &out));
  
      //fill values:
      switch (out_type_[i]) {
        case DT_INT32: {
          OP_REQUIRES_OK(ctx, ExtractValuesBinary<int32>((*output[i]), recs[i]));
          break;
        }
        case DT_INT64: {
          OP_REQUIRES_OK(ctx, ExtractValuesBinary<int64>((*output[i]), recs[i]));
          break;
        }
        case DT_FLOAT: {
          OP_REQUIRES_OK(ctx, ExtractValuesBinary<float>((*output[i]), recs[i]));
          break;
        }
        case DT_DOUBLE: {
          OP_REQUIRES_OK(ctx, ExtractValuesBinary<double>((*output[i]), recs[i]));
          break;
        }
        default: {
          OP_REQUIRES_OK(ctx, errors::InvalidArgument("data type ", out_type_[i], " not supported."));
        }
      }
    }
    
    //clean up:
    for(unsigned int i=0; i<recs.size(); i++){
      H5Tclose(recs[i].type);
    }
    return;
  }

  //the format is:
  // rec = num_buffs|buffs(num_buffs)
  // where each buff has the following data layout:
  // buff = type_buff_size|type_buff(buff_size)|num_dims|dims(num_dims)|data(prod_i dims(i))
  // this routine should just split into individual buffers so that we can parse them one by one
  // later. We are going to use strings because we cannot access the buffer directly.
  // it would be better if we could use StringPiece here instead of string but I will
  // leave that for later optimization
  std::vector<HDF5File::DatasetInfo> DecodeHDF5Op::ParseRecord(const string& record){
    //create offset for keeping track of index
    hsize_t offset = 0;
    //first, check how many buffers are in that record:
    size_t num_buffs = *(reinterpret_cast<const hsize_t*>(&record[offset]));
    offset += sizeof(hsize_t);
    
    ////allocate result vector
    std::vector<HDF5File::DatasetInfo> result;
    for(size_t d=0; d<num_buffs; ++d){
      
      // get type
      HDF5File::DatasetInfo info;
      hsize_t type_enc_size = *(reinterpret_cast<const hsize_t*>(&record[offset]));
      offset += sizeof(hsize_t);
      char* tmpbuff = new char[type_enc_size];
      memcpy(tmpbuff,&record[offset],type_enc_size);
      offset += type_enc_size;
      info.type_enc = string(tmpbuff,type_enc_size);
      delete [] tmpbuff;
      info.type = H5Tdecode(reinterpret_cast<const unsigned char*>(info.type_enc.c_str()));
      info.type_size = H5Tget_size(info.type);
      
      //get dims:
      info.ndims = *(reinterpret_cast<const hsize_t*>(&record[offset]));
      offset += sizeof(hsize_t);
      info.dims.resize(info.ndims);
      memcpy(&info.dims[0],&record[offset],info.ndims*sizeof(hsize_t));
      offset += info.ndims*sizeof(hsize_t);
      
      //get size of dataset
      info.dset_size=1;
      for(unsigned int i=0; i<info.ndims; i++) info.dset_size*=info.dims[i];
      info.buff = const_cast<char*>(&record[offset]);
      offset += info.dset_size*info.type_size;
      
      //assign to result vector
      result.push_back(info);
    }
    
    //return result
    return result;
  }

  REGISTER_KERNEL_BUILDER(Name("DecodeHDF5").Device(DEVICE_CPU), DecodeHDF5Op);

}  // namespace tensorflow

#endif