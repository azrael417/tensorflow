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

#include <limits.h>
#include "third_party/hdf5/hdf5.h"

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/lib/io/hdf5_reader.h"

namespace tensorflow {
  namespace io {

    HDF5Reader::HDF5Reader(HDF5File* file, const std::vector<string>& datasets) : src_(file), datasets_(datasets) {
      Status s;
      //initialize all the datasets we are going to read from:
      for(unsigned int d=0; d<datasets_.size(); d++){
        s = src_->InitDataset(datasets_[d]);
        if(!s.ok()){
          LOG(FATAL) << "Dataset " << datasets_[d] << " could not be initialized for file " << src_->GetFilename();
        }
      }
      //initialize current line to zero
      current_line_ = 0;
    }

    Status HDF5Reader::ReadRecord(string* record) {
      Status s;
      
      //read from the first dataset
      hsize_t dsize = static_cast<hsize_t>(datasets_.size());
      string dummy, dummyrecord(reinterpret_cast<char*>(&dsize), sizeof(hsize_t));
      string token;
      s = src_->Read(datasets_[0], current_line_, record, &dummy);
      if( !s.ok() ) return s;
      strings::StrAppend(&dummyrecord, dummy);
      for(unsigned int d=1; d<dsize; d++){
        s = src_->Read(datasets_[d],current_line_, &token, &dummy);
        if( !s.ok() ) return s;
        strings::StrAppend(record,":",token);
        strings::StrAppend(&dummyrecord, dummy);
      }
      current_line_++;
      
      //update status
      s = Status::OK();
      
      return s;
    }

  }  // namespace io
}  // namespace tensorflow

#endif
