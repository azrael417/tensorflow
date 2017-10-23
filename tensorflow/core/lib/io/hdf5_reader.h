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

#ifndef TENSORFLOW_LIB_IO_HDF5_READER_H_
#define TENSORFLOW_LIB_IO_HDF5_READER_H_

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

  class HDF5File;

  namespace io {

    // Low-level interface to read HDF5 files.
    // Note: this class is not thread safe; external synchronization required.
    class HDF5Reader {
    public:
      // Create a reader that will return log records from "*file".
      // "*file" must remain live while this Reader is in use.
      explicit HDF5Reader(HDF5File* file, const std::vector<string>& datasets);
      
      virtual ~HDF5Reader() = default;

      // Read the record at "line_number" for all datasets into *record. 
      // Returns OK on success, OUT_OF_RANGE for end of file, or something else for an error.
      Status ReadRecord(string* record);
      
      //get the current line in the file
      size_t GetLine() const{ return current_line_; }
      
    private:
      HDF5File* src_;
      std::vector<string> datasets_;
      size_t current_line_;

      TF_DISALLOW_COPY_AND_ASSIGN(HDF5Reader);
    };

  }  // namespace io
}  // namespace tensorflow

#endif  // TENSORFLOW_LIB_IO_HDF5_READER_H_
