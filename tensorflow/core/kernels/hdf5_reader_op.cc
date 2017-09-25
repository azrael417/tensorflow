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

//standard stuff
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/io/hdf5_reader.h"

//specific stuff
#include "third_party/hdf5/hdf5.h"
//#include "tensorflow/core/lib/strings/str_util.h"

#include <sys/stat.h>

namespace tensorflow {
    
  class HDF5Reader : public ReaderBase {
  public:
    HDF5Reader(const string& node_name, std::vector<string> datasets, Env* env) : ReaderBase(strings::StrCat("HDF5Reader '", node_name, "'")), 
    env_(env), datasets_(datasets){}


    ~HDF5Reader(){}


    Status OnWorkStartedLocked() override {
      TF_RETURN_IF_ERROR(env_->NewHDF5File(current_work(), &file_));
      reader_.reset(new io::HDF5Reader(file_.get(), datasets_));
      
      return Status::OK();
    }


    Status OnWorkFinishedLocked() override {
      //reset everything
      reader_.reset();
      file_.reset();
      
      return Status::OK();
    }


    Status ReadLocked(string* key, string* value, bool* produced, bool* at_end) override {

      //reached the end of the file
      Status s = reader_->ReadRecord(value);
      if (errors::IsOutOfRange(s)) {
        *at_end = true;
        return Status::OK();
      }
      
      //not at end of file? produce the key
      string keystring = datasets_[0];
      for(unsigned int i=1; i<datasets_.size(); ++i){
        strings::StrAppend(&keystring, ":", datasets_[i]);
      }
      *key = strings::StrCat(keystring, "@", current_work(), ":", reader_->GetLine());

      //mark as produced and say goodbye
      *produced = true;
      return Status::OK();
    }


    Status ResetLocked() override {
      //reset everything
      reader_.reset();
      file_.reset();
      //return reset OK:
      return ReaderBase::ResetLocked();
    }

  private:
    Env* const env_;
    std::vector<string> datasets_;
    std::unique_ptr<HDF5File> file_;
    std::unique_ptr<io::HDF5Reader> reader_;
  };


  class HDF5ReaderOp : public ReaderOpKernel {
  public:
    explicit HDF5ReaderOp(OpKernelConstruction* context)
    : ReaderOpKernel(context) {
      std::vector<string> datasets;
      OP_REQUIRES_OK(context, context->GetAttr("datasets", &datasets));
      OP_REQUIRES(context, datasets.size() > 0,
      errors::InvalidArgument("please provide a (list of) dataset name(s) to load from"));
      Env* env = context->env();
      SetReaderFactory([this, datasets, env]() {
        return new HDF5Reader(name(), datasets, env);
      });
    }
  };

  REGISTER_KERNEL_BUILDER(Name("HDF5Reader").Device(DEVICE_CPU), HDF5ReaderOp);

}

#endif
