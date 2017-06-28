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

//#ifdef TENSORFLOW_USE_HDF5

//standard stuff
#include "tensorflow/core/framework/reader_op_kernel.h"
#include "tensorflow/core/framework/reader_base.h"
#include "tensorflow/core/lib/core/errors.h"

//specific stuff
#include <hdf5.h>
#include "tensorflow/core/lib/strings/str_util.h"

#include <sys/stat.h>

namespace tensorflow {

    inline void HDF5_CHECK_FILE_EXISTS(hid_t hdf5_status) {
        CHECK_LT(hdf5_status, 0) << "Error, file does not exist.";
        CHECK_EQ(hdf5_status, 0) << "Error, file exists but is not a valid HDF5 file.";
    }
    
    inline void HDF5_CHECK_FILE(hid_t hdf5_status) {
        CHECK_LT(hdf5_status, 0) << "Error, cannot open file.";
    }
    
    inline void HDF5_CHECK_OBJECT_EXISTS(hid_t hdf5_status){
        CHECK_LT(hdf5_status, 0) << "Error, object lookup failed.";
        CHECK_EQ(hdf5_status, 0) << "Error, object does not exist.";
    }
    
    inline void HDF5_CHECK_OBJECT_IS_DATASET(const H5O_info_t& objinfo){
        CHECK_NE(objinfo.type, H5O_TYPE_DATASET) << "Error, object is not a dataset.";
    }
    
    inline void HDF5_CHECK_OK(hid_t hdf5_status, const string& description){
        CHECK_LT(hdf5_status, 0) << "Error, " << description <<  ".";
    }
    
    class HDF5Reader : public ReaderBase {
    public:
        HDF5Reader(const string& node_name, const string& dataset_namestring, Env* env)
            : ReaderBase(strings::StrCat("HDF5Reader '", node_name, "'")),
        env_(env),
        hdf5_env_(nullptr),
        row_num_(0),
        num_rows_(0) {
            //split dataset_namestring at ':' and store in vector:
            hdf5_dset_names_ = str_util::Split(dataset_namestring, ":", str_util::SkipEmpty());
        }


        Status OnWorkStartedLocked() override {
            //check if file is actually and hdf5 file:
            HDF5_CHECK_FILE_EXISTS(H5Fis_hdf5(current_work().c_str()));
            //survived that call? try opening it in readonly mode
            unsigned int flags = H5F_ACC_RDONLY;
            hdf5_env_ = H5Open(current_work().c_str(),flags,H5P_DEFAULT);
            HDF5_CHECK_FILE(hdf5_env_);

            //what should follow now is to see if the datasets we want to read are actually there,
            //but for that I need to find out how to pass that list to the routine first
            for(unsigned int i=0; i<hdf5_dset_names_.size(); ++i){
                
                //we do not need to check everything along the path because we want the routine to error out even if the 
                //full path does not exist
                HDF5_CHECK_OBJECT_EXISTS(H5Oexists_by_name( hdf5_env_, hdf5_dset_names_[i].c_str(), H5P_DEFAULT ));
                
                //check if object is a dataset
                H5O_info_t object_info;
                H5Oget_info_by_name( hdf5_env_, hdf5_dset_names_[i].c_str(), &object_info, H5P_DEFAULT );
                HDF5_CHECK_OBJECT_IS_DATASET(object_info);
                
                //open the dataset and push handle into a list
                hid_t dsetid = H5DOpen(hdf5_env_, hdf5_dset_names_[i].c_str(), H5P_DEFAULT);
                HDF5_CHECK_OK(dsetid,"cannot open dataset " + hdf5_dset_names_[i]);
                hdf5_dset_ids_.push_back(dsetid);
                
                //get the memory space id
                hid_t memid = H5Dget_space(dsetid);
                HDF5_CHECK_OK(memid,"cannot open memory space for dataset "+hdf5_dset_names_[i]);
                hdf5_dset_memids_.push_back(memid);
                
                //determine dimensionality of dataset:
                int ndims = H5Sget_simple_extent_ndims( memid );
                
                //determine dimensionalities and append to list:
                std::vector<hsize_t> dims(ndims);
                H5Sget_simple_extent_dims(memid, dims.data(), NULL);
                hdf5_dset_dims_.push_back(dims);
            }
            
            //check if the size of all 0-axis is the same across the datasets used:
            int num_rows_ = hdf5_dset_dims_[0][0];
            for(unsigned i=1; i<hdf5_dset_dims_.size(); ++i){
                CHECK_LT(num_rows_, hdf5_dset_dims_[i][0]) << "Error, datasets " << hdf5_dset_names_[0] <<  " and " << hdf5_dset_names_[i] << " do not have the same extents in axis 0.";
            }
            //do not cache more rows than available
            num_rows_cached_ = min(num_rows_, num_rows_cached_);
            
            //initialize input buffers:
            for(unsigned i=1; i<hdf5_dset_dims_.size(); ++i){
                unsigned int size = 1;
                for(unsigned int d=1; d<hdf5_dset_dims_[i].size(); d++) size *= hdf5_dset_dims_[i][d];
                float* tmpbuf = new float[size];
                hdf5_dset_buffers_[i].push_back(tmpbuf);
            }
            
            return Status::OK();
        }


        Status OnWorkFinishedLocked() override {
            if (hdf5_env_ != nullptr) {
                //close everything which is currently open
                herr_t err=1;
                unsigned int types=H5F_OBJ_DATASET | H5F_OBJ_GROUP | H5F_OBJ_DATATYPE | H5F_OBJ_ATTR;

                //get number of objects still open:
                ssize_t num_open = H5Fget_obj_count(hdf5_env_,types);
                if (num_open > 0) { 
                    std::vector<hid_t> open_object_ids(num_open, 0); 
                    H5Fget_obj_ids(hdf5_env_, types, num_open, &(open_object_ids.front()) ); 
                    for(unsigned int i=0; i<num_open; ++i){
                        err = H5Oclose(open_object_ids[i]);
                    }
                }
                
                //free buffers
                for(unsigned int i=0; i<hdf5_dset_names_.size(); ++i){
                    if(hdf5_dset_buffers_[i] != nullptr){
                        delete [] hdf5_dset_buffers_[i];
                    }
                }
                
                //close file and reset everything
                err = H5Fclose(hdf5_env_);
                num_rows_ = 0;
                row_num_ = 0;
                num_rows_cached_ = 128;
                hdf5_dset_names_.clear();
                hdf5_dset_ids_.clear();
                hdf5_dset_memids_.clear();
                hdf5_dset_dims_.clear();
                hdf5_dset_buffers_.clear();
                hdf5_env_ = nullptr;
            }
            return Status::OK();
        }


        string ReadRow(const unsigned int& dset_index){
            //vector arrays
            std::vector<hsize_t> start(hdf5_dset_dims_[dset_index_].size()), count(hdf5_dset_dims_[dset_index_].size());
            
            //set hyperslab parameters
            start[0] = row_num_;
            count[0] = 1;
            for(unsigned int i=0; i<hdf5_dset_dims_[dset_index].size(); ++i){
                start[i] = 0;
                count[i] = hdf5_dset_dims_[dset_index];
            }
            //select the slab, backup the old one
            hid_t file_space = hdf5_dset_memids_[dset_index];
            HDF5_CHECK_OK(HDF5_CHECK_OK(H5Sselect_hyperslab(&file_space, H5S_SELECT_SET, start.data(), NULL, count.data(), NULL), "file-hyperslab for dataset " + hdf5_dset_names_[dset_index]+".");
            hid_t mem_space = H5Screate_simple(static_cast<hsize_t>(hdf5_dset_dims_[dset_index].size()), hdf5_dset_dims_[dset_index].data(), NULL);
            HDF5_CHECK_OK(hslab_mem,"cannot create memory space.");
            
            //read from the slab
            HDF5_CHECK_OK(H5Dread(hdf5_dset_ids_[dset_index], H5T_NATIVE_FLOAT, mem_space, file_space, hid_t xfer_plist_id, hdf5_dset_buffers_[dset_index_]),"cannot read row "+std::to_string(row_num_)+" from dataset "+hdf5_dset_names_[dset_index]|+".");
            
            //create output string
            result='';
            const int run = 0;
            //skip first dimension because this is n-sample dimension
            for(unsigned int d=1; d<hdf5_dset_dims_[dset_index].size(); ++d){
                result += to_string(hdf5_dset_buffers_[dset_index_][run]);
                run++;
                for(unsigned int i=1; i<hdf5_dset_dims_[dset_index][d].size(); ++i){
                    result += ','+to_string(hdf5_dset_buffers_[dset_index_][run]);
                    run++;
                }
                result += ";";
            }
            
            return result;
        }


        Status ReadLocked(string* key, string* value, bool* produced, bool* at_end) override {
            //reached the end of the file
            if(row_num_ >= num_rows_) {
                *at_end = true;
                return Status::OK();
            }
            
            //not at end of file? produce the key
            string keystring = hdf5_dset_names[0];
            for(unsigned int i=1; i<hdf5_dset_names_.size(); ++i){
                keystring +=  ":" + hdf5_dset_names_[i];
            }
            keystring += '@' + current_work();
            *key = strings::StrCat(keystring, ":", row_num_);
            
            //now take care of value
            *value = ReadRow(0);
            for(unsigned int i=1; i<hdf5_dset_names_.size(); i++){
                *value += ":" + ReadRow(i);
            }
            
            //increase row number
            row_num_++;
            
            //mark as produced and say goodbye
            *produced = true;
            return Status::OK();
        }

        Status ResetLocked() override {
            //reset row number
            row_num_ = 0;
            //return reset OK:
            return ReaderBase::ResetLocked();
        }

    private:
        Env* const env_;
        hid_t hdf5_env_;
        std::vector<string> hdf5_dset_names_;
        std::vector<hid_t> hdf5_dset_ids_, hdf5_dset_memids_;
        std::vector< std::vector<hsize_t> > hdf5_dset_dims_;
        std::vector<float*> hdf5_dset_buffers_;
        unsigned int row_num_, num_rows_;
    };

    class HDF5ReaderOp : public ReaderOpKernel {
    public:
        explicit HDF5ReaderOp(OpKernelConstruction* context)
        : ReaderOpKernel(context) {
            string dataset_namestring = "";
            OP_REQUIRES_OK(context, context->GetAttr("datasets", &dataset_namestring));
            OP_REQUIRES(context, dataset_namestring != "",
                        errors::InvalidArgument("please provide a (list of) dataset name(s) to load from"));
            Env* env = context->env();
            SetReaderFactory([this, dataset_namestring, env]() {
                return new HDF5Reader(name(), env);
            });
        }
    };

    REGISTER_KERNEL_BUILDER(Name("HDF5Reader").Device(DEVICE_CPU), HDF5ReaderOp);

}

//#endif