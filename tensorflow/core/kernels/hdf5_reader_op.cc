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

//specific stuff
#include "third_party/hdf5/hdf5.h"
#include "tensorflow/core/lib/strings/str_util.h"

#include <sys/stat.h>

namespace tensorflow {

    inline void HDF5_CHECK_FILE_EXISTS(hid_t hdf5_status) {
        CHECK_NE(hdf5_status, 0) << "Error, file exists but is not a valid HDF5 file.";
        CHECK_GT(hdf5_status, 0) << "Error, file does not exist.";
    }
    
    inline void HDF5_CHECK_FILE(hid_t hdf5_status, const string& filename) {
        CHECK_GE(hdf5_status, 0) << "Error, could not open file " << filename << ".";
    }
    
    inline void HDF5_CHECK_OBJECT_EXISTS(hid_t hdf5_status, const string& objname){
        CHECK_NE(hdf5_status, 0) << "Error, object " << objname << " does not exist.";
        CHECK_GT(hdf5_status, 0) << "Error, object lookup for " << objname << " failed.";
    }
    
    inline void HDF5_CHECK_OBJECT_IS_DATASET(const H5O_info_t& objinfo, const string& objname){
        CHECK_EQ(objinfo.type, H5O_TYPE_DATASET) << "Error, object " << objname << " is not a dataset.";
    }
    
    inline void HDF5_CHECK_OK(hid_t hdf5_status, const string& description){
        CHECK_GE(hdf5_status, 0) << "Error, " << description <<  ".";
    }
    
    class HDF5Reader : public ReaderBase {
    public:
        HDF5Reader(const string& node_name, std::vector<string> datasets, Env* env) : ReaderBase(strings::StrCat("HDF5Reader '", node_name, "'")), 
        env_(env), hdf5_env_(0), hdf5_dset_names_(datasets), row_num_(0), num_rows_(0) {
            
            //default property list for data access
            plist_id_ = H5Pcreate(H5P_DATASET_XFER);
            
            //fill supported types vector: using initializer list but that can cause trouble with some compilers:
            supported_types_ = {H5T_NATIVE_FLOAT, H5T_NATIVE_DOUBLE, H5T_NATIVE_INT};
        }
        
        ~HDF5Reader(){
            H5Pclose(plist_id_);
        }

        Status OnWorkStartedLocked() override {
            
            //check if file is actually and hdf5 file:
            HDF5_CHECK_FILE_EXISTS(H5Fis_hdf5(current_work().c_str()));
            //survived that call? try opening it in readonly mode
            unsigned int flags = H5F_ACC_RDONLY;
            hdf5_env_ = H5Fopen(current_work().c_str(),flags,H5P_DEFAULT);
            HDF5_CHECK_FILE(hdf5_env_, current_work());

            //what should follow now is to see if the datasets we want to read are actually there,
            //but for that I need to find out how to pass that list to the routine first
            for(unsigned int i=0; i<hdf5_dset_names_.size(); ++i){
                
                //we do not need to check everything along the path because we want the routine to error out even if the 
                //full path does not exist
                HDF5_CHECK_OBJECT_EXISTS(H5Oexists_by_name( hdf5_env_, hdf5_dset_names_[i].c_str(), H5P_DEFAULT ), hdf5_dset_names_[i]);
                
                //check if object is a dataset
                H5O_info_t object_info;
                H5Oget_info_by_name( hdf5_env_, hdf5_dset_names_[i].c_str(), &object_info, H5P_DEFAULT );
                HDF5_CHECK_OBJECT_IS_DATASET(object_info, hdf5_dset_names_[i]);
                
                //open the dataset and push handle into a list
                hid_t dsetid = H5Dopen(hdf5_env_, hdf5_dset_names_[i].c_str(), H5P_DEFAULT);
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
                //if vector is provided, attach singleton dimension. that should not do harm to the rest of the code
                if(ndims==1) dims.push_back(1);
                hdf5_dset_dims_.push_back(dims);
                
                //check datatype, only numeric types supported so far:
                hid_t type_id = H5Dget_type(dsetid);
                bool type_supported=false;
                for(unsigned int i=0; i<supported_types_.size(); ++i){
                    if( H5Tequal(type_id,supported_types_[i]) > 0 ){
                        type_supported=true;
			hdf5_dset_types_.push_back(type_id);
                        break;
                    }
                }
                CHECK_EQ(type_supported,true) << "Error, datatype of dataset " << hdf5_dset_names_[i] << " not supported. Only Numeric types are supported at the moment.";
            }
            
            //check if the size of all 0-axis is the same across the datasets used:
            num_rows_ = hdf5_dset_dims_[0][0];
            for(unsigned i=1; i<hdf5_dset_dims_.size(); ++i){
                CHECK_EQ(num_rows_, hdf5_dset_dims_[i][0]) << "Error, datasets " << hdf5_dset_names_[0] <<  " and " << hdf5_dset_names_[i] << " do not have the same extents in axis 0.";
            }
            
            //initialize input buffers:
            for(unsigned i=0; i<hdf5_dset_dims_.size(); ++i){
                
                //compute sizes for buffers
                unsigned int size = 1;
                for(unsigned int d=1; d<hdf5_dset_dims_[i].size(); d++) size *= hdf5_dset_dims_[i][d];
                hdf5_dset_sizes_.push_back(size);
                
                //allocate buffers
		int typesize=H5Tget_size(hdf5_dset_types_[i]);
                char* tmpbuf = new char[size*typesize];
                hdf5_dset_buffers_.push_back(tmpbuf);
            }
            
            return Status::OK();
        }


        Status OnWorkFinishedLocked() override {
            if (hdf5_env_ > 0) {
                //close everything which is currently open
                unsigned int types = H5F_OBJ_DATASET | H5F_OBJ_GROUP | H5F_OBJ_DATATYPE | H5F_OBJ_ATTR;

                //get number of objects still open:
                ssize_t num_open = H5Fget_obj_count(hdf5_env_,types);
                if (num_open > 0) { 
                    std::vector<hid_t> open_object_ids(num_open, 0); 
                    H5Fget_obj_ids(hdf5_env_, types, num_open, &(open_object_ids.front()) ); 
                    for(unsigned int i=0; i<num_open; ++i){
                        H5Oclose(open_object_ids[i]);
                    }
                }
                
                //free buffers
                for(unsigned int i=0; i<hdf5_dset_names_.size(); ++i){
                    if(hdf5_dset_buffers_[i] != nullptr){
                        delete [] hdf5_dset_buffers_[i];
                    }
                }
                
                //close file and reset everything
                H5Fclose(hdf5_env_);
                num_rows_ = 0;
                row_num_ = 0;
                hdf5_dset_names_.clear();
                hdf5_dset_ids_.clear();
                hdf5_dset_memids_.clear();
                hdf5_dset_dims_.clear();
                hdf5_dset_buffers_.clear();
                hdf5_dset_sizes_.clear();
		hdf5_dset_types_.clear();
                hdf5_env_ = 0;
            }
            return Status::OK();
        }


        Status ReadLocked(string* key, string* value, bool* produced, bool* at_end) override {
            
            //reached the end of the file
            if(row_num_ >= num_rows_) {
                *at_end = true;
                return Status::OK();
            }
            
            //not at end of file? produce the key
            string keystring = hdf5_dset_names_[0];
            for(unsigned int i=1; i<hdf5_dset_names_.size(); ++i){
                keystring = strings::StrCat(keystring, ":", hdf5_dset_names_[i]);
            }
            *key = strings::StrCat(keystring, "@", current_work(), ":", row_num_);
                        
            //now take care of value
            *value = ReadRow(0);
            for(unsigned int i=1; i<hdf5_dset_names_.size(); i++){
                *value = strings::StrCat(*value, ":", ReadRow(i));
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
        std::vector<hid_t> hdf5_dset_ids_, hdf5_dset_memids_, hdf5_dset_types_, supported_types_;
        std::vector< std::vector<hsize_t> > hdf5_dset_dims_;
        std::vector<char*> hdf5_dset_buffers_;
        std::vector<size_t> hdf5_dset_sizes_;
        unsigned int row_num_, num_rows_;
        hid_t plist_id_;

        
        inline string EncodeTokenASCII(char* buff, const hid_t& type_id){
	  char* result = new char[strings::kFastToBufferSize];
	  if(H5Tequal(type_id,H5T_NATIVE_FLOAT)){
	    float tmpval = (*reinterpret_cast<float*>(buff));
	    strings::FloatToBuffer(tmpval,result);
	  }
	  else if(H5Tequal(type_id,H5T_NATIVE_INT)){
	    int tmpval = (*reinterpret_cast<int*>(buff));
	    strings::FastInt32ToBufferLeft(tmpval,result);
	  }
	  else if(H5Tequal(type_id,H5T_NATIVE_DOUBLE)){
	    double tmpval = (*reinterpret_cast<double*>(buff));
	    strings::DoubleToBuffer(tmpval,result);
	  }
	  string res(result);
	  delete result;
	  return res;
        }


        string EncodeASCII(const unsigned int& dset_index){
            //create output string
            //first, get dimensions
            string result = strings::StrCat("(", hdf5_dset_dims_[dset_index][1]);
            for(unsigned int d=2; d<hdf5_dset_dims_[dset_index].size(); ++d){
                result = strings::StrCat(result, ",", hdf5_dset_dims_[dset_index][d]);
            }
            result = strings::StrCat(result, ")[");
            //now get the data
	    int typesize = H5Tget_size(hdf5_dset_types_[dset_index]);
            result = strings::StrCat(result, EncodeTokenASCII(&(hdf5_dset_buffers_[dset_index][0]), hdf5_dset_types_[dset_index]));
            for(unsigned int r=1; r<hdf5_dset_sizes_[dset_index]; ++r){
	      result = strings::StrCat(result, ",", EncodeTokenASCII(&(hdf5_dset_buffers_[dset_index][r*typesize]), hdf5_dset_types_[dset_index]));
            }
            result = strings::StrCat(result, "]");
            
            return result;
        }


        string ReadRow(const unsigned int& dset_index){
            //vector arrays
            std::vector<hsize_t> start(hdf5_dset_dims_[dset_index].size()), count(hdf5_dset_dims_[dset_index].size());
            
            //set hyperslab parameters
            start[0] = row_num_;
            count[0] = 1;
            for(unsigned int i=1; i<hdf5_dset_dims_[dset_index].size(); ++i){
                start[i] = 0;
                count[i] = hdf5_dset_dims_[dset_index][i];
            }
            
            //select the slab, backup the old one
            hid_t file_space = hdf5_dset_memids_[dset_index];
            HDF5_CHECK_OK(H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start.data(), NULL, count.data(), NULL), "file-hyperslab for dataset " + hdf5_dset_names_[dset_index]+".");
            hid_t mem_space = H5Screate_simple(static_cast<hsize_t>(hdf5_dset_dims_[dset_index].size()-1), &(hdf5_dset_dims_[dset_index][1]), NULL);
            HDF5_CHECK_OK(mem_space,"cannot create memory space.");
            
            //read from the slab
	    if(H5Tequal(hdf5_dset_types_[dset_index],H5T_NATIVE_FLOAT)){
	      HDF5_CHECK_OK(H5Dread(hdf5_dset_ids_[dset_index], hdf5_dset_types_[dset_index], mem_space, file_space, plist_id_, reinterpret_cast<float*>(hdf5_dset_buffers_[dset_index])),strings::StrCat("cannot read row ",row_num_," from dataset ",hdf5_dset_names_[dset_index],"."));
	    }
	    else if(H5Tequal(hdf5_dset_types_[dset_index],H5T_NATIVE_INT)){
	      HDF5_CHECK_OK(H5Dread(hdf5_dset_ids_[dset_index], hdf5_dset_types_[dset_index], mem_space, file_space, plist_id_, reinterpret_cast<int*>(hdf5_dset_buffers_[dset_index])),strings::StrCat("cannot read row ",row_num_," from dataset ",hdf5_dset_names_[dset_index],"."));
	    }
	    else if(H5Tequal(hdf5_dset_types_[dset_index],H5T_NATIVE_DOUBLE)){
	      HDF5_CHECK_OK(H5Dread(hdf5_dset_ids_[dset_index], hdf5_dset_types_[dset_index], mem_space, file_space, plist_id_, reinterpret_cast<double*>(hdf5_dset_buffers_[dset_index])),strings::StrCat("cannot read row ",row_num_," from dataset ",hdf5_dset_names_[dset_index],"."));
	    }
            
	    //close the spaces:
	    H5Sclose(file_space);
	    H5Sclose(mem_space);

            return EncodeASCII(dset_index);
        }
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
