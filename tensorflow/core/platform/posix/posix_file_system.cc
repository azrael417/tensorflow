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

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "tensorflow/core/lib/core/error_codes.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/posix/error.h"
#include "tensorflow/core/platform/posix/posix_file_system.h"

#ifdef TENSORFLOW_USE_HDF5
//specific stuff
#include "third_party/hdf5/hdf5.h"
#endif

namespace tensorflow {

// pread() based random-access
class PosixRandomAccessFile : public RandomAccessFile {
 private:
  string filename_;
  int fd_;

 public:
  PosixRandomAccessFile(const string& fname, int fd)
      : filename_(fname), fd_(fd) {}
  ~PosixRandomAccessFile() override { close(fd_); }

  Status Read(uint64 offset, size_t n, StringPiece* result,
              char* scratch) const override {
    Status s;
    char* dst = scratch;
    while (n > 0 && s.ok()) {
      ssize_t r = pread(fd_, dst, n, static_cast<off_t>(offset));
      if (r > 0) {
        dst += r;
        n -= r;
        offset += r;
      } else if (r == 0) {
        s = Status(error::OUT_OF_RANGE, "Read less bytes than requested");
      } else if (errno == EINTR || errno == EAGAIN) {
        // Retry
      } else {
        s = IOError(filename_, errno);
      }
    }
    *result = StringPiece(scratch, dst - scratch);
    return s;
  }
};


#ifdef TENSORFLOW_USE_HDF5
// structured-access to HDF5
class PosixHDF5File : public StructuredAccessFile {
 private:
  string filename_;
  int fd_;
  hid_t hdf5_fd_;
  hid_t plist_id_;
  
  struct DatasetInfo{
    std::vector<hsize_t> dims;
    hid_t id;
    hid_t type;
  };
  
  std::map<string, DatasetInfo> dsetinfo;
  
  Status hdf5_check_file_exists_open(const string& fname, hid_t& fd) {
    Status s;
    unsigned int flags = H5F_ACC_RDONLY;
    
    hid_t hdf5_status = H5Fis_hdf5(fname.c_str());
    if(hdf5_status <= 0){
      s = IOError(fname, int(hdf5_status));
    }
    else{
      fd = H5Fopen(fname.c_str(), flags, H5P_DEFAULT);
      if(fd <= 0){
        s = IOError(fname, int(fd));
      }
      else{
        s = Status::OK();
      }
    }
    return s;
  }
  
  
  Status hdf5_check_dataset_exists(const string& dname) const{
    Status s;
    
    //check if object exists and has right type
    hid_t hdf5_status = H5Oexists_by_name( hdf5_fd_, dname.c_str(), H5P_DEFAULT );
    if(hdf5_status<=0){
      s = IOError("object: "+dname+" does not exist",hdf5_status);
    }
    else{
      H5O_info_t object_info;
      H5Oget_info_by_name( hdf5_fd_, dname.c_str(), &object_info, H5P_DEFAULT );
      if(object_info.type!=H5O_TYPE_DATASET){
        s = IOError("dataset: "+dname+" not a dataset",object_info.type);
      }
      else{
        s = Status::OK();
      }
    }
    return s;
  }
  
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
    else if(H5Tequal(type_id,H5T_NATIVE_LONG)){
      long tmpval = (*reinterpret_cast<long*>(buff));
      strings::FastInt64ToBufferLeft(tmpval,result);
    }
    else if(H5Tequal(type_id,H5T_NATIVE_DOUBLE)){
      double tmpval = (*reinterpret_cast<double*>(buff));
      strings::DoubleToBuffer(tmpval,result);
    }
    string res(result);
    delete result;
    return res;
  }
  
  string EncodeASCII(const DatasetInfo& info, const hsize_t& buff_size, char* buff){
    //create output string
    //create dimensions
    string result = strings::StrCat("(", info.dims[1]);
    for(unsigned int d=2; d<info.dims.size(); ++d){
      result = strings::StrCat(result, ",", info.dims[d]);
    }
    result = strings::StrCat(result, ")[");
    //now get the data
    int typesize = H5Tget_size(info.type);
    result = strings::StrCat(result, EncodeTokenASCII(&(buff[0]), info.type));
    for(unsigned int r=1; r<buff_size; ++r){
      result = strings::StrCat(result, ",", EncodeTokenASCII(&(buff[r*typesize]), info.type));
    }
    result = strings::StrCat(result, "]");
          
    return result;
  }

 public:
   
  PosixHDF5File(const string& fname, int fd) : filename_(fname), fd_(fd) {
    //close the handle immediately, so that we could call H5Open on it:
    close(fd_);
    
    //default property list for data access
    plist_id_ = H5Pcreate(H5P_DATASET_XFER);
    
    //do some checks:
    Status s = hdf5_check_file_exists_open(fname, hdf5_fd_);
  }
  
  
  ~PosixHDF5File() override { 
    //close the objects I still have a handle on:
    std::map<string, DatasetInfo>::iterator it;
    for(it=dsetinfo.begin(); it!=dsetinfo.end(); ++it){
      //close type
      H5Tclose(it->second.type);
      //close dataset
      H5Dclose(it->second.id);
    }
    
    //clear the map
    dsetinfo.clear();
    
    //close everything which remains open
    unsigned int types = H5F_OBJ_DATASET | H5F_OBJ_GROUP | H5F_OBJ_DATATYPE | H5F_OBJ_ATTR;

    //get number of objects still open:
    ssize_t num_open = H5Fget_obj_count(hdf5_fd_, types);
    if (num_open > 0) { 
      std::vector<hid_t> open_object_ids(num_open, 0); 
      H5Fget_obj_ids(hdf5_fd_, types, num_open, &(open_object_ids.front()) ); 
      for(unsigned int i=0; i<num_open; ++i){
        H5Oclose(open_object_ids[i]);
      }
    }
    
    //close porperty list
    H5Pclose(plist_id_);
    
    //close the file finally
    H5Fclose(hdf5_fd_);
  }


  Status InitDataset(const string& dset){
    Status s;
    //first, check if dataset exists:
    s = hdf5_check_dataset_exists(dset);
    if(s != Status::OK()) return s;
    
    //create temporary object
    DatasetInfo info;
    
    //open dataset
    info.id = H5Dopen(hdf5_fd_, dset.c_str(), H5P_DEFAULT);
    
    //get space:
    hid_t mem_id = H5Dget_space( info.id );
    
    //determine dimensionality of dataset:
    int ndims = H5Sget_simple_extent_ndims( mem_id );
    info.dims.resize(ndims);
    H5Sget_simple_extent_dims(mem_id, info.dims.data(), NULL);
    
    //if vector is provided, attach singleton dimension.
    if(ndims==1) info.dims.push_back(1);
    
    //check datatype, only numeric types supported so far:
    info.type = H5Tget_native_type(H5Dget_type(info.id),H5T_DIR_DESCEND);
  
    //add to hashmap
    dsetinfo[dset] = info;
    
    //close objects:
    H5Sclose(mem_id);
    
    return s;
  }


  Status Read(const string& dset, const long& row_num, StringPiece* result) {
    Status s;
    
    //if dataset not initialized, do it now:
    if(dsetinfo.find( dset ) != dsetinfo.end()) s = InitDataset(dset);
    
    //get size needed for buffer
    DatasetInfo info = dsetinfo[dset];
    hsize_t buff_size = 1;
    for(unsigned int d=1; d<info.dims.size(); d++) buff_size *= info.dims[d];
    
    //allocate buffers
    char* buff = new char[buff_size];
    
    //set hyperslab parameters
    std::vector<hsize_t> start, count;
    //read one row only
    start[0] = row_num;
    count[0] = 1;
    //read the full thing for the rest
    for(unsigned int d=1; d<info.dims.size(); ++d){
      start[d] = 0;
      count[d] = info.dims[d];
    }
    
    //no checks here as those were already performed before
    hid_t file_space = H5Dget_space(info.id);
    H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start.data(), NULL, count.data(), NULL);
    hid_t mem_space = H5Screate_simple(static_cast<hsize_t>(info.dims.size()-1), &(info.dims[1]), NULL);
          
    //read from the slab
    if( H5Tequal(info.type, H5T_NATIVE_FLOAT) ){
      H5Dread(info.id, info.type, mem_space, file_space, plist_id_, reinterpret_cast<float*>(buff));
    }
    else if( H5Tequal(info.type, H5T_NATIVE_INT) ){
      H5Dread(info.id, info.type, mem_space, file_space, plist_id_, reinterpret_cast<int*>(buff));
    }
    else if( H5Tequal(info.type, H5T_NATIVE_LONG) ){
      H5Dread(info.id, info.type, mem_space, file_space, plist_id_, reinterpret_cast<long*>(buff));
    }
    else if( H5Tequal(info.type, H5T_NATIVE_DOUBLE) ){
      H5Dread(info.id, info.type, mem_space, file_space, plist_id_, reinterpret_cast<double*>(buff));
    }
    else{
      s = Status(error::OUT_OF_RANGE," error, datatype currently not supported.");
    }
          
    //close the spaces:
    H5Sclose(file_space);
    H5Sclose(mem_space);
    
    string tmpresult = EncodeASCII(info,buff_size,buff);
    *result = StringPiece(tmpresult);
    s = Status::OK();
    
    //free buffer
    delete [] buff;
    
    return s;
  }
};
#endif


class PosixWritableFile : public WritableFile {
 private:
  string filename_;
  FILE* file_;

 public:
  PosixWritableFile(const string& fname, FILE* f)
      : filename_(fname), file_(f) {}

  ~PosixWritableFile() override {
    if (file_ != nullptr) {
      // Ignoring any potential errors
      fclose(file_);
    }
  }

  Status Append(const StringPiece& data) override {
    size_t r = fwrite(data.data(), 1, data.size(), file_);
    if (r != data.size()) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

  Status Close() override {
    Status result;
    if (fclose(file_) != 0) {
      result = IOError(filename_, errno);
    }
    file_ = nullptr;
    return result;
  }

  Status Flush() override {
    if (fflush(file_) != 0) {
      return IOError(filename_, errno);
    }
    return Status::OK();
  }

  Status Sync() override {
    Status s;
    if (fflush(file_) != 0) {
      s = IOError(filename_, errno);
    }
    return s;
  }
};

class PosixReadOnlyMemoryRegion : public ReadOnlyMemoryRegion {
 public:
  PosixReadOnlyMemoryRegion(const void* address, uint64 length)
      : address_(address), length_(length) {}
  ~PosixReadOnlyMemoryRegion() override {
    munmap(const_cast<void*>(address_), length_);
  }
  const void* data() override { return address_; }
  uint64 length() override { return length_; }

 private:
  const void* const address_;
  const uint64 length_;
};

Status PosixFileSystem::NewRandomAccessFile(
    const string& fname, std::unique_ptr<RandomAccessFile>* result) {
  string translated_fname = TranslateName(fname);
  Status s;
  int fd = open(translated_fname.c_str(), O_RDONLY);
  if (fd < 0) {
    s = IOError(fname, errno);
  } else {
    result->reset(new PosixRandomAccessFile(translated_fname, fd));
  }
  return s;
}

Status PosixFileSystem::NewWritableFile(const string& fname,
                                        std::unique_ptr<WritableFile>* result) {
  string translated_fname = TranslateName(fname);
  Status s;
  FILE* f = fopen(translated_fname.c_str(), "w");
  if (f == nullptr) {
    s = IOError(fname, errno);
  } else {
    result->reset(new PosixWritableFile(translated_fname, f));
  }
  return s;
}

Status PosixFileSystem::NewAppendableFile(
    const string& fname, std::unique_ptr<WritableFile>* result) {
  string translated_fname = TranslateName(fname);
  Status s;
  FILE* f = fopen(translated_fname.c_str(), "a");
  if (f == nullptr) {
    s = IOError(fname, errno);
  } else {
    result->reset(new PosixWritableFile(translated_fname, f));
  }
  return s;
}

Status PosixFileSystem::NewReadOnlyMemoryRegionFromFile(
    const string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result) {
  string translated_fname = TranslateName(fname);
  Status s = Status::OK();
  int fd = open(translated_fname.c_str(), O_RDONLY);
  if (fd < 0) {
    s = IOError(fname, errno);
  } else {
    struct stat st;
    ::fstat(fd, &st);
    const void* address =
        mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (address == MAP_FAILED) {
      s = IOError(fname, errno);
    } else {
      result->reset(new PosixReadOnlyMemoryRegion(address, st.st_size));
    }
    close(fd);
  }
  return s;
}

Status PosixFileSystem::FileExists(const string& fname) {
  if (access(TranslateName(fname).c_str(), F_OK) == 0) {
    return Status::OK();
  }
  return errors::NotFound(fname, " not found");
}

Status PosixFileSystem::GetChildren(const string& dir,
                                    std::vector<string>* result) {
  string translated_dir = TranslateName(dir);
  result->clear();
  DIR* d = opendir(translated_dir.c_str());
  if (d == nullptr) {
    return IOError(dir, errno);
  }
  struct dirent* entry;
  while ((entry = readdir(d)) != nullptr) {
    StringPiece basename = entry->d_name;
    if ((basename != ".") && (basename != "..")) {
      result->push_back(entry->d_name);
    }
  }
  closedir(d);
  return Status::OK();
}

Status PosixFileSystem::DeleteFile(const string& fname) {
  Status result;
  if (unlink(TranslateName(fname).c_str()) != 0) {
    result = IOError(fname, errno);
  }
  return result;
}

Status PosixFileSystem::CreateDir(const string& name) {
  Status result;
  if (mkdir(TranslateName(name).c_str(), 0755) != 0) {
    result = IOError(name, errno);
  }
  return result;
}

Status PosixFileSystem::DeleteDir(const string& name) {
  Status result;
  if (rmdir(TranslateName(name).c_str()) != 0) {
    result = IOError(name, errno);
  }
  return result;
}

Status PosixFileSystem::GetFileSize(const string& fname, uint64* size) {
  Status s;
  struct stat sbuf;
  if (stat(TranslateName(fname).c_str(), &sbuf) != 0) {
    *size = 0;
    s = IOError(fname, errno);
  } else {
    *size = sbuf.st_size;
  }
  return s;
}

Status PosixFileSystem::Stat(const string& fname, FileStatistics* stats) {
  Status s;
  struct stat sbuf;
  if (stat(TranslateName(fname).c_str(), &sbuf) != 0) {
    s = IOError(fname, errno);
  } else {
    stats->length = sbuf.st_size;
    stats->mtime_nsec = sbuf.st_mtime * 1e9;
    stats->is_directory = S_ISDIR(sbuf.st_mode);
  }
  return s;
}

Status PosixFileSystem::RenameFile(const string& src, const string& target) {
  Status result;
  if (rename(TranslateName(src).c_str(), TranslateName(target).c_str()) != 0) {
    result = IOError(src, errno);
  }
  return result;
}

}  // namespace tensorflow
