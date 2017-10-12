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

#include <sys/stat.h>
#include <algorithm>
#include <deque>

#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/gtl/stl_util.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/file_system.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/protobuf.h"

//DEBUG
#include <chrono>
//DEBUG


namespace tensorflow {

namespace {

constexpr int kNumThreads = 8;

// Run a function in parallel using a ThreadPool, but skip the ThreadPool
// on the iOS platform due to its problems with more than a few threads.
void ForEach(int first, int last, const std::function<void(int)>& f) {
#if TARGET_OS_IPHONE
  for (int i = first; i < last; i++) {
    f(i);
  }
#else
  int num_threads = std::min(kNumThreads, last - first);
  thread::ThreadPool threads(Env::Default(), "ForEach", num_threads);
  for (int i = first; i < last; i++) {
    threads.Schedule([f, i] { f(i); });
  }
#endif
}

}  // anonymous namespace

FileSystem::~FileSystem() {}

string FileSystem::TranslateName(const string& name) const {
  // If the name is empty, CleanPath returns "." which is incorrect and
  // we should return the empty path instead.
  if (name.empty()) return name;
  return io::CleanPath(name);
}

Status FileSystem::IsDirectory(const string& name) {
  // Check if path exists.
  TF_RETURN_IF_ERROR(FileExists(name));
  FileStatistics stat;
  TF_RETURN_IF_ERROR(Stat(name, &stat));
  if (stat.is_directory) {
    return Status::OK();
  }
  return Status(tensorflow::error::FAILED_PRECONDITION, "Not a directory");
}

RandomAccessFile::~RandomAccessFile() {}

WritableFile::~WritableFile() {}

FileSystemRegistry::~FileSystemRegistry() {}

bool FileSystem::FilesExist(const std::vector<string>& files,
                            std::vector<Status>* status) {
  bool result = true;
  for (const auto& file : files) {
    Status s = FileExists(file);
    result &= s.ok();
    if (status != nullptr) {
      status->push_back(s);
    } else if (!result) {
      // Return early since there is no need to check other files.
      return false;
    }
  }
  return result;
}

Status FileSystem::GetMatchingPaths(const string& pattern,
                                    std::vector<string>* results) {
  results->clear();
  // Find the fixed prefix by looking for the first wildcard.
  string fixed_prefix = pattern.substr(0, pattern.find_first_of("*?[\\"));
  string eval_pattern = pattern;
  std::vector<string> all_files;
  string dir = io::Dirname(fixed_prefix).ToString();
  // If dir is empty then we need to fix up fixed_prefix and eval_pattern to
  // include . as the top level directory.
  if (dir.empty()) {
    dir = ".";
    fixed_prefix = io::JoinPath(dir, fixed_prefix);
    eval_pattern = io::JoinPath(dir, pattern);
  }

  // Setup a BFS to explore everything under dir.
  std::deque<string> dir_q;
  dir_q.push_back(dir);
  Status ret;  // Status to return.
  // children_dir_status holds is_dir status for children. It can have three
  // possible values: OK for true; FAILED_PRECONDITION for false; CANCELLED
  // if we don't calculate IsDirectory (we might do that because there isn't
  // any point in exploring that child path).
  std::vector<Status> children_dir_status;
  while (!dir_q.empty()) {
    string current_dir = dir_q.front();
    dir_q.pop_front();
    std::vector<string> children;
    Status s = GetChildren(current_dir, &children);
    ret.Update(s);
    if (children.empty()) continue;
    // This IsDirectory call can be expensive for some FS. Parallelizing it.
    children_dir_status.resize(children.size());
    ForEach(0, children.size(), [this, &current_dir, &children, &fixed_prefix,
                                 &children_dir_status](int i) {
      const string child_path = io::JoinPath(current_dir, children[i]);
      // In case the child_path doesn't start with the fixed_prefix then
      // we don't need to explore this path.
      if (!StringPiece(child_path).starts_with(fixed_prefix)) {
        children_dir_status[i] =
            Status(tensorflow::error::CANCELLED, "Operation not needed");
      } else {
        children_dir_status[i] = IsDirectory(child_path);
      }
    });
    for (int i = 0; i < children.size(); ++i) {
      const string child_path = io::JoinPath(current_dir, children[i]);
      // If the IsDirectory call was cancelled we bail.
      if (children_dir_status[i].code() == tensorflow::error::CANCELLED) {
        continue;
      }
      // If the child is a directory add it to the queue.
      if (children_dir_status[i].ok()) {
        dir_q.push_back(child_path);
      }
      all_files.push_back(child_path);
    }
  }

  // Match all obtained files to the input pattern.
  for (const auto& f : all_files) {
    if (Env::Default()->MatchPath(f, eval_pattern)) {
      results->push_back(f);
    }
  }
  return ret;
}

Status FileSystem::DeleteRecursively(const string& dirname,
                                     int64* undeleted_files,
                                     int64* undeleted_dirs) {
  CHECK_NOTNULL(undeleted_files);
  CHECK_NOTNULL(undeleted_dirs);

  *undeleted_files = 0;
  *undeleted_dirs = 0;
  // Make sure that dirname exists;
  Status exists_status = FileExists(dirname);
  if (!exists_status.ok()) {
    (*undeleted_dirs)++;
    return exists_status;
  }
  std::deque<string> dir_q;      // Queue for the BFS
  std::vector<string> dir_list;  // List of all dirs discovered
  dir_q.push_back(dirname);
  Status ret;  // Status to be returned.
  // Do a BFS on the directory to discover all the sub-directories. Remove all
  // children that are files along the way. Then cleanup and remove the
  // directories in reverse order.;
  while (!dir_q.empty()) {
    string dir = dir_q.front();
    dir_q.pop_front();
    dir_list.push_back(dir);
    std::vector<string> children;
    // GetChildren might fail if we don't have appropriate permissions.
    Status s = GetChildren(dir, &children);
    ret.Update(s);
    if (!s.ok()) {
      (*undeleted_dirs)++;
      continue;
    }
    for (const string& child : children) {
      const string child_path = io::JoinPath(dir, child);
      // If the child is a directory add it to the queue, otherwise delete it.
      if (IsDirectory(child_path).ok()) {
        dir_q.push_back(child_path);
      } else {
        // Delete file might fail because of permissions issues or might be
        // unimplemented.
        Status del_status = DeleteFile(child_path);
        ret.Update(del_status);
        if (!del_status.ok()) {
          (*undeleted_files)++;
        }
      }
    }
  }
  // Now reverse the list of directories and delete them. The BFS ensures that
  // we can delete the directories in this order.
  std::reverse(dir_list.begin(), dir_list.end());
  for (const string& dir : dir_list) {
    // Delete dir might fail because of permissions issues or might be
    // unimplemented.
    Status s = DeleteDir(dir);
    ret.Update(s);
    if (!s.ok()) {
      (*undeleted_dirs)++;
    }
  }
  return ret;
}

Status FileSystem::RecursivelyCreateDir(const string& dirname) {
  StringPiece scheme, host, remaining_dir;
  io::ParseURI(dirname, &scheme, &host, &remaining_dir);
  std::vector<StringPiece> sub_dirs;
  while (!remaining_dir.empty()) {
    Status status = FileExists(io::CreateURI(scheme, host, remaining_dir));
    if (status.ok()) {
      break;
    }
    if (status.code() != error::Code::NOT_FOUND) {
      return status;
    }
    // Basename returns "" for / ending dirs.
    if (!remaining_dir.ends_with("/")) {
      sub_dirs.push_back(io::Basename(remaining_dir));
    }
    remaining_dir = io::Dirname(remaining_dir);
  }

  // sub_dirs contains all the dirs to be created but in reverse order.
  std::reverse(sub_dirs.begin(), sub_dirs.end());

  // Now create the directories.
  string built_path = remaining_dir.ToString();
  for (const StringPiece sub_dir : sub_dirs) {
    built_path = io::JoinPath(built_path, sub_dir);
    Status status = CreateDir(io::CreateURI(scheme, host, built_path));
    if (!status.ok() && status.code() != tensorflow::error::ALREADY_EXISTS) {
      return status;
    }
  }
  return Status::OK();
}


#ifdef TENSORFLOW_USE_HDF5
// HDF5 specific stuff
Status HDF5File::hdf5_check_file_exists(const string& fname) const {
  Status s;
  
  hid_t hdf5_status = H5Fis_hdf5(fname.c_str());
  if(hdf5_status < 0) s = Status(error::NOT_FOUND, strings::StrCat("file does not exist; ", static_cast<int>(hdf5_status)));
  else if(hdf5_status == 0) s = Status(error::INVALID_ARGUMENT,"file is not an hdf5-file");
  else s = Status::OK();
  
  return s;
}  


Status HDF5File::hdf5_check_dataset_exists(const string& dname) const{
  Status s;
  
  //check if object exists and has right type
  hid_t hdf5_status = H5Oexists_by_name( hdf5_fd_, dname.c_str(), H5P_DEFAULT );
  if(hdf5_status<=0){
    s = Status(error::NOT_FOUND, strings::StrCat("object "+dname+" does not exist; ", static_cast<int>(hdf5_status)));
  }
  else{
    H5O_info_t object_info;
    H5Oget_info_by_name( hdf5_fd_, dname.c_str(), &object_info, H5P_DEFAULT );
    if(object_info.type!=H5O_TYPE_DATASET){
      s = Status(error::INVALID_ARGUMENT, strings::StrCat("dataset: "+dname+" not a dataset; ", static_cast<int>(object_info.type)));
    }
    else{
      s = Status::OK();
    }
  }
  return s;
}


//encodes the record the following way: 
//buff = type_buff_size|type_buff(buff_size)|num_dims|dims(num_dims)|data(prod_i dims(i))
// all size types are hsize_t.
string HDF5File::EncodeBinary(const DatasetInfo* info) const{
  //datatype size of buffer
  hsize_t tsize = static_cast<hsize_t>(info->type_enc.size());
  StringPiece ntype_piece = StringPiece( reinterpret_cast<const char*>(&tsize), sizeof(tsize) );
  //datatype
  StringPiece type_piece = StringPiece( info->type_enc.c_str(), tsize );
  //number of dimensions 
  StringPiece ndims_piece = StringPiece( reinterpret_cast<const char*>(&info->ndims), sizeof(info->ndims) );
  //mode sizes
  StringPiece dims_piece = StringPiece( reinterpret_cast<const char*>(&info->dims[1]), (info->ndims)*sizeof(hsize_t) );
  //data
  StringPiece data_piece = StringPiece( info->buff, (info->type_size)*(info->dset_size) );
    
  //concatenate everything
  return strings::StrCat( ntype_piece, type_piece, ndims_piece, dims_piece, data_piece );
}


HDF5File::HDF5File(const string& fname, const hid_t& fapl_id) : filename_(fname), fapl_id_(fapl_id) {
  //default property list for data access
  dapl_id_ = H5Pcreate(H5P_DATASET_XFER);
  
  //do some checks:
  Status s = hdf5_check_file_exists(fname);
  if( s == Status::OK() ){
    hdf5_fd_ = H5Fopen(fname.c_str(), H5F_ACC_RDONLY, fapl_id_);
  }
}
  
  
HDF5File::~HDF5File() { 
  //close the objects I still have a handle on:
  std::map<string, DatasetInfo>::iterator it;
  for(it=dsetinfo.begin(); it!=dsetinfo.end(); ++it){
    //free buffers
    delete [] it->second.buff;
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
  
  //close porperty lists
  H5Pclose(dapl_id_);
  H5Pclose(fapl_id_);
  
  //close the file finally
  H5Fclose(hdf5_fd_);
}


Status HDF5File::InitDataset(const string& dset){
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
  //in here we store the number of dims - 1 (first dim is batched) for the string piece:
  info.ndims = info.dims.size()-1;
  
  //get datatype, only numeric types supported so far:
  info.type = H5Tget_native_type(H5Dget_type(info.id), H5T_DIR_DESCEND);
  //encode the type into a string which will be prepended to the output
  size_t tencsize = 0;
  H5Tencode(info.type, NULL, &tencsize);
  unsigned char* tmpstr = new unsigned char[tencsize];
  H5Tencode(info.type, tmpstr, &tencsize);
  info.type_enc = string(reinterpret_cast<char*>(tmpstr), tencsize);
  delete [] tmpstr;
  
  //determine size and allocate buffers
  hsize_t buff_size = 1;
  for(unsigned int d=1; d<info.dims.size(); d++) buff_size *= info.dims[d];
  info.dset_size = buff_size;
  info.type_size = H5Tget_size(info.type);
  
  //allocate buffers
  info.buff = new char[info.dset_size*info.type_size];
  
  //add to hashmap
  dsetinfo[dset] = info;
  
  //close objects:
  H5Sclose(mem_id);
  
  return s;
}


Status HDF5File::Read(const string& dset, const size_t& row_num, string* result) const {
  Status s;
    
  //if dataset not initialized, do it now:
  if(dsetinfo.find( dset ) == dsetinfo.end()){
    s = Status(error::FAILED_PRECONDITION,"you need to initialize a dataset first using the InitDataset(<name>) member function before reading from it");
    return s;
  }
  if(s != Status::OK()) return s;
  
  //get size needed for buffer
  const DatasetInfo* info = &(dsetinfo.at(dset));
    
  //check if we are already wrapping around:
  if(row_num>=info->dims[0]) return Status(error::OUT_OF_RANGE,"row number specified bigger than the 0-dimension of the dataset");

  //set hyperslab parameters
  std::vector<hsize_t> start(info->dims.size()), count(info->dims.size());
  //read one row only
  start[0] = static_cast<hsize_t>(row_num);
  count[0] = 1;
  //read the full thing for the rest
  for(unsigned int d=1; d<info->dims.size(); ++d){
    start[d] = 0;
    count[d] = info->dims[d];
  }
    
  //no checks here as those were already performed before
  hid_t file_space = H5Dget_space(info->id);
  H5Sselect_hyperslab(file_space, H5S_SELECT_SET, start.data(), NULL, count.data(), NULL);
  hid_t mem_space = H5Screate_simple(static_cast<hsize_t>(info->ndims), &(info->dims[1]), NULL);

  //read from the slab
  if( H5Tequal(info->type, H5T_NATIVE_FLOAT) ){
    H5Dread(info->id, info->type, mem_space, file_space, dapl_id_, reinterpret_cast<float*>(info->buff));
  }
  else if( H5Tequal(info->type, H5T_NATIVE_INT) ){
    H5Dread(info->id, info->type, mem_space, file_space, dapl_id_, reinterpret_cast<int*>(info->buff));
  }
  else if( H5Tequal(info->type, H5T_NATIVE_LONG) ){
    H5Dread(info->id, info->type, mem_space, file_space, dapl_id_, reinterpret_cast<long*>(info->buff));
  }
  else if( H5Tequal(info->type, H5T_NATIVE_DOUBLE) ){
    H5Dread(info->id, info->type, mem_space, file_space, dapl_id_, reinterpret_cast<double*>(info->buff));
  }
  else{
    s = Status(error::INVALID_ARGUMENT," error, datatype currently not supported.");
  }

  //close the spaces:
  H5Sclose(file_space);
  H5Sclose(mem_space);

  //encode binary
  *result = EncodeBinary(info);
  
  //status is OK
  s = Status::OK();
  return s;
}
#endif


}  // namespace tensorflow
