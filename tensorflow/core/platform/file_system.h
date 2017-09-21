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

#ifndef TENSORFLOW_CORE_PLATFORM_FILE_SYSTEM_H_
#define TENSORFLOW_CORE_PLATFORM_FILE_SYSTEM_H_

#include <stdint.h>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/file_statistics.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

#ifdef TENSORFLOW_USE_HDF5
//hdf5-specific stuff
#include "third_party/hdf5/hdf5.h"
#endif

#ifdef PLATFORM_WINDOWS
#undef DeleteFile
#endif

namespace tensorflow {

class HDF5File;
class RandomAccessFile;
class ReadOnlyMemoryRegion;
class WritableFile;

/// A generic interface for accessing a file system.  Implementations
/// of custom filesystem adapters must implement this interface,
/// RandomAccessFile, WritableFile, and ReadOnlyMemoryRegion classes.
class FileSystem {
 public:
  /// \brief Creates a brand new random access read-only file with the
  /// specified name.
  ///
  /// On success, stores a pointer to the new file in
  /// *result and returns OK.  On failure stores NULL in *result and
  /// returns non-OK.  If the file does not exist, returns a non-OK
  /// status.
  ///
  /// The returned file may be concurrently accessed by multiple threads.
  ///
  /// The ownership of the returned RandomAccessFile is passed to the caller
  /// and the object should be deleted when is not used.
  virtual Status NewRandomAccessFile(
      const string& fname, std::unique_ptr<RandomAccessFile>* result) = 0;
  
#ifdef TENSORFLOW_USE_HDF5
  /// \brief Creates a brand new structured access HDF5 read-only file with the
  /// specified name.
  ///
  /// On success, stores a FileSystem-dependent access control list 
  /// and the filename to the new file in *result and returns OK.
  /// On failure stores NULL in *result and returns non-OK.
  /// If the file does not exist, returns a non-OK status.
  ///
  /// The returned file may be concurrently accessed by multiple threads.
  ///
  /// The ownership of the returned HDF5File is passed to the caller
  /// and the object should be deleted when is not used. 
  virtual Status NewHDF5File(
      const string& fname, std::unique_ptr<HDF5File>* result) = 0;
#endif

  /// \brief Creates an object that writes to a new file with the specified
  /// name.
  ///
  /// Deletes any existing file with the same name and creates a
  /// new file.  On success, stores a pointer to the new file in
  /// *result and returns OK.  On failure stores NULL in *result and
  /// returns non-OK.
  ///
  /// The returned file will only be accessed by one thread at a time.
  ///
  /// The ownership of the returned WritableFile is passed to the caller
  /// and the object should be deleted when is not used.
  virtual Status NewWritableFile(const string& fname,
                                 std::unique_ptr<WritableFile>* result) = 0;

  /// \brief Creates an object that either appends to an existing file, or
  /// writes to a new file (if the file does not exist to begin with).
  ///
  /// On success, stores a pointer to the new file in *result and
  /// returns OK.  On failure stores NULL in *result and returns
  /// non-OK.
  ///
  /// The returned file will only be accessed by one thread at a time.
  ///
  /// The ownership of the returned WritableFile is passed to the caller
  /// and the object should be deleted when is not used.
  virtual Status NewAppendableFile(const string& fname,
                                   std::unique_ptr<WritableFile>* result) = 0;

  /// \brief Creates a readonly region of memory with the file context.
  ///
  /// On success, it returns a pointer to read-only memory region
  /// from the content of file fname. The ownership of the region is passed to
  /// the caller. On failure stores nullptr in *result and returns non-OK.
  ///
  /// The returned memory region can be accessed from many threads in parallel.
  ///
  /// The ownership of the returned ReadOnlyMemoryRegion is passed to the caller
  /// and the object should be deleted when is not used.
  virtual Status NewReadOnlyMemoryRegionFromFile(
      const string& fname, std::unique_ptr<ReadOnlyMemoryRegion>* result) = 0;

  /// Returns OK if the named path exists and NOT_FOUND otherwise.
  virtual Status FileExists(const string& fname) = 0;

  /// Returns true if all the listed files exist, false otherwise.
  /// if status is not null, populate the vector with a detailed status
  /// for each file.
  virtual bool FilesExist(const std::vector<string>& files,
                          std::vector<Status>* status);

  /// \brief Returns the immediate children in the given directory.
  ///
  /// The returned paths are relative to 'dir'.
  virtual Status GetChildren(const string& dir,
                             std::vector<string>* result) = 0;

  /// \brief Given a pattern, stores in *results the set of paths that matches
  /// that pattern. *results is cleared.
  ///
  /// pattern must match all of a name, not just a substring.
  ///
  /// pattern: { term }
  /// term:
  ///   '*': matches any sequence of non-'/' characters
  ///   '?': matches a single non-'/' character
  ///   '[' [ '^' ] { match-list } ']':
  ///        matches any single character (not) on the list
  ///   c: matches character c (c != '*', '?', '\\', '[')
  ///   '\\' c: matches character c
  /// character-range:
  ///   c: matches character c (c != '\\', '-', ']')
  ///   '\\' c: matches character c
  ///   lo '-' hi: matches character c for lo <= c <= hi
  ///
  /// Typical return codes:
  ///  * OK - no errors
  ///  * UNIMPLEMENTED - Some underlying functions (like GetChildren) are not
  ///                    implemented
  /// The default implementation uses a combination of GetChildren, MatchPath
  /// and IsDirectory.
  virtual Status GetMatchingPaths(const string& pattern,
                                  std::vector<string>* results);

  /// \brief Obtains statistics for the given path.
  virtual Status Stat(const string& fname, FileStatistics* stat) = 0;

  /// \brief Deletes the named file.
  virtual Status DeleteFile(const string& fname) = 0;

  /// \brief Creates the specified directory.
  /// Typical return codes:
  ///  * OK - successfully created the directory.
  ///  * ALREADY_EXISTS - directory with name dirname already exists.
  ///  * PERMISSION_DENIED - dirname is not writable.
  virtual Status CreateDir(const string& dirname) = 0;

  /// \brief Creates the specified directory and all the necessary
  /// subdirectories.
  /// Typical return codes:
  ///  * OK - successfully created the directory and sub directories, even if
  ///         they were already created.
  ///  * PERMISSION_DENIED - dirname or some subdirectory is not writable.
  virtual Status RecursivelyCreateDir(const string& dirname);

  /// \brief Deletes the specified directory.
  virtual Status DeleteDir(const string& dirname) = 0;

  /// \brief Deletes the specified directory and all subdirectories and files
  /// underneath it. undeleted_files and undeleted_dirs stores the number of
  /// files and directories that weren't deleted (unspecified if the return
  /// status is not OK).
  /// REQUIRES: undeleted_files, undeleted_dirs to be not null.
  /// Typical return codes:
  ///  * OK - dirname exists and we were able to delete everything underneath.
  ///  * NOT_FOUND - dirname doesn't exist
  ///  * PERMISSION_DENIED - dirname or some descendant is not writable
  ///  * UNIMPLEMENTED - Some underlying functions (like Delete) are not
  ///                    implemented
  virtual Status DeleteRecursively(const string& dirname,
                                   int64* undeleted_files,
                                   int64* undeleted_dirs);

  /// \brief Stores the size of `fname` in `*file_size`.
  virtual Status GetFileSize(const string& fname, uint64* file_size) = 0;

  /// \brief Overwrites the target if it exists.
  virtual Status RenameFile(const string& src, const string& target) = 0;

  /// \brief Translate an URI to a filename for the FileSystem implementation.
  ///
  /// The implementation in this class cleans up the path, removing
  /// duplicate /'s, resolving .. and . (more details in
  /// tensorflow::lib::io::CleanPath).
  virtual string TranslateName(const string& name) const;

  /// \brief Returns whether the given path is a directory or not.
  ///
  /// Typical return codes (not guaranteed exhaustive):
  ///  * OK - The path exists and is a directory.
  ///  * FAILED_PRECONDITION - The path exists and is not a directory.
  ///  * NOT_FOUND - The path entry does not exist.
  ///  * PERMISSION_DENIED - Insufficient permissions.
  ///  * UNIMPLEMENTED - The file factory doesn't support directories.
  virtual Status IsDirectory(const string& fname);

  FileSystem() {}

  virtual ~FileSystem();
};

/// A file abstraction for randomly reading the contents of a file.
class RandomAccessFile {
 public:
  RandomAccessFile() {}
  virtual ~RandomAccessFile();

  /// \brief Reads up to `n` bytes from the file starting at `offset`.
  ///
  /// `scratch[0..n-1]` may be written by this routine.  Sets `*result`
  /// to the data that was read (including if fewer than `n` bytes were
  /// successfully read).  May set `*result` to point at data in
  /// `scratch[0..n-1]`, so `scratch[0..n-1]` must be live when
  /// `*result` is used.
  ///
  /// On OK returned status: `n` bytes have been stored in `*result`.
  /// On non-OK returned status: `[0..n]` bytes have been stored in `*result`.
  ///
  /// Returns `OUT_OF_RANGE` if fewer than n bytes were stored in `*result`
  /// because of EOF.
  ///
  /// Safe for concurrent use by multiple threads.
  virtual Status Read(uint64 offset, size_t n, StringPiece* result,
                      char* scratch) const = 0;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(RandomAccessFile);
};


#ifdef TENSORFLOW_USE_HDF5
/// HDF5 can deal with different 
class HDF5File {
 public:
  HDF5File() {}
  HDF5File(const string& fname, const hid_t& fapl_id);
  ~HDF5File();
  
  /// \brief The read function takes a dataset name to read from as well as
  /// a row_number to specify the corresponding row to read.
  ///
  /// Safe for concurrent use by multiple threads.
  Status Read(const string& dset, const size_t& row_num, StringPiece* result) const;
  
  /// \brief This function prepares internal buffers and performs shape and datatype checks on
  /// the specified dataset. You need to initialize a dataset before reading from it.
  ///
  /// Safe for concurrent use by multiple threads.
  Status InitDataset(const string& dset);

 private:
  //variables
  string filename_;
  hid_t hdf5_fd_;
  //access property lists
  hid_t dapl_id_, fapl_id_;
  
  struct DatasetInfo{
    std::vector<hsize_t> dims;
    hid_t id;
    hid_t type;
  };
  std::map<string, DatasetInfo> dsetinfo;
  
  //functions
  Status hdf5_check_file_exists(const string& fname) const;
  Status hdf5_check_dataset_exists(const string& dname) const;
  inline string EncodeTokenASCII(char* buff, const hid_t& type_id) const;
  string EncodeASCII(const DatasetInfo& info, const hsize_t& buff_size, char* buff) const;
  
  TF_DISALLOW_COPY_AND_ASSIGN(HDF5File);
};
#endif


/// \brief A file abstraction for sequential writing.
///
/// The implementation must provide buffering since callers may append
/// small fragments at a time to the file.
class WritableFile {
 public:
  WritableFile() {}
  virtual ~WritableFile();

  /// \brief Append 'data' to the file.
  virtual Status Append(const StringPiece& data) = 0;

  /// \brief Close the file.
  ///
  /// Flush() and de-allocate resources associated with this file
  ///
  /// Typical return codes (not guaranteed to be exhaustive):
  ///  * OK
  ///  * Other codes, as returned from Flush()
  virtual Status Close() = 0;

  /// \brief Flushes the file and optionally syncs contents to filesystem.
  ///
  /// This should flush any local buffers whose contents have not been
  /// delivered to the filesystem.
  ///
  /// If the process terminates after a successful flush, the contents
  /// may still be persisted, since the underlying filesystem may
  /// eventually flush the contents.  If the OS or machine crashes
  /// after a successful flush, the contents may or may not be
  /// persisted, depending on the implementation.
  virtual Status Flush() = 0;

  /// \brief Syncs contents of file to filesystem.
  ///
  /// This waits for confirmation from the filesystem that the contents
  /// of the file have been persisted to the filesystem; if the OS
  /// or machine crashes after a successful Sync, the contents should
  /// be properly saved.
  virtual Status Sync() = 0;

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(WritableFile);
};

/// \brief A readonly memmapped file abstraction.
///
/// The implementation must guarantee that all memory is accessible when the
/// object exists, independently from the Env that created it.
class ReadOnlyMemoryRegion {
 public:
  ReadOnlyMemoryRegion() {}
  virtual ~ReadOnlyMemoryRegion() = default;

  /// \brief Returns a pointer to the memory region.
  virtual const void* data() = 0;

  /// \brief Returns the length of the memory region in bytes.
  virtual uint64 length() = 0;
};

// START_SKIP_DOXYGEN

#ifndef SWIG
// Degenerate file system that provides no implementations.
class NullFileSystem : public FileSystem {
 public:
  NullFileSystem() {}

  ~NullFileSystem() override = default;

  Status NewRandomAccessFile(
      const string& fname, std::unique_ptr<RandomAccessFile>* result) override {
    return errors::Unimplemented("NewRandomAccessFile unimplemented");
  }
  
#ifdef TENSORFLOW_USE_HDF5
  Status NewHDF5File(
      const string& fname, std::unique_ptr<HDF5File>* result) override {
    return errors::Unimplemented("NewHDF5File unimplemented");
  }
#endif

  Status NewWritableFile(const string& fname,
                         std::unique_ptr<WritableFile>* result) override {
    return errors::Unimplemented("NewWritableFile unimplemented");
  }

  Status NewAppendableFile(const string& fname,
                           std::unique_ptr<WritableFile>* result) override {
    return errors::Unimplemented("NewAppendableFile unimplemented");
  }

  Status NewReadOnlyMemoryRegionFromFile(
      const string& fname,
      std::unique_ptr<ReadOnlyMemoryRegion>* result) override {
    return errors::Unimplemented(
        "NewReadOnlyMemoryRegionFromFile unimplemented");
  }

  Status FileExists(const string& fname) override {
    return errors::Unimplemented("FileExists unimplemented");
  }

  Status GetChildren(const string& dir, std::vector<string>* result) override {
    return errors::Unimplemented("GetChildren unimplemented");
  }

  Status DeleteFile(const string& fname) override {
    return errors::Unimplemented("DeleteFile unimplemented");
  }

  Status CreateDir(const string& dirname) override {
    return errors::Unimplemented("CreateDir unimplemented");
  }

  Status DeleteDir(const string& dirname) override {
    return errors::Unimplemented("DeleteDir unimplemented");
  }

  Status GetFileSize(const string& fname, uint64* file_size) override {
    return errors::Unimplemented("GetFileSize unimplemented");
  }

  Status RenameFile(const string& src, const string& target) override {
    return errors::Unimplemented("RenameFile unimplemented");
  }

  Status Stat(const string& fname, FileStatistics* stat) override {
    return errors::Unimplemented("Stat unimplemented");
  }
};
#endif

// END_SKIP_DOXYGEN

/// \brief A registry for file system implementations.
///
/// Filenames are specified as an URI, which is of the form
/// [scheme://]<filename>.
/// File system implementations are registered using the REGISTER_FILE_SYSTEM
/// macro, providing the 'scheme' as the key.
class FileSystemRegistry {
 public:
  typedef std::function<FileSystem*()> Factory;

  virtual ~FileSystemRegistry();
  virtual Status Register(const string& scheme, Factory factory) = 0;
  virtual FileSystem* Lookup(const string& scheme) = 0;
  virtual Status GetRegisteredFileSystemSchemes(
      std::vector<string>* schemes) = 0;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_FILE_SYSTEM_H_
