// Copyright (c) 2014-2015 The AsyncFTRL Project
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef SRC_FILE_PARSER_H
#define SRC_FILE_PARSER_H

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>
#include <string>
#include <fstream>
#include <zlib.h>
#include "hdfs.h"
#include "src/lock.h"
#include <climits>
const size_t lower_bound = 1000;
const size_t upper_bound = 520000000;
const int buf_in_out_expand = 1;

template<typename T>
class FileParserBase {
public:
	FileParserBase() {}
	virtual ~FileParserBase() {}

public:
	virtual bool OpenFile(const char* path) = 0;
	virtual bool OpenNextFile() = 0;
	virtual bool CloseFile() = 0;

	virtual bool ReadSample(T& y, std::vector<std::pair<size_t, T> >& x) = 0;
	virtual bool ReadSample(int& click,int& impre, std::vector<std::pair<size_t, T> >& x) = 0;
	virtual bool ReadSampleMultiThread(T& y, std::vector<std::pair<size_t, T> >& x) = 0;
	virtual bool ReadSampleMultiThread(int& click,int& impre, std::vector<std::pair<size_t, T> >& x) = 0;

public:
	static bool FileExists(const char* path);
};

// FileParser: parse training file with LIBSVM-format
template<typename T>
class FileParser : public FileParserBase<T> {
public:
	FileParser();
	virtual ~FileParser();

	virtual bool OpenFile(const char* path);
	virtual bool OpenNextFile();
	virtual bool CloseFile();

	FILE* HdfsOpen(const char* path);

	// Read a new line and Parse to <x, y>, thread-safe but not optimized for multi-threading
	virtual bool ReadSample(T& y, std::vector<std::pair<size_t, T> >& x);
	virtual bool ReadSample(int& click,int& impre, std::vector<std::pair<size_t, T> >& x);

	// Read a new line and Parse to <x, y>, with multi-threading capability
	virtual bool ReadSampleMultiThread(T& y, std::vector<std::pair<size_t, T> >& x);
	virtual bool ReadSampleMultiThread(int& click,int& impre, std::vector<std::pair<size_t, T> >& x) ;

	bool ParseSample(char* buf, T& y,
		std::vector<std::pair<size_t, T> >& x);

    //new function to parse news feature file format
	bool ParseSample(char* buf, int& click,int& impre,
		std::vector<std::pair<size_t, T> >& x);

	// Read a new line using external buffer
	char* ReadLine(char *buf, size_t& buf_size);

private:
	// Read a new line using internal buffer and copy that to allocated new memory
	char* ReadLine();

	char* (FileParser::*ReadLineImpl)(char *buf, size_t& buf_size);
	char* gz_ReadLineImpl(char *buf, size_t& buf_size);
	char* uz_ReadLineImpl(char *buf, size_t& buf_size);

private:
	enum { kDefaultBufSize = 40240,kFileBufSize = 10000000} buf_enum;

	FILE *list_file_desc_;
	char *list_buf_;
	size_t list_buf_size_;

	FILE *file_desc_;
	gzFile gz_file_desc_;
	char* buf_;
	size_t buf_size_;

	//read from hdfs to f_in_buf_, uncompress to f_out_buf
	unsigned char* f_in_buf_;
	unsigned char* f_out_buf_;
	size_t f_in_buf_size_;
	size_t f_out_buf_size_;

	SpinLock lock_;
};


template<typename T>
T* fp_alloc_func(size_t size) {
	void* ptr = malloc(size * sizeof(T));
	return reinterpret_cast<T*>(ptr);
}

template<typename T>
T* fp_realloc_func(T* buf, size_t size) {
	void* ptr = realloc(reinterpret_cast<void*>(buf), size * sizeof(T));
	return reinterpret_cast<T*>(ptr);
}


template<typename T>
bool FileParserBase<T>::FileExists(const char* path) {
	FILE *fp = fopen(path, "r");
	if (fp) {
		fclose(fp);
		return true;
	}

	return false;
}




template<typename T>
FileParser<T>::FileParser() : list_file_desc_(NULL), list_buf_(NULL), list_buf_size_(0), file_desc_(NULL), buf_(NULL), buf_size_(0), gz_file_desc_(NULL) {
	list_buf_size_ = kDefaultBufSize;
	list_buf_ = fp_alloc_func<char>(list_buf_size_);
	
	buf_size_ = kDefaultBufSize;
	buf_ = fp_alloc_func<char>(buf_size_);

	f_in_buf_size_ = kFileBufSize;
	f_out_buf_size_ = kFileBufSize * buf_in_out_expand;
	f_in_buf_ = fp_alloc_func<unsigned char>(f_in_buf_size_); 
	f_out_buf_ = fp_alloc_func<unsigned char>(f_out_buf_size_); 
}

template<typename T>
FileParser<T>::~FileParser() {
	if (list_file_desc_)
	{
		fclose(list_file_desc_);
		list_file_desc_ = NULL;
	}

	if (file_desc_)
	{
		fclose(file_desc_);
		file_desc_ = NULL;
	}

	if (gz_file_desc_) {
		gzclose(gz_file_desc_);
		gz_file_desc_ = NULL;
	}

	if (list_buf_)
	{
		free(list_buf_);
		list_buf_ = NULL;
	}
	list_buf_size_ = 0;

	if (buf_) {
		free(buf_);
		buf_ = NULL;
	}
	if (f_in_buf_) {
		free(f_in_buf_);
		f_in_buf_ = NULL;
	}

	if (f_out_buf_) {
		free(f_out_buf_);
		f_out_buf_ = NULL;
	}

	buf_size_ = 0;
}

template<typename T>
bool FileParser<T>::OpenFile(const char* path) {
	list_file_desc_ = fopen(path, "r");
	if (!list_file_desc_)
	{
		fprintf(stderr,"OpenFile(): open filelist %s failed!\n", path);
		return false;
	}
	fprintf(stderr,"OpenFile(): open filelist %s success!\n", path);

	if (fgets(list_buf_, list_buf_size_-1, list_file_desc_) == NULL)
	{
		fprintf(stderr,"OpenFile(): get first file from %s failed!\n", path);
		return false;
	}

	int file_name_len = strlen(list_buf_);
	list_buf_[file_name_len-1] = '\0';
	fprintf(stderr,"OpenFile(): get first file %s from %s success!\n", list_buf_, path);

	if (memcmp(list_buf_, "hdfs", 4) == 0)
	{
		//file_desc_ = fopen(list_buf_, "r");
		if (NULL != file_desc_){
			fclose(file_desc_);
			file_desc_ = NULL;
		}
		file_desc_ = HdfsOpen(list_buf_);
		if (!file_desc_)
		{
			fprintf(stderr,"OpenFile(): open first file %s failed!\n", list_buf_);
			return false;
		}
		ReadLineImpl = &FileParser::uz_ReadLineImpl;
		fprintf(stderr,"OpenFile(): open first file %s success!\n", list_buf_);
	}
	else
	{
		gz_file_desc_ = gzopen(list_buf_, "r");
		if (!gz_file_desc_) 
		{
			fprintf(stderr,"OpenFile(): open first file %s failed!\n", list_buf_);
			return false;
		}
		ReadLineImpl = &FileParser::gz_ReadLineImpl;
		fprintf(stderr,"OpenFile(): open first file %s success!\n", list_buf_);
	}
	return true;
}

template<typename T>
bool FileParser<T>::OpenNextFile() 
{
	if (!list_file_desc_)
	{
		fprintf(stderr,"OpenNextFile(): list_file_desc_ == NULL!\n");
		return false;
	}

	if (fgets(list_buf_, list_buf_size_-1, list_file_desc_) == NULL)
	{
		fprintf(stderr,"OpenNextFile(): get next filename failed!\n");
		return false;
	}

	/*
	if (gz_file_desc_)
	{
		gzclose(gz_file_desc_);
		gz_file_desc_ = NULL;
	}
	*/
	if (file_desc_)
	{
		fclose(file_desc_);
		file_desc_ = NULL;
	}

	int file_name_len = strlen(list_buf_);
	list_buf_[file_name_len-1] = '\0';
	
	if (memcmp(list_buf_, "hdfs", 4) == 0)
	{
		if (NULL != file_desc_){
			fclose(file_desc_);
			file_desc_ = NULL;
		}

		file_desc_ = HdfsOpen(list_buf_);
		if (!file_desc_) 
		{
			fprintf(stderr,"OpenNextFile(): open next file %s failed!\n", list_buf_);
			return false;
		}
		ReadLineImpl = &FileParser::uz_ReadLineImpl;
	}
	else
	{
		gz_file_desc_ = gzopen(list_buf_, "r");
		if (!gz_file_desc_) 
		{
			fprintf(stderr,"OpenNextFile(): open next file %s failed!\n", list_buf_);
			return false;
		}
		ReadLineImpl = &FileParser::gz_ReadLineImpl;
	}

	return true;
}

template<typename T>
bool FileParser<T>::CloseFile() 
{
	if (file_desc_)
	{
		fclose(file_desc_);
		file_desc_ = NULL;
	}

	if (gz_file_desc_) 
	{
		gzclose(gz_file_desc_);
		gz_file_desc_ = NULL;
	}

	if (list_file_desc_)
	{
		fclose(list_file_desc_);
		list_file_desc_ = NULL;
	}

	return true;
}

template<typename T>
FILE* FileParser<T>::HdfsOpen(const char * hdfs_path){
    if (NULL == hdfs_path) 
		return NULL;

	std::ifstream fin("./hdfs_conf");
	if ( fin.fail() ){
		fprintf(stderr,"please supply a file name hdfs_conf, the content is like hdfs://master1.vsvs.com \t 9000\n");
		exit(-1);
	}
	std::string ip,port;
	fin >> ip >> port; 
	fin.close();
	
    hdfsFS fs = hdfsConnect(ip.c_str(),atoi(port.c_str()));

    if (!fs) {
        fprintf(stderr, "Cannot connect to HDFS.\n");
        exit(-1);
    }

	hdfsFile inFile = hdfsOpenFile(fs, hdfs_path, O_RDONLY, 0, 0, 0);
    if (!inFile) {
        fprintf(stderr, "Failed to open %s for reading!\n", hdfs_path);
        exit(-2);
    }
     // Read from file.
    hdfsFileInfo * file_info = hdfsGetPathInfo(fs, hdfs_path);
	size_t file_bytes = file_info->mSize;

	fprintf(stderr, "hdfs read size  %ld !\n", file_bytes);
	//fprintf(stderr, "max out size  %ld !\n", file_bytes * buf_in_out_expand);
	f_in_buf_ = (unsigned char*)realloc(reinterpret_cast<void*>(f_in_buf_), sizeof(unsigned char) * file_bytes);
	memset ( (void *) f_in_buf_, 0, sizeof(unsigned char) * file_bytes);
	size_t readSize = 0;
	size_t tmp_readSize = 0;
	while(readSize < file_bytes){
		tmp_readSize = hdfsPread(fs, inFile,readSize, (void*)&(f_in_buf_[readSize]), 50000000);
		readSize += tmp_readSize;
	}
	//fprintf(stderr,"real read size %ld\n", readSize);


	unsigned long avo = INT_MAX ;
	f_out_buf_ = (unsigned char*)realloc(reinterpret_cast<void*>(f_out_buf_), avo);
	memset ( (void *) f_out_buf_, 0, avo);
	z_stream gzip_stream;

	gzip_stream.zalloc = (alloc_func)0;
	gzip_stream.zfree = (free_func)0;
	gzip_stream.opaque = (voidpf)0;

	unsigned remain = 0;
	if (file_bytes > UINT_MAX)
		remain = file_bytes - UINT_MAX;
	fprintf(stderr,"avail out %ld\n", avo);
	gzip_stream.next_in  = f_in_buf_ ;
	gzip_stream.avail_in = UINT_MAX;
	gzip_stream.next_out = f_out_buf_ ;
	gzip_stream.avail_out = avo;
	gzip_stream.total_out = 0;
	//fprintf(stderr,"current out %ld,avail out %ld,avail in %ld \n", gzip_stream.total_out,gzip_stream.avail_out,gzip_stream.avail_in);


auto zerror2str = [](int err) -> std::string {
	switch(err) {
		case Z_ERRNO:
			return "Z_ERRNO";
		case Z_STREAM_ERROR:
			return "Z_STREAM_ERROR";
		case Z_DATA_ERROR:
			return "Z_DATA_ERROR";
		case Z_MEM_ERROR:
			return "Z_MEM_ERROR";
		case Z_BUF_ERROR:
			return "Z_BUF_ERROR";
		case Z_VERSION_ERROR:
			return "Z_VERSION_ERROR";
		default:
			fprintf(stderr,"unkonwn error code %d\n",err);
			return "NOT AN ERROR";
		}
	};

	int ret = 0;

	ret = inflateInit2(&gzip_stream, 16 + MAX_WBITS);
	if (ret != Z_OK) {
		fprintf(stderr,"deflate init error\n");
	}   
	int k = 1;
	
	while(true) {
		ret = inflate(&gzip_stream, Z_SYNC_FLUSH);
		if (ret == Z_STREAM_END) {
			break;
		}
		else if (ret == Z_OK && gzip_stream.avail_out == 0){
			//fprintf(stderr,"current compressed %ld,avail out %ld \n", gzip_stream.total_out,gzip_stream.avail_out);
			f_out_buf_ = (unsigned char*)realloc(reinterpret_cast<void*>(f_out_buf_), sizeof(unsigned char) * avo * (k+1));
			gzip_stream.next_out = (unsigned char *)&(f_out_buf_[gzip_stream.total_out]);
			gzip_stream.avail_out =  sizeof(unsigned char) * avo * (k+1) - gzip_stream.total_out;
			//fprintf(stderr,"current out %ld,avail out %ld,avail in %ld \n", gzip_stream.total_out,gzip_stream.avail_out,gzip_stream.avail_in);
			k++;
		}
		else if (ret == Z_OK){
			//not finish
			unsigned start_indx = UINT_MAX - gzip_stream.avail_in;
			gzip_stream.next_in = &f_in_buf_[start_indx];
			gzip_stream.avail_in = remain + gzip_stream.avail_in;
			continue;
		}
		else{
			 fprintf(stderr,"error %s\n",zerror2str(ret).c_str());
			 break;
		}
	}

	ret = inflateEnd(&gzip_stream);

     
	size_t real_unc_size = gzip_stream.total_out;
	f_out_buf_ = (unsigned char*)realloc(reinterpret_cast<void*>(f_out_buf_), real_unc_size);
	fprintf(stderr, "hdfs uncompressed size  %ld !\n", real_unc_size);
	FILE *file_desc = fmemopen((void*)f_out_buf_,real_unc_size,"r");

    hdfsCloseFile(fs, inFile);
    //hdfsDisconnect(fs);

	return file_desc;
}

template<typename T>
char* FileParser<T>::gz_ReadLineImpl(char* buf, size_t& buf_size) 
{
	if (!gz_file_desc_) 
	{
		return NULL;
	}

	if (gzgets(gz_file_desc_, buf, buf_size-1) == NULL) 
	{
		if (OpenNextFile())
		{
			return (this->*ReadLineImpl)(buf, buf_size);
		}
		else
		{
			return NULL;
		}
	}

	while (strrchr(buf, '\n') == NULL) 
	{
		buf_size *= 2;
		buf = fp_realloc_func<char>(buf, buf_size);
		size_t len = strlen(buf);
		if (gzgets(gz_file_desc_, buf+len, buf_size-len-1) == NULL) break;
	}

	return buf;
}

template<typename T>
char* FileParser<T>::uz_ReadLineImpl(char* buf, size_t& buf_size) 
{
	if (!file_desc_) 
	{
		return NULL;
	}

	if (fgets(buf, buf_size-1,  file_desc_) == NULL) 
	{
		if (OpenNextFile())
		{
			return (this->*ReadLineImpl)(buf, buf_size);
		}
		else
		{
			return NULL;
		}
	}

	while (strrchr(buf, '\n') == NULL) 
	{
		buf_size *= 2;
		buf = fp_realloc_func<char>(buf, buf_size);
		size_t len = strlen(buf);
		if (fgets(buf+len, buf_size-len-1, file_desc_) == NULL) break;
	}

	return buf;
}

template<typename T>
char* FileParser<T>::ReadLine() {
	std::lock_guard<SpinLock> lock(lock_);

	char *buf = (this->*ReadLineImpl)(buf_, buf_size_);
	if (buf) {
		buf_ = buf;
		return strdup(buf);
	}

	return NULL;
}

template<typename T>
char* FileParser<T>::ReadLine(char *buf, size_t& buf_size) {
	std::lock_guard<SpinLock> lock(lock_);
	return (this->*ReadLineImpl)(buf, buf_size);
}

template<typename T>
T string_to_real(const char *nptr, char **endptr);

template<>
float string_to_real<float> (const char *nptr, char **endptr) {
	return strtof(nptr, endptr);
}

template<>
double string_to_real<double> (const char *nptr, char **endptr) {
	return strtod(nptr, endptr);
}

template<typename T>
bool FileParser<T>::ParseSample(char* buf, T& y,
		std::vector<std::pair<size_t, T> >& x) {
	if (buf == NULL) return false;

	char *endptr, *ptr;
	char *p = strtok_r(buf, " \t\n", &ptr);
	if (p == NULL) return false;

	y = string_to_real<T> (p, &endptr);
	if (endptr == p || *endptr != '\0') return false;
	if (y < 0) y = 0;

	x.clear();
	// add bias term
	x.push_back(std::make_pair((size_t)0, (T)1));
	while (1) {
		char *idx = strtok_r(NULL, ":", &ptr);
		char *val = strtok_r(NULL, " \t", &ptr);
		if (val == NULL) break;

		bool error_found = false;
		size_t k = (size_t) strtol(idx, &endptr, 10);
        k++;
		if (endptr == idx || *endptr != '\0' || static_cast<int>(k) < 0) {
			error_found = true;
		}

		T v = string_to_real<T> (val, &endptr);
		if (endptr == val || (*endptr != '\0' && !isspace(*endptr))) {
			error_found = true;
		}

		if (!error_found) {
			x.push_back(std::make_pair(k, v));
		}
	}

	return true;
}
template<typename T>
bool FileParser<T>::ParseSample(char* buf, int& click,int& impre,
		std::vector<std::pair<size_t, T> >& x) {
	if (buf == NULL) return false;

	char *endptr, *ptr;
	char *cl = strtok_r(buf, " \t", &ptr);
	if (cl == NULL) return false;
	click = string_to_real<T> (cl, &endptr);
	char *im = strtok_r(NULL, " \t\n", &ptr);
	if (im == NULL) return false;
	impre = string_to_real<T> (im, &endptr);
	 //fprintf(stderr,"click %d impre %d",click,impre);

	if (endptr == im || *endptr != '\0') return false;

	x.clear();
	// add bias term
	x.push_back(std::make_pair((size_t)0, (T)1));
	while (1) {
		char *idx = strtok_r(NULL, " \t", &ptr);
		char *val = strtok_r(NULL, " \t", &ptr);
		if (val == NULL) break;

		bool error_found = false;
		size_t k = (size_t) strtol(idx, &endptr, 10);
        //indx +1 to avoid conflict with bias at pos 0
        k++;


	if (endptr == idx || *endptr != '\0' || static_cast<int>(k) < 0) {
		error_found = true;
	}

	T v = string_to_real<T> (val, &endptr);
	if (endptr == val || (*endptr != '\0' && !isspace(*endptr))) {
		error_found = true;
	}
	if (!error_found) {
		//if (k > lower_bound)
			x.push_back(std::make_pair(k, v));
	}
    else{
    	fprintf(stderr,"error found\n");
	}
  }

	return true;
}

template<typename T>
bool FileParser<T>::ReadSample(T& y,
		std::vector<std::pair<size_t, T> >& x) {
	std::lock_guard<SpinLock> lock(lock_);
	char *buf = (this->*ReadLineImpl)(buf_, buf_size_);
	if (!buf) return false;

	buf_ = buf;
	return ParseSample(buf, y, x);
}

template<typename T>
bool FileParser<T>::ReadSample(int& click,int & impre,
		std::vector<std::pair<size_t, T> >& x) {
	std::lock_guard<SpinLock> lock(lock_);
	char *buf = (this->*ReadLineImpl)(buf_, buf_size_);
	if (!buf) {
		return false;
	}

	buf_ = buf;
	return ParseSample(buf, click,impre, x);
}

template<typename T>
bool FileParser<T>::ReadSampleMultiThread(T& y,
		std::vector<std::pair<size_t, T> >& x) {
	char *buf = ReadLine();
	if (!buf) return false;

	bool suc = ParseSample(buf, y, x);
	free(buf);
	return suc;
}

template<typename T>
bool FileParser<T>::ReadSampleMultiThread(int& click,int& impre,
		std::vector<std::pair<size_t, T> >& x) {
	char *buf = ReadLine();
	if (!buf) return false;

	bool suc =  ParseSample(buf, click,impre, x);
	free(buf);
	return suc;
}


#endif // SRC_FILE_PARSER_H
/* vim: set ts=4 sw=4 tw=0 noet :*/
