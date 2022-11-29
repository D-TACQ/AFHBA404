/* ------------------------------------------------------------------------- */
/* File.h  D-TACQ ACQ400 FMC  DRIVER                                   
 * Project: ACQ420_FMC
 * Created: 5 Mar 2016  			/ User: pgm
 * ------------------------------------------------------------------------- *
 *   Copyright (C) 2016 Peter Milne, D-TACQ Solutions Ltd         *
 *                      <peter dot milne at D hyphen TACQ dot com>           *
 *                                                                           *
 *  This program is free software; you can redistribute it and/or modify     *
 *  it under the terms of Version 2 of the GNU General Public License        *
 *  as published by the Free Software Foundation;                            *
 *                                                                           *
 *  This program is distributed in the hope that it will be useful,          *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of           *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            *
 *  GNU General Public License for more details.                             *
 *                                                                           *
 *  You should have received a copy of the GNU General Public License        *
 *  along with this program; if not, write to the Free Software              *
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.                *
 *
 * TODO 
 * TODO
\* ------------------------------------------------------------------------- */

#ifndef FILE_H_
#define FILE_H_

#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>

#include <string>
#include <stdarg.h>
#include <unistd.h>

#include "local.h"


class File {
	FILE *_fp;

	void _file(const char* mode, bool check = true) {
		_fp = fopen(fname, mode);
		if (check && _fp == 0){
			perror(fname);
			exit(1);
		}
	}
	char* buf;
public:
	const char* fname;
	static const bool NOCHECK = false;

	File(const char *_fname, const char* mode = "r", bool check = true): buf(0), fname(_fname) {
		_file(mode, check);
	}
	File(const char* path, const char *_fname, const char* mode, bool check = true){
		buf = new char[strlen(path)+1+strlen(fname)+1];
		strcpy(buf, path);
		strcat(buf, "/");
		strcat(buf, _fname);
		fname = buf;
		_file(mode, check);
	}
	~File() {
		if (_fp) fclose(_fp);
		if (buf) delete [] buf;
	}
	FILE* fp() {
		return _fp;
	}
	bool exists() {
		return fp() != 0;
	}
	int fd() {
		return fileno(_fp);
	}
	FILE* operator() () {
		return _fp;
	}
	int printf(const char* fmt, ...){
		va_list argp;
		va_start(argp, fmt);
		int rc = vfprintf(_fp, fmt, argp);
		va_end(argp);
		return rc;
	}
};

#ifndef NOMAPPING

template <class T>
class Mapping {
	int len;
	T* _data;
	int fd;
public:
	Mapping(std::string fname, int _len) : len(_len) {
		int fd = open(fname.data(), O_RDWR, 0777);
		_data = static_cast<T*>(mmap(0, len,
			PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0));
		if (_data == MAP_FAILED){
			perror(fname.data());
			exit(1);
		}
	}
	~Mapping() {
		munmap(_data, len);
		close(fd);
	}
	const T* operator() () {
		return _data;
	}
};

#endif

static inline FILE *fopen_safe(const char* fname, const char* mode = "r")
{
	FILE *fp = fopen(fname, mode);
	if (fp == 0){
		perror(fname);
		exit(errno);
	}
	return fp;
}

template <class T>
static
T getvalue(const char* fname, const char* mode = "r")
{
	File f(fname, mode);
	FILE* fp = f();
	T value = 0;
	if (fread(&value, sizeof(T), 1, fp) != 1){
		fprintf(stderr, "ERROR: %s fread \"%s\" fail\n", PFN, fname);
		exit(1);
	}
	return value;
}

template <class T>
static
T getvalue(File& file)
{
	FILE* fp = file();
	T value = 0;
	if (fread(&value, sizeof(T), 1, fp) != 1){
		fprintf(stderr, "ERROR: %s fread \"%s\" fail\n", PFN, file.fname);
		exit(1);
	}
	return value;
}


#endif /* FILE_H_ */
