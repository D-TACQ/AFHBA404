/*
 * Env.h
 *
 *  Created on: 19 Sep 2019
 *      Author: pgm
 */

#ifndef ENV_H_
#define ENV_H_

#include <stdlib.h>

#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <map>

class Env {
	std::map<std::string,std::string> _env;
public:

	Env(const char* fname);
	std::string& operator() (std::string key);

	static int getenv(const char* key, int def) {
		const char* sv = ::getenv(key);
		if (sv){
			return strtol(sv, 0, 0);
		}else{
			return def;
		}
	}
	static const char* getenv(const char* key, const char* def) {
		const char* sv = ::getenv(key);
		if (sv){
			return sv;
		}else{
			return def;
		}
	}
};




#endif /* ENV_H_ */
