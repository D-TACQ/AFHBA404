/*
 * Env.cpp
 *
 *  Created on: 19 Sep 2019
 *      Author: pgm
 */


// @@todo : DO not enforce namespace on client


#include "Env.h"

using namespace std;

Env::Env(const char* fname) {
	ifstream fs(fname);
	string line;
	while(getline(fs, line)){
		//			cout <<  line << endl;
		if (line.find("#") == 0){
//			cerr << line << " skip comment" <<endl;
			continue;
		}
		size_t pos;
		if ((pos = line.find("=")) != std::string::npos ){
			string key = line.substr(0, pos);
			string value = line.substr(pos+1);
			value.erase(std::remove(value.begin(), value.end(), '"'), value.end());

//			cerr << "k:" << key << " v:" << value << endl;
			_env[key] = value;
		}
	}
	fs.close();
}

string& Env::operator() (std::string key) {
	return _env[key];
}
