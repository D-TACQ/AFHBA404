/*
 * AcqSys.cpp
 *
 *  Created on: 27 Feb 2020
 *      Author: pgm
 */

#include "AcqSys.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include "nlohmann/json.hpp"
#include <string.h>

using json = nlohmann::json;

VI::VI() {
	memset(this, 0, sizeof(VI));
}

int VI::len(void) const {
	return AI16*2 + AI32*4 + DI32*4 + SPAD*4;
}
VI& VI::operator += (const VI& right) {
	this->AI16 += right.AI16;
	this->AI32 += right.AI32;
	this->DI32 += right.DI32;
	this->SPAD += right.SPAD;
	return *this;
}

VO::VO() {
	memset(this, 0, sizeof(VO));
}
VO& VO::operator += (const VO& right) {
	this->AO16 += right.AO16;
	this->DO32 += right.DO32;
	return *this;
}

int VO::len(void) const {
	return AO16*2 + DO32*4;
}

IO::IO(string _name, VI _vi, VO _vo): name(_name), vi(_vi), vo(_vo), _string(0)
{

}

IO::~IO()
{
	if (_string) delete _string;
}

string IO::toString(void)
{
	return name + " VI:" + to_string(vi.len()) + " VO:" + to_string(vo.len());
}

void HBA::dump_config(void)
{
	cerr << toString() << endl;
	for (auto uut: uuts){
		cerr << "\t" << uut->toString() <<endl;
	}
}
void HBA::dump_data(const char* basename)
{
	cerr << "dump_data" <<endl;
}


ACQ::ACQ(string _name, VI _vi, VO _vo, VI& sys_vi_cursor, VO& sys_vo_cursor) :
		IO(_name, _vi, _vo),
		vi_cursor(sys_vi_cursor), vo_cursor(sys_vo_cursor)
{
	// @@todo hook the device driver.
	sys_vi_cursor += vi;
	sys_vo_cursor += vo;
}

bool ACQ::newSample(void)
{
	return false;
}

unsigned ACQ::tlatch(void)
{
	return 0;
}

void ACQ::arm(int nsamples)
{

}

int get_int(json j)
{
	try {
		return	j.get<int>();
	} catch (std::exception& e){
		return 0;
	}
}

HBA::HBA(int _devnum, vector <ACQ*> _uuts, VI _vi, VO _vo):
		IO("HBA"+to_string(_devnum), _vi, _vo),
		uuts(_uuts), vi(_vi), vo(_vo)
{}

int devnum;

HBA& HBA::create(const char* json_def)
{
	json j;
	std::ifstream i(json_def);
	i >> j;

	struct VI VI_sys;
	struct VO VO_sys;
	vector <ACQ*> uuts;

	for (auto uut : j["AFHBA"]["UUT"]) {
		VI vi;

		vi.AI16 = get_int(uut["VI"]["AI16"]);
		vi.AI32 = get_int(uut["VI"]["AI32"]);
		vi.DI32 = get_int(uut["VI"]["DI32"]);
		vi.SPAD = get_int(uut["VI"]["SPAD"]);

		VO vo;
		vo.AO16 = get_int(uut["VO"]["AO16"]);
		vo.DO32 = get_int(uut["VO"]["AO16"]);
		uuts.push_back(new ACQ(uut["name"], vi, vo, VI_sys, VO_sys));
	}
	return * new HBA(::devnum, uuts, VI_sys, VO_sys);
}


