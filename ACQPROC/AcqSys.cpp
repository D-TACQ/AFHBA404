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

#include <stdio.h>		// because sprintf()

using json = nlohmann::json;

VI::VI() {
	memset(this, 0, sizeof(VI));
}

int VI::len(void) const {
	return AI16*2 + AI32*4 + DI32*4 + SP32*4;
}
VI& VI::operator += (const VI& right) {
	this->AI16 += right.AI16;
	this->AI32 += right.AI32;
	this->DI32 += right.DI32;
	this->SP32 += right.SP32;
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

void HBA::dump_config()
{
	cerr << toString() << endl;
	int port = 0;
	for (auto uut: uuts){
		cerr << "\t" << "[" << port << "] " << uut->toString() <<endl;
		++port;
	}
}
void HBA::dump_data(const char* basename)
{
	cerr << "dump_data" <<endl;
}


ACQ::ACQ(string _name, VI _vi, VO _vo, VI& sys_vi_cursor, VO& sys_vo_cursor) :
		IO(_name, _vi, _vo),
		vi_cursor(sys_vi_cursor), vo_cursor(sys_vo_cursor), nowait(false), wd_mask(0)
{
	// @@todo hook the device driver.
	sys_vi_cursor += vi;
	sys_vo_cursor += vo;
}

string ACQ::toString() {
	char wd[80] = {};
	if (wd_mask){
		sprintf(wd, " WD mask: 0x%08x", wd_mask);
	}
	return IO::toString() + " SPAD:" + to_string(vi_cursor.SP32) + 
			" VI Offsets " + to_string(vi_cursor.AI16)+ "," + to_string(vi_cursor.SP32) + wd;
}
bool ACQ::newSample(int sample)
{
	cerr << " new sample: " << sample << " " << getName() << endl;
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

void store_config(json j, string fname, HBA hba)
{
	int ii = 0;
	j["OFFSETS"] = json::array();
	for (auto uut : hba.uuts){
		json &offsets = j["OFFSETS"][ii++];
		std::map<std::string, int> vi_map {
			{ "AI16", uut->vi_cursor.AI16 },
			{ "DI32", uut->vi_cursor.DI32 },
			{ "SP32", uut->vi_cursor.SP32 }
		};
		offsets.push_back(json::object_t::value_type("VI", vi_map));
		std::map<std::string, int> vo_map {
			{ "AO16", uut->vo_cursor.AO16 },
			{ "DO32", uut->vo_cursor.DO32 }
		};
		offsets.push_back(json::object_t::value_type("VO", vo_map));
	}
	std::ofstream o("runtime.json");
	o << std::setw(4) << j << std::endl;
}

int devnum;

HBA& HBA::create(const char* json_def)
{
	json j;
	std::ifstream i(json_def);
	i >> j;

	struct VI VI_sys;
	struct VO VO_sys;
	vector <ACQ*> uuts;
	int port = 0;
	string port0_type;

	for (auto uut : j["AFHBA"]["UUT"]) {
		VI vi;

		vi.AI16 = get_int(uut["VI"]["AI16"]);
		vi.AI32 = get_int(uut["VI"]["AI32"]);
		vi.DI32 = get_int(uut["VI"]["DI32"]);
		vi.SP32 = get_int(uut["VI"]["SP32"]);

		VO vo;
		vo.AO16 = get_int(uut["VO"]["AO16"]);
		vo.DO32 = get_int(uut["VO"]["AO16"]);
		ACQ *acq = new ACQ(uut["name"], vi, vo, VI_sys, VO_sys);

		try {
			int wd_bit = uut["WD_BIT"].get<int>();
			if (port == 0){
				acq->wd_mask = 1 << wd_bit;
			}else{
				cerr << "WARNING: " << uut["name"] << " attempted to set WD_BIT when PORT (" << port << ") != 0" <<endl;
			}
		} catch (exception& e) {
			;
		}
		// check unit compatibility
		if (port == 0){
			port0_type = uut["type"];
		}else{
			if (port0_type == "pcs" && uut["type"] == "bolo"){
				cerr << "NOTICE: port " << port << " is bolo in non-bolo set, set nowait" << endl;
				acq->nowait = true;
			}else if (port0_type == "bolo" && uut["type"] != "bolo"){
				cerr << "WARNING: port " << port << " is NOT bolo when port0 IS bolo" << endl;
			}
		}
		uuts.push_back(acq);
		++port;
	}
	HBA& the_hba = * new HBA(::devnum, uuts, VI_sys, VO_sys);
	store_config(j, "runtime.json", the_hba);
	return the_hba;
}


