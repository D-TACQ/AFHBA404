/** @file AcqSys.cpp
 * @brief creates system configuration from config.
 *
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

#include <assert.h>



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

VI VI::offsets(void) const
{
	VI viff;
	viff.AI16 = 0;
	viff.AI32 = 0;
	assert(!(AI16 != 0 && AI32 != 0));
	assert(sizeof(int) == 4);
	viff.DI32 = AI16*sizeof(short) + AI32*sizeof(int);
	viff.SP32 = viff.DI32 + DI32*sizeof(int);
	return viff;
}

VO::VO() {
	memset(this, 0, sizeof(VO));
}
VO& VO::operator += (const VO& right) {
	this->AO16 += right.AO16;
	this->DO32 += right.DO32;
	this->PW32 += right.PW32;
	this->CC32 += right.CC32;
	return *this;
}

VO VO::offsets(void) const
{
	VO voff;
	voff.AO16 = 0;
	voff.DO32 = AO16*sizeof(short);
	voff.PW32 = voff.DO32 + DO32*sizeof(unsigned);
	voff.CC32 = voff.PW32 + PW32*sizeof(PW32V);
	return voff;
}


int VO::hwlen(void) const {
	return AO16*2 + DO32*4 + PW32*sizeof(PW32V);
}

int VO::len(void) const {
	return hwlen() + CC32*4;
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


ACQ::ACQ(int _devnum, string _name, VI _vi, VO _vo, VI _vi_offsets, VO _vo_offsets, VI& sys_vi_cursor, VO& sys_vo_cursor) :
		IO(_name, _vi, _vo),
		devnum(_devnum),
		vi_offsets(_vi_offsets), vo_offsets(_vo_offsets),
		vi_cursor(sys_vi_cursor), vo_cursor(sys_vo_cursor),
		nowait(false), wd_mask(0), pollcount(0)
{
	// @@todo hook the device driver.
	sys_vi_cursor += vi;
	sys_vo_cursor += vo;
}

ACQ::~ACQ() {

}
string ACQ::toString() {
	char wd[80] = {};
	if (wd_mask){
		sprintf(wd, " WD mask: 0x%08x", wd_mask);
	}
	return "dev:" + to_string(devnum) + " " + IO::toString() +  " Offset of SPAD IN VI :" + to_string(vi_offsets.SP32) + "\n"
			" System Interface Indices " + to_string(vi_cursor.AI16)+ "," + to_string(vi_cursor.SP32) + wd;
}
bool ACQ::newSample(int sample)
{
	cerr << " new sample: " << sample << " " << getName() << endl;
	return true;
}

unsigned ACQ::tlatch(void)
{
	return 0;
}

void ACQ::arm(int nsamples)
{
	cerr << "placeholder: ARM unit " << getName() << " now" <<endl;
}

int get_int(json j, int default_value = 0)
{
	try {
		return	j.get<int>();
	} catch (std::exception& e){
		return default_value;
	}
}

#define HBA_DEVNUM(_u) ((_u)[0]->devnum&~0x3)

HBA::HBA(vector <ACQ*> _uuts, VI _vi, VO _vo):
		devnum(HBA_DEVNUM(_uuts)),
		IO("HBA"+to_string(HBA_DEVNUM(_uuts)), _vi, _vo),
		uuts(_uuts), vi(_vi), vo(_vo)
{
	for (auto uut: uuts){
		devs.push_back(uut->devnum);
	}
}

HBA::~HBA() {
    for (auto uut : uuts){
        delete uut;
    }
}
extern "C" {
	extern int sched_fifo_priority;
}


void HBA::processSample(SystemInterface& systemInterface, int sample)
{
	for (auto uut : uuts){
		while(!uut->newSample(sample)){
			sched_fifo_priority>1 || sched_yield();
			++uut->pollcount;
			if (G::maxpoll && sample && uut->pollcount > G::maxpoll){
				fprintf(stderr, "ERROR: poll timeout on uut %s at sample %d\n", uut->getName().c_str(), sample);
				throw -22;
			}
		}
	}
	for (auto uut : uuts){
		uut->action(systemInterface);
	}
	systemInterface.ringDoorbell(sample);
	for (auto uut : uuts){
		uut->action2(systemInterface);
	}
}

string HBA::toString()
{
	string dev_str = "devs=";
	int idev = 0;
	for (auto dev: devs){
		if (idev++ != 0){
			dev_str += ",";
		}
		dev_str += to_string(dev);
	}
	return IO::toString() + " " + dev_str;
}

typedef pair<std::string, int> KVP;		/** Key Value Pair */
typedef std::map <std::string, int> KVM;	/** Key Value Map  */

typedef pair<std::string, string> KVPS; /** Key Value Pair, COMMENT */
typedef std::map <std::string, string> KVMS;	/** Key Value Map  */

//#define COM(comment) KVPS("__comment__", comment)

#define COM(i) "__comment" #i "__"

#define INSERT_IF(map, vx, vxo, field) \
	if (uut->vx.field){	\
		map.insert(KVP(#field, uut->vxo.field)); \
	}

void add_comments(json& jsys, string& fname)
{
	jsys[COM(1)] = "created from " + fname;
	jsys[COM(2)] = "LOCAL VI_OFFSETS: field offset VI in bytes";
	jsys[COM(3)] = "LOCAL VO_OFFSETS: field offset VO in bytes";
	jsys[COM(4)] = "LOCAL VX_LEN: length of VI|VO in bytes";
	jsys[COM(5)] = "GLOBAL_LEN: total length of each type in SystemInterface";
	jsys[COM(6)] = "GLOBAL_INDICES: index of field in type-specific array in SI";
	jsys[COM(7)] = "SPIX: Scratch Pad Index, index of field in SP32";


	KVM spix_map;
	spix_map.insert(KVP("TLATCH", 	 SPIX::TLATCH));
	spix_map.insert(KVP("USECS", 	 SPIX::USECS));
	spix_map.insert(KVP("POLLCOUNT", SPIX::POLLCOUNT));
	jsys["SPIX"] = spix_map;
}
void add_si_lengths(json& jsys, HBA& hba)
{
	SystemInterface si(hba);
	json &jgl = jsys["GLOBAL_LEN"];
	json &jlen_vi = jgl["VI"];
	jlen_vi = {
		    { "AI16", si.AI16_count() },
		    { "AI32", si.AI32_count() },
		    { "DI32", si.DI32_count() },
		    { "SP32", si.SP32_count() }
		  };
	json &jlen_vo = jgl["VO"];
	jlen_vo = {
		    { "AO16", si.AO16_count() },
		    { "DO32", si.DO32_count() },
		    { "PW32", si.PW32_count() },
		    { "CC32", si.CC32_count() }
		  };

}
void store_config(json j, string fname, HBA& hba)
{
	json &jsys = j["SYS"];
	add_comments(jsys, fname);
	add_si_lengths(jsys, hba);

	int ii = 0;
	for (auto uut : hba.uuts){
		json &jlo = jsys["UUT"]["LOCAL"][ii];

		jlo["VX_LEN"] = { {  "VI", uut->vi.len() }, { "VO", uut->vo.len() } };

		KVM vi_offsets_map;
		INSERT_IF(vi_offsets_map, vi, vi_offsets, AI16);
		INSERT_IF(vi_offsets_map, vi, vi_offsets, AI32);
		INSERT_IF(vi_offsets_map, vi, vi_offsets, DI32);
		INSERT_IF(vi_offsets_map, vi, vi_offsets, SP32);
		jlo["VI_OFFSETS"] = vi_offsets_map;

		KVM vo_offsets_map;
		INSERT_IF(vo_offsets_map, vo, vo_offsets, AO16);
		INSERT_IF(vo_offsets_map, vo, vo_offsets, DO32);
		INSERT_IF(vo_offsets_map, vo, vo_offsets, PW32);
		INSERT_IF(vo_offsets_map, vo, vo_offsets, CC32);
		jlo["VO_OFFSETS"] = vo_offsets_map;

		json &jix = jsys["UUT"]["GLOBAL_INDICES"][ii++];
		KVM vi_map;
		INSERT_IF(vi_map, vi, vi_cursor, AI16);
		INSERT_IF(vi_map, vi, vi_cursor, AI32);
		INSERT_IF(vi_map, vi, vi_cursor, DI32);
		INSERT_IF(vi_map, vi, vi_cursor, SP32);
		jix["VI"] = vi_map;

		KVM vo_map;
		INSERT_IF(vo_map, vo, vo_cursor, AO16);
		INSERT_IF(vo_map, vo, vo_cursor, DO32);
		INSERT_IF(vo_map, vo, vo_cursor, PW32);
		INSERT_IF(vo_map, vo, vo_cursor, CC32);
		jix["VO"] = vo_map;
	}


	std::ofstream o("runtime.json");
	o << std::setw(4) << j << std::endl;
}

int HBA::maxsam;

HBA* HBA::the_hba;


bool strstr(string haystack, string needle)
{
	return haystack.find(needle) != string::npos;
}


/** HBA::Create() factory function. */
HBA& HBA::create(const char* json_def, int _maxsam)
{
	json j;
	std::ifstream i(json_def);
	i >> j;

	maxsam = _maxsam;
	struct VI VI_sys;
	struct VO VO_sys;
	vector <ACQ*> uuts;
	int port = 0;
	int iuut = 0;
	string first_type;

	int hba_devnum = get_int(j["AFHBA"]["DEVNUM"]);

	for (auto uut : j["AFHBA"]["UUT"]) {

		if (get_int(uut["DEVNUM"], -1) != -1){
			hba_devnum = get_int(uut["DEVNUM"]);
			port = 0;
		}

		VI vi;

		vi.AI16 = get_int(uut["VI"]["AI16"]);
		vi.AI32 = get_int(uut["VI"]["AI32"]);
		vi.DI32 = get_int(uut["VI"]["DI32"]);
		vi.SP32 = get_int(uut["VI"]["SP32"]);

		VO vo;
		vo.AO16 = get_int(uut["VO"]["AO16"]);
		vo.DO32 = get_int(uut["VO"]["DO32"]);
		vo.PW32 = get_int(uut["VO"]["PW32"]);
		vo.CC32 = get_int(uut["VO"]["CC32"]);

		ACQ *acq = ACQ::factory(hba_devnum+port, uut["name"], vi, vo, vi.offsets(), vo.offsets(), VI_sys, VO_sys);

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
		string uut_type = uut["type"];
		// check unit compatibility
		if (iuut == 0){
			first_type = uut_type;
		}else{
			if (::strstr(first_type, "pcs") && ::strstr(uut_type, "bolo")){
				cerr << "NOTICE: port " << port << " is bolo in non-bolo set, set nowait" << endl;
				acq->nowait = true;
			}else if (::strstr(first_type, "pcs") && ::strstr(uut_type, "nowait")){
				cerr << "NOTICE: port " << port << " set nowait" << endl;
				acq->nowait = true;
			}else if (::strstr(first_type, "bolo") && !::strstr(uut_type, "bolo")){
				cerr << "WARNING: port " << port << " is NOT bolo when port0 IS bolo" << endl;
			}
		}
		uuts.push_back(acq);
		++port;
		++iuut;
	}
	the_hba = new HBA(uuts, VI_sys, VO_sys);
	store_config(j, json_def, *the_hba);
	return *the_hba;
}

SystemInterface::SystemInterface(const HBA& _hba) : hba(_hba)
/* make a gash SI to allow simulated operation. The real shm is customer specific */
{
	IN.AI16 = new_zarray<short>(AI16_count());
	IN.AI32 = new_zarray<int>(AI32_count());
	IN.DI32 = new_zarray<unsigned>(DI32_count());    // needs to be bigger for PWM
	IN.SP32 = new_zarray<unsigned>(SP32_count());

	OUT.AO16 = new_zarray<short>(AO16_count());
	OUT.DO32 = new_zarray<unsigned>(DO32_count());
	OUT.PW32 = new_zarray<PW32V>(PW32_count());
	OUT.CC32 = new_zarray<unsigned>(CC32_count());
}
SystemInterface::~SystemInterface()
{
	delete [] IN.AI16;
	delete [] IN.AI32;
	delete [] IN.DI32;
	delete [] IN.SP32;
	delete [] OUT.AO16;
	delete [] OUT.DO32;
	delete [] OUT.PW32;
	delete [] OUT.CC32;
}





