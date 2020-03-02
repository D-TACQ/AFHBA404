/*
 * AcqSys.h
 *
 *  Created on: 27 Feb 2020
 *      Author: pgm
 */

#ifndef ACQPROC_ACQSYS_H_
#define ACQPROC_ACQSYS_H_

#include <vector>
#include <string>

using namespace std;


/** VI : models Vector Input */
struct VI {
	int len(void) const;
	VI& operator+= (const VI& right);
	VI offsets(void) const;
	int AI16;
	int AI32;
	int DI32;
	int SP32;
	VI();
};

/** VO : Models Vector Output */
struct VO {
	int len(void) const;
	VO& operator+= (const VO& right);
	VO offsets(void) const;
	int AO16;
	int DO32;
	VO();
};

/** SystemInterface : Models interface with external PCS. */
struct SystemInterface {
	struct Inputs {
		short *AI16;
		int *AI32;
		unsigned *DI32;
		unsigned *SP32;
	} IN;
	struct Outputs {
		short* AO16;
		unsigned *DO32;
	} OUT;
	SystemInterface();
	virtual ~SystemInterface() {}
	virtual void ringDoorbell(int sample) {}
};

/** IO Base Class */
class IO {

	string name;
	string* _string;
public:
	const VI vi;
	const VO vo;

	IO(string _name, VI _vi, VO _vo);
	virtual ~IO();
	virtual string toString();
	string getName() {
		return name;
	}
};


/** class ACQ abstract model of an ACQ2106 box. */
class ACQ: public IO
{
	ACQ(string _name, VI _vi, VO _vo, VI _vi_offsets, VO _vo_offsets, VI& sys_vi_cursor, VO& sys_vo_cursor);
    virtual ~ACQ() {}
protected:
	bool nowait;	// newSample doesn't block for new Tlatch (eg bolo in set with non bolo uuts
	unsigned wd_mask;

public:
	const VI vi_offsets; /* byte offset for each Input type in Local Vector In */
	const VO vo_offsets; /* byte offset for each Output type in Local Vector Out */
	const VI vi_cursor;	/* offset for each Input type in System Interface */
	const VO vo_cursor;	/* offset for each Output type in System Interface */

	virtual string toString();

	virtual bool newSample(int sample);
	/*< checks host buffer for new sample, if so copies to lbuf and reports true */
	virtual void action(SystemInterface& systemInterface) {}
	virtual unsigned tlatch(void);
	/*< returns latest tlatch from lbuf */
	virtual void arm(int nsamples);
	/*< prepare to run a shot nsamples long, arm the UUT. */
friend class HBA;
friend class ACQ_HW;
};

/** struct Dev : interface to AFHBA404 device driver. */
struct Dev;

/** class ACQ_HW : concrete model of ACQ2106 box. */
class ACQ_HW: public ACQ
{
	Dev* dev;
	unsigned tl0;

	unsigned *dox;

	ACQ_HW(string _name, VI _vi, VO _vo, VI _vi_offsets,
			VO _vo_offsets, VI& sys_vi_cursor, VO& sys_vo_cursor);
	virtual ~ACQ_HW();

protected:

public:
	virtual bool newSample(int sample);
	/*< checks host buffer for new sample, if so copies to lbuf and reports true */
	virtual void action(SystemInterface& systemInterface);
	virtual unsigned tlatch(void);
	/*< returns latest tlatch from lbuf */
	virtual void arm(int nsamples);
	/*< prepare to run a shot nsamples long, arm the UUT. */
friend class HBA;
};

/** models a Host Bus Adapter like AFHBA404. */
class HBA: public IO
{
	HBA(int _devnum, vector <ACQ*> _uuts, VI _vi, VO _vo);

public:
	virtual ~HBA() {}
	int devnum;
	vector<ACQ*> uuts;
	const VI vi;
	const VO vo;
	static int maxsam;
	static HBA& create(const char* json_def, int _maxsam);

	virtual void processSample(SystemInterface& systemInterface, int sample);

	void dump_config();
	/*< output complete configuration with calculated offsets */
	void dump_data(const char* basename);
	/*< output raw data for each ACQ */
};




#endif /* ACQPROC_ACQSYS_H_ */
