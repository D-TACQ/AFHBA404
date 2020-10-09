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

#include <string.h>

using namespace std;

int getenv(const char* key, int def, int (*cvt)(const char* key) = atoi);


/** Models Vector Input.
 * Vector Input is a single sample input data set, pushed to DRAM by single DMA from ACQ
 */
struct VI {
	int len(void) const;
	VI& operator+= (const VI& right);
	VI offsets(void) const;
	int AI16;		/**< #AI16 values from the HW. */
	int AI32;		/**< #AI32 values from the HW. */
	int DI32;		/**< #DI32 values from the HW. */
	int SP32;		/**< #SP32 values from the HW. */
	VI();
};

/** SPIX: defined fields in SP32 array */
enum SPIX {
	TLATCH = 0,			/**< Sample Number */
	USECS = 1,			/**< microseconds since trigger */
	POLLCOUNT = 2,			/**< pollcount: number of times SW polled for incoming. <=1 : ERROR (data too early) */
};

#define PW32LEN  32
typedef unsigned PW32V[PW32LEN];

/** Models Vector Output.
 * Vector Output is a single sample output data set, fetched by single DMA from ACQ
 */
struct VO {
	int len(void) const;
	int hwlen(void) const;
	VO& operator+= (const VO& right);
	VO offsets(void) const;
	int AO16;		/**< #AO16 values from the algorithm. OUTPUT to HW */
	int DO32;		/**< #DO32 values from the algorithm. OUTPUT to HW */
	int PW32;		/**< #PW32 values from the algorithm. OUTPUT to HW */
	int CC32;		/**< #CALC values from the algorithm. NOT OUTPUT to HW */
	VO();
};


struct SystemInterface;

/** Base Class */
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


/** abstract model of an ACQ2106 box. */
class ACQ: public IO
{
protected:
	ACQ(int devnum, string _name, VI _vi, VO _vo, VI _vi_offsets, VO _vo_offsets, VI& sys_vi_cursor, VO& sys_vo_cursor);
        virtual ~ACQ();

	bool nowait;	// newSample doesn't block for new Tlatch (eg bolo in set with non bolo uuts
	unsigned wd_mask;
	int pollcount;
	int devnum;
public:
	const VI vi_offsets; 	/**< byte offset for each Input type in Local Vector In */
	const VO vo_offsets; 	/**< byte offset for each Output type in Local Vector Out */
	const VI vi_cursor;	/**< index for each Input type in System Interface */
	const VO vo_cursor;	/**< index for each Output type in System Interface */

	virtual string toString();

	virtual bool newSample(int sample);
	/*< checks host buffer for new sample, if so copies to lbuf and reports true */
	virtual void action(SystemInterface& systemInterface) {}
	virtual void action2(SystemInterface& systemInterface) {}
	virtual unsigned tlatch(void);
	/**< returns latest tlatch from lbuf */
	virtual void arm(int nsamples);
	/**< prepare to run a shot nsamples long, arm the UUT. */
    static ACQ* factory(int devnum, string _name, VI _vi, VO _vo, VI _vi_offsets,
    		VO _vo_offsets, VI& sys_vi_cursor, VO& sys_vo_cursor);

    friend class HBA;
};

/** interface to AFHBA404 device driver. */
struct Dev;



/** Models a Host Bus Adapter like AFHBA404. */
class HBA: public IO
{
	HBA(vector <ACQ*> _uuts, VI _vi, VO _vo);
	static HBA* the_hba;	/**< singleton, ugly interface. */
public:
	virtual ~HBA();
	int devnum;
	vector<ACQ*> uuts;	/**< vector of ACQ UUT's */
	vector<int> devs;	/**< AFHBA devnum. NB: "HBA" ... works with >4 ports, so multiple HBA's supported */
	const VI vi;		/**< total system size each Input type. */
	const VO vo;		/**< total system size each Output type. */
	static int maxsam;	/**< max samples in shot (for raw memory alloc) */

	static HBA& create(const char* json_def, int _maxsam);
	static HBA& instance() { return *the_hba; }

	virtual void start_shot();
	virtual void processSample(SystemInterface& systemInterface, int sample);
	/**< core run time function, processSample. */

	void dump_config();
	/**< output complete configuration with calculated offsets */
	void dump_data(const char* basename);
	/**< output raw data for each ACQ */

	virtual string toString();
};

namespace G {
	extern int nsamples;
	extern int verbose;
	extern int dummy_first_loop;
	extern int samples_buffer;
	extern int maxpoll;
};

/** Models interface with external PCS.
 * Users can create a custom subclass to implement shared memory, comms
 */
struct SystemInterface {
private:
	const HBA& hba;

public:
	/** ONE vector each type, all VI from all UUTS are split into types and
	 *   aggregated in the appropriate vectors.
	 */
	struct Inputs {
		short *AI16;
		int *AI32;
		unsigned *DI32;
		unsigned *SP32;
	} IN;
	/**< ONE vector each type, scatter each type to appropriate VO all UUTS
	 */
	struct Outputs {
		short* AO16;
		unsigned *DO32;
		PW32V *PW32;			/* 32 demand values packed into shorts */
		unsigned *CC32;			/* calc values from PCS .. NOT outputs. */
	} OUT;

	SystemInterface(const HBA& _hba);
	virtual ~SystemInterface();
	virtual void trigger()
	{}
	virtual void ringDoorbell(int sample)
	/**< alert PCS that there is new data .. implement by subclass.
	 */
	{}

	static SystemInterface& factory(const HBA&);

	unsigned tlatch() {
		return IN.SP32[0];
	}
	unsigned AI16_count() const {
		return hba.vi.AI16;
	}
	unsigned AI32_count() const {
		return hba.vi.AI32;
	}
	unsigned DI32_count() const {
		return hba.vi.DI32;
	}
	unsigned SP32_count() const {
		return hba.vi.SP32;
	}
	unsigned AO16_count() const {
		return hba.vo.AO16;
	}
	unsigned DO32_count() const {
		return hba.vo.DO32;
	}
	unsigned PW32_count() const {
		return hba.vo.PW32;
	}
	unsigned CC32_count() const {
		return hba.vo.CC32;
	}
};

template <class T>
T* new_zarray(int nelems)
{
	T* nz_array = new T[nelems];
	memset(nz_array, 0, nelems*sizeof(T));
	return nz_array;
}

#define PFN __PRETTY_FUNCTION__


#endif /* ACQPROC_ACQSYS_H_ */
