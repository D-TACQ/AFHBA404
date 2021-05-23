/** @file DefaultSystemInterface.cpp
 *  @brief SystemInterface implementation example
 *  overloads ringDoorbell() to actually "do work".
 *  a real implementation should perform a shared memory interface with another process.
 *
 *  Created on: 14 Mar 2020
 *      Author: pgm
 *
 *  Demo Modes: dup1, duplicate AI0
 *      export SINGLE_THREAD_CONTROL=control_dup1
 *
 */

#include "AcqSys.h"
#include <string.h>



#define PWM_MAGIC	0xbe8bb8fa			// makes for a good display

class DummySingleThreadControlSystemInterface: public SystemInterface {

public:
	DummySingleThreadControlSystemInterface(const HBA& hba) :
		SystemInterface(hba)
	{
		if (G::verbose) printf("%s::%s DUP1:%d\n", __FILE__, PFN, DUP1);
	}
	static int DUP1;

	virtual void ringDoorbell(int sample){
		G::verbose && printf("%s(%d)\n", PFN, sample);

		short xx = IN.AI16[DUP1];
		for (int ii = 0; ii < AO16_count(); ++ii){
			OUT.AO16[ii] = xx;
		}
		unsigned tl = tlatch();
		for (int ii = 0; ii < DO32_count(); ++ii){
			OUT.DO32[ii] = tl;
		}
		for (int ii = 0; ii < PW32_count(); ++ii){
			for (int cc = 0; cc < PW32LEN; ++cc){
				OUT.PW32[ii][cc] = PWM_MAGIC;
			}
		}
	}
};

int DummySingleThreadControlSystemInterface::DUP1;



#define AI128AO64_HACK	2


class MatrixMultiplySingleThreadControlSystemInterface: public SystemInterface {

	float* aMX;

	void fill_diagonal(float gain)
	{
		if (G::verbose) printf("%s::%s GAIN:%.3f\n", __FILE__, PFN, gain);

		for (int ao = 0; ao < AO16_count(); ++ao){
			aMX[ao*AI16_count() + ao] = gain;
		}
	}
	void fill_n_diagonal(float gain)
	/* when AO_count() < AI_count() (usual case), keep making diagonals for ALL AI */
	{
		if (G::verbose) printf("%s::%s GAIN:%.3f\n", __FILE__, PFN, gain);

		for (int id = 0; id*AO16_count() < AI16_count(); ++id){
			for (int ao = 0; ao < AO16_count(); ++ao){
				aMX[ao*AI16_count() + AO16_count()/AI128AO64_HACK*id + ao] = gain;
			}
		}
	}
	void fill_all(float gain)
	{
		if (G::verbose) printf("%s::%s GAIN:%.3f\n", __FILE__, PFN, gain);

		for (int ao = 0; ao < AO16_count(); ++ao){
                        for (int ai = 0; ai < AI16_count()/AI128AO64_HACK; ++ai){
                        	aMX[ao*AI16_count() + ai] = gain;
                        }
                }

	}
	void fill_matrix()
	{
		float gain = 1.0;
		const char* key = getenv("GAIN");
		if (key){
			gain = atof(key);
		}
		if (G::verbose) printf("%s::%s GAIN:%.3f\n", __FILE__, PFN, gain);

		switch (MODE){
		case 1:
			return fill_diagonal(gain);
		case 2:
			return fill_n_diagonal(gain);
		case 3:
			return fill_all(gain);
		default:
			return fill_all(0.0);
		}
	}
	const char float2char(float xx){
		if (xx*10 >= 10){
			return 'X';
		}else if (xx < 0.1 && xx > 0.0001){
			return 'x';
		}else{
			int ii = (int)(xx*10)%10;
			if (ii){
				return ii+'0';
			}else{
				return '.';
			}
		}
	}
	void print_matrix() {
		for (int ao = 0; ao < AO16_count(); ++ao){
                        for (int ai = 0; ai < AI16_count(); ++ai){
                        	printf("%c", float2char(aMX[ao*AI16_count() + ai]));
                        }
                        printf("\n");
                }
	}
public:
	MatrixMultiplySingleThreadControlSystemInterface(const HBA& hba) :
		SystemInterface(hba)
	{
		if (G::verbose) printf("%s::%s MODE:%d\n", __FILE__, PFN, MODE);

		aMX = new float[AO16_count()*AI16_count()];
		fill_matrix();
		if (G::verbose > 1){
			print_matrix();
		}
	}
	static int MODE;   /* 1: single diagonal, 3: multiple diagonals (assuming AICHAN > AO CHAN) */

	virtual void ringDoorbell(int sample){
		G::verbose && printf("%s(%d)\n", PFN, sample);
		int ao, ai;

		for (ao = 0; ao < AO16_count(); ++ao){
			float tmp = 0;
			for (ai = 0; ai < AI16_count(); ++ai){
				tmp += IN.AI16[ai]*aMX[ao*AI16_count() + ai];
			}
			int raw = tmp;
			if (raw > 32768){
				raw = 32768;
			}else if (raw < -32768){
				raw = 32768;
			}
			G::verbose && ao==0 && printf("%s(%d) %.2f %d\n", PFN, sample, tmp, raw);
			OUT.AO16[ao] = raw;
		}

		for (int ii = 0; ii < DO32_count(); ++ii){
			OUT.DO32[ii] = tlatch();
		}
	}
};

int MatrixMultiplySingleThreadControlSystemInterface::MODE;

SystemInterface& SystemInterface::factory(const HBA& hba)
{
	if (G::verbose) printf("%s::%s\n", __FILE__, PFN);

	const char* key = getenv("ZCOPY");
	if (key && *key == 'y'){
		printf("%s ZCOPY selected\n", PFN);
		return * new SystemInterface(hba);
	}

	key = getenv("SINGLE_THREAD_CONTROL");
	if (key){
		if (sscanf(key, "control_dup1=%d", &DummySingleThreadControlSystemInterface::DUP1) == 1 ||
		    strcmp(key, "control_dup1") == 0){
			return * new DummySingleThreadControlSystemInterface(hba);
		}
		if (sscanf(key, "control_mx=%d", &MatrixMultiplySingleThreadControlSystemInterface::MODE) == 1){
			return * new MatrixMultiplySingleThreadControlSystemInterface(hba);
		}
	}

	return * new SystemInterface(hba);
}

