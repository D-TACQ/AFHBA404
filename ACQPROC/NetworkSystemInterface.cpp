/*
 * NetworkSystemInterface.cpp
 *
 *  Created on: 24 Sep 2021
 *      Author: pgm
 */




#include "AcqSys.h"
#include <string.h>


class NetworkSystemInterface: public SystemInterface {

	FILE* fp_out; 	// this could be a network device, default is stdout */
	FILE* fp_in;    // this could be a non-blocking network device, default is stdin */

	// Output to stdout in real life will be BINARY. hexdump required */
	// Non-blocking input on stdin ... @todo.
#define A_PACKET_IS_WAITING 0

protected:
	// a concrete subclass could override with useful implementations ..
	virtual void sendPacket()
	{
		fwrite(IN.AI16, AI16_count(), sizeof(short), fp_out);
		fwrite(IN.SP32, 4, sizeof(long), fp_out);
	}
	virtual void receivePacket()
	{
		if (A_PACKET_IS_WAITING){
			int nread = fread(OUT.AO16, AO16_count(), sizeof(short), fp_in);
		}
	}
public:
	NetworkSystemInterface(const HBA& hba) :
		SystemInterface(hba),
		fp_out(stdout),
		fp_in(stdin)
	{
		G::verbose && printf("%s::%s\n", __FILE__, PFN);
	}


	virtual void ringDoorbell(int sample){
		G::verbose && printf("%s(%d)\n", PFN, sample);

		sendPacket();
		receivePacket();
	}
};

SystemInterface& SystemInterface::factory(const HBA& hba)
{
	if (G::verbose) printf("%s::%s\n", __FILE__, PFN);

	return * new NetworkSystemInterface(hba);
}
