/*
 * InlineDataHandler.cpp
 *
 *  Created on: 3 Aug 2020
 *      Author: pgm
 */

#include "InlineDataHandler.h"

InlineDataHandler::InlineDataHandler() {
	// TODO Auto-generated constructor stub

}

InlineDataHandler::~InlineDataHandler() {
	// TODO Auto-generated destructor stub
}


InlineDataHandler* InlineDataHandler::factory(RTM_T_Device* ai_dev)
{
	return new InlineDataHandler;
}

