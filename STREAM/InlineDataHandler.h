/*
 * InlineDataHandler.h
 *
 *  Created on: 3 Aug 2020
 *      Author: pgm
 */

#ifndef STREAM_INLINEDATAHANDLER_H_
#define STREAM_INLINEDATAHANDLER_H_

class InlineDataHandler {
protected:
	InlineDataHandler();
	virtual ~InlineDataHandler();

public:
	virtual void handleBuffer(int ibuf, const void *src, int len) {}

	static InlineDataHandler* factory();
};

#endif /* STREAM_INLINEDATAHANDLER_H_ */
