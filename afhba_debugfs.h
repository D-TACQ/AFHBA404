/*
 * afhba_debugfs.h
 *
 *  Created on: 13 Aug 2014
 *      Author: pgm
 */

#ifndef AFHBA_DEBUGFS_H_
#define AFHBA_DEBUGFS_H_

void afhba_createDebugfs(struct AFHBA_DEV* adev);
void afhba_removeDebugfs(struct AFHBA_DEV* adev);

extern const char* afhba_devnames[];

#endif /* AFHBA_DEBUGFS_H_ */
