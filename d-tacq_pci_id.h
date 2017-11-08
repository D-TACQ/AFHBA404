/*
 * d-tacq_pci_id.h
 *
 *  Created on: 28 Nov 2015
 *      Author: pgm
 */

#ifndef D_TACQ_PCI_ID_H_
#define D_TACQ_PCI_ID_H_

#define PCI_VENDOR_ID_XILINX      0x10ee
#define PCI_DEVICE_ID_XILINX_PCIE 0x0007
// D-TACQ changes the device ID to work around unwanted zomojo lspci listing */
#define PCI_DEVICE_ID_DTACQ_PCIE  0xadc1

#define PCI_SUBVID_DTACQ	0xd1ac
#define PCI_SUBDID_FHBA_2G	0x4100
#define PCI_SUBDID_FHBA_4G_OLD	0x4101
#define PCI_SUBDID_FHBA_4G	0x4102
#define PCI_SUBDID_FHBA_4G2	0x4103
#define PCI_SUBDID_FHBA_4G4	0x4104	/* AFHBA404 */
#define PCI_SUBDID_HBA_KMCU	0x4105	/* KMCU #1 Z7035 using one lane of 4 */
#define PCI_SUBDID_HBA_KMCU2	0x4106	/* KMCU #2 Z7030 single lane */

#define PCI_SUBDID_CPSC		0x4110
#define PCI_SUBDID_RTMT_AO	0x4080

#endif /* D_TACQ_PCI_ID_H_ */
