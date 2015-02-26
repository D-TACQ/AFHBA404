/* ------------------------------------------------------------------------- */
/* afhba_spi.c AFHBA						     */
/*
 * afhba_spi.c
 *
 *  Created on: 24 Jan 2015
 *      Author: pgm
 */

/* ------------------------------------------------------------------------- */
/*   Copyright (C) 2015 Peter Milne, D-TACQ Solutions Ltd                    *
 *                      <peter dot milne at D hyphen TACQ dot com>           *
 *                                                                           *
 *  This program is free software; you can redistribute it and/or modify     *
 *  it under the terms of Version 2 of the GNU General Public License        *
 *  as published by the Free Software Foundation;                            *
 *                                                                           *
 *  This program is distributed in the hope that it will be useful,          *
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of           *
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the            *
 *  GNU General Public License for more details.                             *
 *                                                                           *
 *  You should have received a copy of the GNU General Public License        *
 *  along with this program; if not, write to the Free Software              *
 *  Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.                */
/* ------------------------------------------------------------------------- */

#include <linux/device.h>
#include <linux/delay.h>
#include <linux/interrupt.h>
#include <linux/kernel.h>
#include <linux/pci.h>
#include <linux/time.h>
#include <linux/init.h>
#include <linux/timex.h>
#include <linux/vmalloc.h>
#include <linux/mm.h>

#include <linux/module.h>
#include <linux/moduleparam.h>
/*
#include <asm/mach/flash.h>
*/
#include <linux/spi/flash.h>
#include <linux/spi/spi.h>
#include <linux/spi/spi_bitbang.h>

#include "acq-fiber-hba.h"

int afhba_spi_debug;
module_param(afhba_spi_debug, int, 0644);

int spi_write_msleep = 1;
module_param(spi_write_msleep, int, 0664);


struct afhba_spi {
	/* bitbang has to be first ?why? */
	struct spi_bitbang	bitbang;

	/* data buffers */
	const unsigned char	*tx;
	unsigned char		*rx;
	int			len;
	int			count;
	struct spi_master	*master;
	struct device		*dev;
	struct AFHBA_DEV 	*adev;
	u32 ctrl_mirror;
	int sent_reset;
};

static inline void afhba_spi_write_ctl(struct afhba_spi *hw, u32 value)
{
	struct AFHBA_DEV *adev = hw->adev;
	hw->ctrl_mirror = value;

	dev_dbg(pdev(adev), "\t%04x := 0x%08x)", HOST_SPI_FLASH_CONTROL_REG, value);
	afhba_write_reg(adev, HOST_SPI_FLASH_CONTROL_REG, hw->ctrl_mirror);
}

static inline u32 afhba_spi_read_ctl(struct afhba_spi *hw)
{
	struct AFHBA_DEV *adev = hw->adev;
	u32 cc = afhba_read_reg(adev, HOST_SPI_FLASH_CONTROL_REG);

	dev_dbg(pdev(adev), "\t%04x => 0x%08x", HOST_SPI_FLASH_CONTROL_REG, cc);
	return cc;
}

static void afhba_spi_wait_busy(struct afhba_spi *hw)
{
	struct AFHBA_DEV *adev = hw->adev;
	int pollcat = 0;

	dev_dbg(pdev(adev), "start..");
	while((afhba_spi_read_ctl(hw)&AFHBA_SPI_BUSY) != 0){
		yield();
		//msleep(1);
		++pollcat;
	}
	dev_dbg(pdev(adev), "done in %d", pollcat);
}

static void afhba_spi_start(struct afhba_spi *hw, int on)
{
	struct AFHBA_DEV *adev = hw->adev;
	if (on){
		hw->ctrl_mirror |= AFHBA_SPI_CTL_START;
	}else{
		hw->ctrl_mirror &= ~AFHBA_SPI_CTL_START;
	}
	dev_dbg(pdev(adev), "%08x %s", hw->ctrl_mirror, on? "ON": "OFF");
	afhba_spi_write_ctl(hw, hw->ctrl_mirror);
}
static inline void afhba_spi_write(struct afhba_spi *hw, unsigned char cc)
{
	struct AFHBA_DEV *adev = hw->adev;
	dev_dbg(pdev(adev), "\t%04x := %02x", HOST_SPI_FLASH_DATA_REG, cc);

	afhba_write_reg(hw->adev, HOST_SPI_FLASH_DATA_REG, cc);
	afhba_spi_start(hw, 1);
	afhba_spi_wait_busy(hw);
	afhba_spi_start(hw, 0);
}

static inline unsigned char afhba_spi_read(struct afhba_spi *hw)
{
	struct AFHBA_DEV *adev = hw->adev;
	unsigned char cc;

	afhba_spi_start(hw, 1);
	afhba_spi_wait_busy(hw);
	afhba_spi_start(hw, 0);
	cc = afhba_read_reg(hw->adev, HOST_SPI_FLASH_DATA_REG);
	dev_dbg(pdev(adev), "\t\tread(%04x) => %02x", HOST_SPI_FLASH_DATA_REG, cc);

	return cc;
}

static inline struct afhba_spi *to_hw(struct spi_device *sdev)
{
	return spi_master_get_devdata(sdev->master);
}

static inline unsigned int hw_txbyte(struct afhba_spi *hw, int count)
{
	return hw->tx ? hw->tx[count] : 0;
}

static void rtm_t_spi_chipsel(struct spi_device *spi, int value)
{
	unsigned int cspol = spi->mode & SPI_CS_HIGH ? 1 : 0;
	struct afhba_spi *hw = to_hw(spi);
	struct AFHBA_DEV *adev = hw->adev;

	dev_dbg(pdev(adev), "value %d", value);
	if (afhba_spi_debug >= 2){
		afhba_spi_read_ctl(hw);
	}

	switch(value){
	case BITBANG_CS_INACTIVE:
		afhba_spi_write_ctl(hw, cspol? 0: AFHBA_SPI_CS);
		break;
	case BITBANG_CS_ACTIVE:
		afhba_spi_write_ctl(hw, cspol? AFHBA_SPI_CS: 0);
		break;
	}

	if (afhba_spi_debug >= 2){
		afhba_spi_read_ctl(hw);
	}
}
static int rtm_t_spi_setupxfer(struct spi_device *spi,
				 struct spi_transfer *t)
{
	struct afhba_spi *hw = to_hw(spi);

	spin_lock(&hw->bitbang.lock);
	if (!hw->bitbang.busy) {
		hw->bitbang.chipselect(spi, BITBANG_CS_INACTIVE);
	}
	spin_unlock(&hw->bitbang.lock);
	return 0;
}

static int rtm_t_spi_setup(struct spi_device *spi)
{
	int rc;


	if (spi->bits_per_word != 8){
		spi->bits_per_word = 8;
	}
	if ((spi->mode & SPI_LSB_FIRST) != 0)
		return -EINVAL;

	rc = rtm_t_spi_setupxfer(spi, NULL);
	return rc;
}


static int rtm_t_spi_txrx(struct spi_device *spi, struct spi_transfer *t)
{
	struct afhba_spi *hw = to_hw(spi);
	struct AFHBA_DEV *adev = hw->adev;

	dev_dbg(pdev(adev), "txrx: tx %p, rx %p, len %d",
		t->tx_buf, t->rx_buf, t->len);

	hw->tx = t->tx_buf;
	hw->rx = t->rx_buf;

	if (hw->tx != NULL && hw->rx != NULL){
		dev_err(pdev(adev), "tx and rx requested. This is NOT good!");
	}

	hw->len = t->len;
	hw->count = 0;

	if (hw->tx){
	/* send the first byte */
		afhba_spi_write(hw, hw_txbyte(hw, 0));

		dev_dbg(pdev(adev), "02: done first write now while %d < %d",
			hw->count, hw->len);
		++hw->count;
	}

	for (; hw->count < hw->len; ++hw->count){
		//msleep(spi_write_msleep);
		yield();

		if (hw->rx){
			hw->rx[hw->count] = afhba_spi_read(hw);
		}

		if (hw->tx){
			afhba_spi_write(hw, hw_txbyte(hw, hw->count));
		}
	}

	dev_dbg(pdev(adev), "return %d\n",hw->count);
	return hw->count;
}


/* arm style
static struct flash_platform_data rtm_t_flash_data = {
	.map_name		= "jedec_probe",
	.width			= 2,
	.type		        = "m25p64"
};
*/
static struct flash_platform_data afhba_flash_data = {
	.name			= "afhba-flash",
	.type			= "w25q64"
};
static struct spi_board_info rtm_t_spi_devices[] = {
	{	/* DataFlash chip */
//		.modalias	= "mtd_dataflash",
//		.modalias	= "jedec_probe",
		.modalias	= "m25p80",
		.chip_select	= 0,
		.max_speed_hz	= 15 * 1000 * 1000,
		.mode = SPI_CS_HIGH,
		.platform_data = &afhba_flash_data
	},
};

int afhba_spi_master_init(struct AFHBA_DEV *adev)
{
	struct device *dev = &adev->pci_dev->dev;
	struct spi_master *master;
	int rc = 0;
	struct afhba_spi *hw;

	dev_info(dev, "call spi_alloc_master, dev %p", dev);

	master = spi_alloc_master(dev, sizeof(struct afhba_spi));
	if (master == NULL){
		dev_err(dev, "No memory for spi_master\n");
		rc = -ENOMEM;
		goto err_nomem;
	}
	hw = spi_master_get_devdata(master);
	memset(hw, 0, sizeof(struct afhba_spi));

	hw->master = spi_master_get(master);
	hw->master->bus_num = adev->idx;
//	hw->pdata = pdev->dev.platform_data;
	hw->dev = dev;
	hw->adev = adev;

	hw->bitbang.master         = hw->master;
	hw->bitbang.setup_transfer = rtm_t_spi_setupxfer;
	hw->bitbang.chipselect     = rtm_t_spi_chipsel;
	hw->bitbang.txrx_bufs      = rtm_t_spi_txrx;
	hw->bitbang.master->setup  = rtm_t_spi_setup;
	hw->bitbang.flags = SPI_CS_HIGH;

	/* register our spi controller */

	dev_dbg(dev, "call spi_bitbang_start()");
	rc = spi_bitbang_start(&hw->bitbang);
	dev_dbg(dev, "back from spi_bitbang_start()");

	if (rc) {
		dev_err(dev, "Failed to register SPI master\n");
		goto err_register;
	}

	dev_dbg(pdev(adev), "registering %s\n", rtm_t_spi_devices[0].modalias);
	spi_new_device(master, &rtm_t_spi_devices[0]);

	return 0;

err_register:

err_nomem:
	return rc;
}

void afhba_spi_master_remove(struct AFHBA_DEV *adev)
{
	dev_warn(pdev(adev), "stub");
}

static int afhba_spi_init_devices(void)
/* proper use of kernel device model would avoid this */
{
	struct AFHBA_DEV *pos;
	int device_count = 0;

	list_for_each_entry(pos, &afhba_devices, list){
		afhba_spi_master_init(pos);
		++device_count;
	}
	return device_count;
}

static int afhba_spi_remove_devices(void)
/* proper use of kernel device model would avoid this */
{
	struct AFHBA_DEV *pos;
	int device_count = 0;

	list_for_each_entry(pos, &afhba_devices, list){
		afhba_spi_master_remove(pos);
		++device_count;
	}
	return device_count;
}


static int afhba_spi_init(void)
{
	/* no platform device driver */
	afhba_spi_init_devices();

	return 0;
}
module_init(afhba_spi_init);

static void afhba_spi_exit(void)
{
	afhba_spi_remove_devices();
}
module_exit(afhba_spi_exit);

MODULE_AUTHOR("Peter Milne <peter.milne@d-tacq.com");
MODULE_DESCRIPTION("ACQ-FIBER-HBA SPI flash driver");
MODULE_LICENSE("GPL");

