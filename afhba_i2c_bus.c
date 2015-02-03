/* ------------------------------------------------------------------------- */
/* afhba_i2c_bus.c AFHBA						     */
/*
 * afhba_i2c_bus.c
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


#include <linux/i2c.h>
#include <linux/i2c-gpio.h>
#include <linux/i2c-algo-bit.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/platform_device.h>

//#include <linux/slab.h>
#include "acq-fiber-hba.h"

struct AFHBA_DEV;	/* opaque structure */

/* we abuse the platform_data resource field by hiding a AFHBA_DEV* in it */
#define GETD(pdata)	((struct AFHBA_DEV*)(pdata+1)->resource)


int hba_sfp_get_value(struct AFHBA_DEV *adev, unsigned gpio)
{
	u32 reg = afhba_read_reg(adev, SFP_I2C_DATA_REG);
	int bit = (reg&(1 << gpio)) != 0;
	dev_dbg(pdev(adev), "read:%d", bit);
	return bit;
}
void hba_sfp_set_value(struct AFHBA_DEV *adev, unsigned gpio, int value)
{
	u32 reg = afhba_read_reg(adev, SFP_I2C_DATA_REG);

	dev_dbg(pdev(adev), "write:%d", value);
	if (value){
		reg |= 1 << (gpio+8);
	}else{
		reg &= ~(1 << (gpio+8));
	}
	afhba_write_reg(adev, SFP_I2C_DATA_REG, reg);
}


static inline int hba_sfp_request(struct AFHBA_DEV *adev, unsigned hba_sfp, const char *label)
{
        return 0;
}

static inline void hba_sfp_free(struct AFHBA_DEV *adev, unsigned hba_sfp)
{

}

static inline int hba_sfp_direction_input(struct AFHBA_DEV *adev, unsigned hba_sfp)
{
        return hba_sfp_get_value(adev, hba_sfp);
}
static inline int hba_sfp_direction_output(struct AFHBA_DEV *adev, unsigned hba_sfp, int value)
{
        hba_sfp_set_value(adev, hba_sfp, value);
        return value;
}

/* Toggle SDA by changing the direction of the pin */
static void i2c_hba_sfp_setsda_dir(void *dev, int state)
{
	struct platform_device *pdev = (struct platform_device *)dev;
	struct i2c_gpio_platform_data *pdata = pdev->dev.platform_data;
	if (state)
		hba_sfp_direction_input(GETD(pdev), pdata->sda_pin);
	else
		hba_sfp_direction_output(GETD(pdev), pdata->sda_pin, 0);
}

/*
 * Toggle SDA by changing the output value of the pin. This is only
 * valid for pins configured as open drain (i.e. setting the value
 * high effectively turns off the output driver.)
 */
static void i2c_hba_sfp_setsda_val(void *dev, int state)
{
	struct platform_device *pdev = (struct platform_device *)dev;
	struct i2c_gpio_platform_data *pdata = pdev->dev.platform_data;

	hba_sfp_set_value(GETD(pdev), pdata->sda_pin, state);
}

/* Toggle SCL by changing the direction of the pin. */
static void i2c_hba_sfp_setscl_dir(void *dev, int state)
{
	struct platform_device *pdev = (struct platform_device *)dev;
	struct i2c_gpio_platform_data *pdata = pdev->dev.platform_data;


	if (state)
		hba_sfp_direction_input(GETD(pdev), pdata->scl_pin);
	else
		hba_sfp_direction_output(GETD(pdev), pdata->scl_pin, 0);
}

/*
 * Toggle SCL by changing the output value of the pin. This is used
 * for pins that are configured as open drain and for output-only
 * pins. The latter case will break the i2c protocol, but it will
 * often work in practice.
 */
static void i2c_hba_sfp_setscl_val(void *dev, int state)
{
	struct platform_device *pdev = (struct platform_device *)dev;
	struct i2c_gpio_platform_data *pdata = pdev->dev.platform_data;

	hba_sfp_set_value(GETD(pdev), pdata->scl_pin, state);
}

int i2c_hba_sfp_getsda(void *dev)
{
	struct platform_device *pdev = (struct platform_device *)dev;
	struct i2c_gpio_platform_data *pdata = pdev->dev.platform_data;

	return hba_sfp_get_value(GETD(pdev), pdata->sda_pin);
}

int i2c_hba_sfp_getscl(void *dev)
{
	struct platform_device *pdev = (struct platform_device *)dev;
	struct i2c_gpio_platform_data *pdata = pdev->dev.platform_data;

	return hba_sfp_get_value(GETD(pdev), pdata->scl_pin);
}

static int i2c_hba_sfp_probe(struct platform_device *pdev)
{
	struct i2c_gpio_platform_data *pdata;
	struct i2c_algo_bit_data *bit_data;
	struct i2c_adapter *adap;
	int ret;

	pdata = pdev->dev.platform_data;
	if (!pdata)
		return -ENXIO;

	ret = -ENOMEM;
	adap = kzalloc(sizeof(struct i2c_adapter), GFP_KERNEL);
	if (!adap)
		goto err_alloc_adap;
	bit_data = kzalloc(sizeof(struct i2c_algo_bit_data), GFP_KERNEL);
	if (!bit_data)
		goto err_alloc_bit_data;

	ret = hba_sfp_request(GETD(pdev), pdata->sda_pin, "sda");
	if (ret)
		goto err_request_sda;
	ret = hba_sfp_request(GETD(pdev), pdata->scl_pin, "scl");
	if (ret)
		goto err_request_scl;

	if (pdata->sda_is_open_drain) {
		hba_sfp_direction_output(GETD(pdev), pdata->sda_pin, 1);
		bit_data->setsda = i2c_hba_sfp_setsda_val;
	} else {
		hba_sfp_direction_input(GETD(pdev), pdata->sda_pin);
		bit_data->setsda = i2c_hba_sfp_setsda_dir;
	}

	if (pdata->scl_is_open_drain || pdata->scl_is_output_only) {
		hba_sfp_direction_output(GETD(pdev), pdata->scl_pin, 1);
		bit_data->setscl = i2c_hba_sfp_setscl_val;
	} else {
		hba_sfp_direction_input(GETD(pdev), pdata->scl_pin);
		bit_data->setscl = i2c_hba_sfp_setscl_dir;
	}

	if (!pdata->scl_is_output_only)
		bit_data->getscl = i2c_hba_sfp_getscl;
	bit_data->getsda = i2c_hba_sfp_getsda;

	if (pdata->udelay)
		bit_data->udelay = pdata->udelay;
	else if (pdata->scl_is_output_only)
		bit_data->udelay = 50;			/* 10 kHz */
	else
		bit_data->udelay = 5;			/* 100 kHz */

	if (pdata->timeout)
		bit_data->timeout = pdata->timeout;
	else
		bit_data->timeout = HZ / 10;		/* 100 ms */

	//bit_data->data = pdata;
	bit_data->data = pdev;

	adap->owner = THIS_MODULE;
	snprintf(adap->name, sizeof(adap->name), "i2c_hba_sfp%d", pdev->id);
	adap->algo_data = bit_data;
	adap->dev.parent = &pdev->dev;

	ret = i2c_bit_add_bus(adap);
	if (ret)
		goto err_add_bus;

	platform_set_drvdata(pdev, adap);

	dev_info(&pdev->dev, "using pins %u (SDA) and %u (SCL%s)\n",
		 pdata->sda_pin, pdata->scl_pin,
		 pdata->scl_is_output_only
		 ? ", no clock stretching" : "");

	return 0;

err_add_bus:
	hba_sfp_free(GETD(pdev), pdata->scl_pin);
err_request_scl:
	hba_sfp_free(GETD(pdev), pdata->sda_pin);
err_request_sda:
	kfree(bit_data);
err_alloc_bit_data:
	kfree(adap);
err_alloc_adap:
	return ret;
}

static int __exit i2c_hba_sfp_remove(struct platform_device *pdev)
{
	struct i2c_gpio_platform_data *pdata;
	struct i2c_adapter *adap;

	adap = platform_get_drvdata(pdev);
	pdata = pdev->dev.platform_data;

	i2c_del_adapter(adap);
	hba_sfp_free(GETD(pdev), pdata->scl_pin);
	hba_sfp_free(GETD(pdev), pdata->sda_pin);
	kfree(adap->algo_data);
	kfree(adap);

	return 0;
}

static struct platform_driver i2c_hba_sfp_driver = {
	.driver		= {
		.name	= "i2c_hba_sfp",
		.owner	= THIS_MODULE,
	},
	.remove	= __exit_p(i2c_hba_sfp_remove),
	.probe  = i2c_hba_sfp_probe
};


static struct i2c_gpio_platform_data rtm_t_i2c_gpio_data_templates[] = {
	{
		.sda_pin	= SFP_I2C_SDA1_R,
		.scl_pin	= SFP_I2C_SCL1_R,
		.udelay 	= 1,		/* AVO min time=1usec */
		.sda_is_open_drain = 1,
		.scl_is_open_drain = 1
	},
	{
		.sda_pin	= SFP_I2C_SDA2_R,
		.scl_pin	= SFP_I2C_SCL2_R,
		.udelay 	= 1,		/* AVO min time=1usec */
		.sda_is_open_drain = 1,
		.scl_is_open_drain = 1
	}
};

#define CHIX(ch)	((ch)==2)	/* 2=>1 *=>0 */

static int afhba_sfp_i2c_create1(struct AFHBA_DEV *adev, int ch)
{
	int rc;
	struct platform_device* pd =
			kzalloc(2*sizeof(struct platform_device), GFP_KERNEL);
	pd->name = "i2c_hba_sfp";
	pd->id = adev->idx*2+CHIX(ch);
	pd->dev.platform_data = &rtm_t_i2c_gpio_data_templates[CHIX(ch)];
	/** HACK ALERT */
	pd[1].resource = (struct resource*)adev;
	rc = platform_device_register(pd);

	if (rc != 0){
		dev_err(pdev(adev), "platform device register failed");
		kfree(pd);
	}else{
		adev->hba_sfp_i2c[CHIX(ch)] = pd;
	}
	return rc;
}

static void afhba_sfp_i2c_remove1(struct AFHBA_DEV *adev, int ch)
{
	struct platform_device* pd = adev->hba_sfp_i2c[CHIX(ch)];
	if (pd){
		adev->hba_sfp_i2c[CHIX(ch)] = 0;
		platform_device_unregister(pd);
		kfree(pd);
	}
}

int afhba_sfp_i2c_create(struct AFHBA_DEV *adev)
{
	afhba_sfp_i2c_create1(adev, 1);
	afhba_sfp_i2c_create1(adev, 2);
	return 0;
}

int afhba_sfp_i2c_remove(struct AFHBA_DEV *adev)
{
	afhba_sfp_i2c_remove1(adev, 1);
	afhba_sfp_i2c_remove1(adev, 2);
	return 0;
}

static int afhba_sfp_init_devices(void)
/* proper use of kernel device model would avoid this */
{
	struct AFHBA_DEV *pos;
	int device_count = 0;

	list_for_each_entry(pos, &afhba_devices, list){
		afhba_sfp_i2c_create(pos);
		++device_count;
	}
	return device_count;
}

static int afhba_sfp_remove_devices(void)
/* proper use of kernel device model would avoid this */
{
	struct AFHBA_DEV *pos;
	int device_count = 0;

	list_for_each_entry(pos, &afhba_devices, list){
		afhba_sfp_i2c_remove(pos);
		++device_count;
	}
	return device_count;
}

static int afhba_i2c_sfp_init(void)
{
	int ret;

	//ret = platform_driver_probe(&i2c_hba_sfp_driver, i2c_hba_sfp_probe);
	ret = platform_driver_register(&i2c_hba_sfp_driver);
	if (ret)
		printk(KERN_ERR "i2c-hba_sfp: probe failed: %d\n", ret);
	afhba_sfp_init_devices();

	return ret;
}
module_init(afhba_i2c_sfp_init);

static void afhba_i2c_sfp_exit(void)
{
	afhba_sfp_remove_devices();
	platform_driver_unregister(&i2c_hba_sfp_driver);
}
module_exit(afhba_i2c_sfp_exit);

MODULE_AUTHOR("Peter Milne <peter.milne@d-tacq.com");
MODULE_DESCRIPTION("ACQ-FIBER-HBA I2C driver");
MODULE_LICENSE("GPL");
