DMA Descriptor:

Field	Bit	Description
DSCRPT_ADDR	31 downto 10	Current Descriptor Address in Host Memory
DSCRPT_INT_ENABLE	8	Current Descriptor Interrupt Enable
DSCRPT_LEN	7 downto 4	Current Descriptor Length see below
DSCRPT_ID	3 downto 0	Current Descriptor ID

The quantity of data transferred for each descriptor is based on the Descriptor Length field. The definition of this field is dependent on whether the LOW_LATENCY flag is set (See DMA Control Register x"2004") as follows
Normal Streaming Mode - LOW_LATENCY = 0
Size Transferred = 2^( DSCRPT_LEN) * 1 Kbyte
For Example if a 64K transfer is required per descriptor, 64 = 2^6 therefore the Descriptor Length is set to 6.
This format allows buffers up to 2^15 = 32MBytes to be used

Note that current driver uses slab-allocated contiguous kernel buffers, maximu size 4MB. Other allocation techniques ("Reserved Memory") may allow use of larger buffers in the future.

Low Latency Mode - LOW_LATENCY = 1
Size Transferred = ( DSCRPT_LEN + 1) * 64 bytes
For Example if a 192 Byte transfer is required per descriptor, 192 = 64x3 therefore the Descriptor length is set to 2. This format allows multiples of 64 byte quantities to be sent, this equates to a 32 channel block of data at 2 bytes per channel
Note the LOW_LATENCY trades off bus efficiency to improve latency. The packet size in Normal Streaming Mode is matched to the Root Complex setting and is at least 128 bytes. In Low Latency Mode this is reduced to 64 bytes which drops the packet efficiency. However since all the FIFOs in the chain are set to a 64 byte packet the data starts to flow as soon as a packet is available increasing the overlap between the FMC Module writing data and the Communications Module sending it out improving the latency.

Summary of sizes:

HTS:  1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32786 KB

LLC:  64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024 BYTES


