# afhba-bufferAB  collect LL data in two buffers.
## requirement : 32ch+16spad x 512kSPS, 1 buffer per msec
## 128bytes/sample, 512 samples/buffer => bufferlent = 65536

### on the UUT
```
acq2106_130> set.site 13 spad=1
acq2106_130> set.site 0 sync_role master 512000
acq2106_130> set.site 0 bufferlen 65536
acq2106_130> set.site 0 spad=1,16,1
acq2106_130> /usr/local/CARE/make_spad_id
acq2106_130> run0 1
acq2106_130> set.site 13 aggregator sites=1
acq2106_130> set.site 13 spad=1
```

### on the HOST
```
[root@hoth AFHBA404]# RTPRIO=10 SPADLONGS=16 NCHAN=32 ./LLCONTROL/afhba-bufferAB 100000 512
NCHAN set 32
SPADLONGS set 16
AI buf pa: A 0x00000000 len 65536
AI buf pa: B 0x00040000 len 65536
AI buf pa: A 0xcf000000 len 65536
AI buf pa: B 0xcf040000 len 65536
ready for data
finished
```

### now start the capture 
```
acq2106_130> set.site 0 CONTINUOUS 1
sleep 10
acq2106_130> set.site 0 CONTINUOUS 0
```

### analyse the data
```
[pgm@hoy5 acq400_hapi]$  ./test_apps/t_latch_histogram.py  --nchan 32 --spad_len=16 --src=afhba.0.log
Finished collecting data
T_LATCH differences:  1 , happened:  100352  times
T_LATCH differences:  2 , happened:  0  times
T_LATCH differences:  3 , happened:  0  times
```

Also plots the histogram

```
[root@hoth AFHBA404]# hexdump -e '32/2 "%04x," 16/4 "%08x," "\n"' afhba.0.log | cut -d, -f1-4,33-40 | more
cf1a,fff2,fffe,fff8,00000001,00000000,00000000,33333333,44444444,55555555,66666666,77777777
cf11,fff2,fffe,fff6,00000002,20000000,22222222,33333333,44444444,55555555,66666666,77777777
cf04,fff4,fffe,fff6,00000003,20000000,22222222,33333333,44444444,55555555,66666666,77777777
cef8,fff2,ffff,fff8,00000004,20000000,22222222,33333333,44444444,55555555,66666666,77777777
cee8,fff1,fffe,fff7,00000005,20000000,22222222,33333333,44444444,55555555,66666666,77777777
cee0,fff2,fffe,fff7,00000006,20000000,22222222,33333333,44444444,55555555,66666666,77777777
ced3,fff3,ffff,fff8,00000007,20000000,22222222,33333333,44444444,55555555,66666666,77777777
cecb,fff3,fffe,fff9,00000008,20000000,22222222,33333333,44444444,55555555,66666666,77777777
ceb8,fff2,fffe,fff9,00000009,20000000,22222222,33333333,44444444,55555555,66666666,77777777
ceac,fff3,fffd,fffa,0000000a,20000000,22222222,33333333,44444444,55555555,66666666,77777777
cea2,fff2,fffe,fff8,0000000b,20000000,22222222,33333333,44444444,55555555,66666666,77777777
ce97,fff2,fffe,fff7,0000000c,20000000,22222222,33333333,44444444,55555555,66666666,77777777
ce89,fff2,fffe,fff8,0000000d,20000000,22222222,33333333,44444444,55555555,66666666,77777777
ce7f,fff3,fffe,fff7,0000000e,20000000,22222222,33333333,44444444,55555555,66666666,77777777
```



