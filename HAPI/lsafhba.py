'''
lsafhba : list all afhba connections
Created on 18 Feb 2022

@author: pgm
'''

import argparse
import acq400_hapi
import json


def mtype(mod):
#    print("is_adc:{}".format(mod.is_adc))
    mt = "none"
    if mod.is_adc.split(" ")[0]=='1':
        mt = "AI"
    elif mod.MTYPE[0] == '4':
        mt = "AO"
    elif mod.MTYPE[0] == '6':
        mt = "DI"
    else:
        mt = "UN"
    return "{}{}".format(mt, 32 if mod.data32=='1' else 16)

def save_VI(cfg, uut):
    NC = { 'AI16': 0, 'AI32': 0, 'DI32': 0, 'SP32': 0 }
    for s in [int(s) for s in uut.sites]:
        mod = uut.modules[s]
        model = mod.MODEL.split(" ")[0]
        d32 = mod.data32 == '1'
        nchan = int(mod.NCHAN)
        is_adc = mod.is_adc.split(" ")[0] == '1'
        mt = mod.MTYPE
        
        
        if is_adc:
            NC["AI{}".format(32 if d32 else 16)] += nchan
        elif mt[0] == '6':
            NC['DI32'] += 1 
    
    len_vi = 0    
    for key, value in NC.items():
        if value > 0:
            len_vi += value * (2 if key == "AI16" else 4)
            cfg.write('"{}": {},\n'.format(key, value))
            
    sp32 = (16*4 - len_vi%64) // 4
    cfg.write('"{}": {}\n'.format("SP32", sp32))
    
def save_VO(cfg, uut):
    NC = { 'AO16': 0, 'AO32': 0, 'DO32': 0 }
    
    for s in [int(s) for s in uut.sites]:
        mod = uut.modules[s]
        model = mod.MODEL.split(" ")[0]
        d32 = mod.data32 == '1'
        nchan = int(mod.NCHAN)
        is_adc = mod.is_adc.split(" ")[0] == '1'
        mt = mod.MTYPE
        
        if is_adc:
            pass
        elif mt[0] == '4':
            NC["AO{}".format(32 if d32 else 16)] += nchan
        elif mt[0] == '6':
            NC['DO32'] += 1 
    
    len_vo = 0    
    for key, value in NC.items():
        if value > 0:
            len_vo += value * (2 if key == "AO16" else 4)
            cfg.write('"{}": {},\n'.format(key, value))
    aosites = uut.get_site_types()['AOSITES']
    if len(aosites) > 0:
        cfg.write('"{}": {},\n'.format('AOSITES', aosites))
    cfg.write('"{}": {}\n'.format("PAD32", 0))
    
    
def save_config(args, cfile, conns, uuts):
    with open(cfile, "w") as cfg:
        cfg.write('{\n')
        cfg.write('"AFHBA": {\n')
        cfg.write('"UUT": [\n')
        ii = 0
        for key, value in conns.items():
            cfg.write('{\n')          
            cfg.write('"DEVNUM": {},\n'.format(value.dev))
            cfg.write('"name": "{}",\n'.format(value.uut))
            cfg.write('"type": "pcs",\n')
            cfg.write('"sync_role": "{}",\n'.format("master" if ii==0 else "slave"))
            cfg.write('"COMMS": "{}",\n'.format(value.cx))
            
            cfg.write('"VI": {\n')
            save_VI(cfg, uuts[ii])            
            cfg.write('},\n')
            
            cfg.write('"VO": {\n')
            save_VO(cfg, uuts[ii]) 
            cfg.write('}\n')            

            cfg.write('{}{}\n'.format('}', ',' if ii < len(uuts)-1 else ''))
            ii += 1
            
        cfg.write(']\n')
        cfg.write('}\n')
        cfg.write('}\n')
            
            
            
        
def lsafhba(args):
    conns = acq400_hapi.afhba404.get_connections()
    uuts = []
    
    for key, value in conns.items():
       print("{} {}".format(key, value))
       uut = acq400_hapi.factory(value.uut)
       uuts.append(uut)                 
       sites = uut.sites
       if args.verbose == 0:
           continue
       print("\tpayload:{}".format(sites))
       for s in [int(s) for s in sites]:
            print("\t\tsite:{} MODULE {} {} {}".format(\
                 s, uut.modules[s].MODEL.split(" ")[0], \
                 mtype(uut.modules[s]), uut.modules[s].NCHAN))
            
    if args.save_config:
        rawfile = "{}.r".format(args.save_config)
        save_config(args, rawfile, conns, uuts)
        
        with open(rawfile, 'r') as fr:
            data = json.load(fr)
            with open(args.save_config, "w") as fp:
                fp.write(json.dumps(data, indent=4))

    
def run_main():
    parser = argparse.ArgumentParser(description='list all attached acq2x06 devices')
    parser.add_argument('--save_config', default=None, help='save configuration skeleton')
    parser.add_argument('--verbose', default=0, type=int, help='increase verbosity')
    
    lsafhba(parser.parse_args())
    
    
    
if __name__ == '__main__':
    run_main()
