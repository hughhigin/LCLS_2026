import sys
import psana
import numpy as np

#usage: get_waveforms.py <run_number>

ds = psana.DataSource(f'exp=cxil1005322:run=7:smd')
det = psana.Detector('Acqiris')
nevents = 35982 #which I got from a previous loop
waveforms = np.zeros((nevents, 4, 1000)) #each waveform array is (4,1000) shape
wftimes = np.zeros((nevents, 4, 1000))
for nevent,evt in enumerate(ds.events()):
    sys.stdout.write('\r%d/%d'%(nevent,nevents))
    # sys.stdout.flush()
    # waveforms are in Volts, times are in Seconds
    waveforms[nevent] = det.waveform(evt)
    wftimes[nevent] = det.wftime(evt)
    if nevent>=nevents-1: break

print()


