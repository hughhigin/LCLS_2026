import sys, shutil
import psana
import h5py
import numpy as np
import matplotlib.pyplot as plt

#usage: add_psana_acqiris_to_reborn_h5.py <run_number>

r = int(sys.argv[1])
run = f'{r:04d}'
detector = 'jungfrau'
# detector = 'epix'

#reborn h5 file
re_path = f'/sdf/data/lcls/ds/cxi/cxil1005322/results/hhig/practice_0603/results/reb_hdf5/cxil1005322_r{run}_{detector}_oldMasks.h5'

# Backup the reborn HDF5 file to be safe
backup_re_path = f'/sdf/data/lcls/ds/cxi/cxil1005322/results/hhig/practice_0603/results/reb_hdf5/cxil1005322_r{run}_{detector}_oldMasks_backup2.h5'

#already did this, don't need to do it again
shutil.copy(re_path, backup_re_path)

with h5py.File(re_path, 'a') as h5re:

    #get seconds, nanoseconds, fiducials from reborn
    re_fid = h5re["/frame_id"][:]
    seconds = re_fid[:,0]
    nanoseconds = re_fid[:,1]
    fiducials = re_fid[:,2]

    nevents = len(seconds)

    #create waveforms array of length nevents from reborn.
    waveforms = np.zeros((nevents, 4, 1000)) #waveform array is (4,1000) shape
    wftimes = np.zeros((nevents, 4, 1000))
    #create array for post_sample_intensity
    psi = np.zeros((nevents))
    j = 0

    #jumping to events by timestamp seemed slow, maybe because of using idx. 
    #Instead, lets just go through events in fastest order using smd, 
    #then match up the timestamp using numpy
    #first, make a dictionary of reborn frame ids:
    re_fid_dict = {tuple(value): idx for idx, value in enumerate(re_fid)}
    ds = psana.DataSource(f'exp=cxil1005322:run={r}:smd')
    det = psana.Detector('Acqiris')
    #also add in the post_sample_intensity (which is the OM terminology), which is CXI-DG2-BMMON
    psi_det = psana.Detector('CXI-DG2-BMMON')
    for nevent,evt in enumerate(ds.events()):
        evtId = evt.get(psana.EventId)
        sec = evtId.time()[0]
        ns = evtId.time()[1]
        fid = evtId.fiducials()
        #grab the reborn index associated with that timestamp
        idx = re_fid_dict[tuple((sec,ns,fid))]
        sys.stdout.write('\r%d/%d   %d'%(j,nevents, idx))
        sys.stdout.flush()
        # waveforms are in Volts, times are in Seconds
        waveforms[idx] = det.waveform(evt)
        wftimes[idx] = det.wftime(evt)
        try:
            psi[idx] = psi_det.get(evt).TotalIntensity()
        except:
            psi[idx] = -1
        j+=1
        # if j>100: break

    waveforms = np.array(waveforms)
    wftimes = np.array(wftimes)

    #the waveforms have an odd repeating pattern. We correct it here.
    offsets = np.load('waveform_offsets.npy')
    waveforms += offsets

    if '/Acqiris' in h5re:
        #Acqiris was our attempt at getting reborn to write the waveforms
        #directly to the h5 file when we made the radials, but I haven't gotten
        #that working yet. So for now delete those to avoid confusion
        del h5re['/Acqiris'] #delete it if it previously existed

    #write waveforms and times to reborn h5 file
    if '/waveforms' in h5re:
        del h5re['/waveforms'] #delete it if it previously existed
    h5re.create_dataset('/waveforms', data=waveforms)
    if '/wftimes' in h5re:
        del h5re['/wftimes'] #delete it if it previously existed
    h5re.create_dataset('/wftimes', data=wftimes)

    #write the reordered array to the reborn file
    if '/post_sample_intensity' in h5re:
        del h5re['/post_sample_intensity'] #delete it if it previously existed
    h5re.create_dataset('/post_sample_intensity', data=psi)

print()
