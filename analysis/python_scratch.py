#!/usr/bin/env python3

    # %% HRH halfway version

    ps_fid = []
    no_psi_fid = []
    psi = []
    for nevent, evt in enumerate(ds.events()):
        evtId = evt.get(psana.EventId)
        tval = evtId.time()
        fid = evtId.fiducials()
        ps_id = [tval[0], tval[1], fid]
        try:
            psi_val = psi_det.get(evt).TotalIntensity()
        except:
            no_psi_fid.append(ps_id)
            psi_val = -1

        ps_fid.append(ps_id)
        psi.append(psi_val)

        # print("Time: " + str(evtId.time()))
        # print("Fid:" + str(evtId.fiducials()))
        # print("\tTotal intensity: " + str(psi_det.get(evt).TotalIntensity()))

    ps_fid = np.array(ps_fid)
    psi = np.array(psi)

    print(str(len(no_psi_fid)) + " events had no Wave8, so recorded as -1")
    print(f"Matching Wave8 post-sample-intensities")

    # this line makes the om frame IDs a dictionary for easy look up
    ps_fid_dict = {tuple(value): idx for idx, value in enumerate(ps_fid)}
    idx = [ps_fid_dict[tuple(value)]
            for value in radials_dict["frame_id"] if tuple(value) in ps_fid_dict]

    # double check that the reordered OM arrays match the reborn arrays for frame id
    # print("OM: ", om_fid[idx][10])
    # print("RE: ", re_fid[10])
