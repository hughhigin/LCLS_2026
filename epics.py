from psana import *


ds = MPIDataSource('exp=xpptut15:run=59:smd')
epicsVarFullName = Detector('HX2:DVD:GCC:01:PMON')
epicsVarAlias = Detector('SampleTemp_GetA')

for nevent, evt in enumerate(ds.events()):
    print(epicsVarFullName())
    print(epicsVarAlias())
    break

