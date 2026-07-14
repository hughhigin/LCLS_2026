%% Read in HDF5

% construct filename for desired experiment, run, and detector
exp_id = "cxi101672626"
h5_dir = "/path/to/h5s/";
run_num = 10;
detector = "jungfrau"

run_str = join(["r",sprintf("%04d",run_num)],"");
h5_file = join([exp_id, run_str, detector],"_") + ".h5";

% Display HDF5 metadata
h5disp(h5_dir + h5_file) % Displays datasets and names
h5_metadata = h5info(h5_dir + h5_file); % Struct of metadata

% Load only digitizer
digitizer = h5read(h5_dir + h5_file, "/digitizer");

% Create struct of limited fields
dataset_names = {'mean', 'dg2', 'digitizer'};

h5_struct = struct();
for ii = 1:numel(dataset_names)
    dataset = dataset_names{ii};
    h5_struct.(dataset) = h5read(h5_dir + h5_file, "/" + dataset);
end

% Load entire HDF5 into a struct - ASSUMES NO SUB-DATASETS
dataset_names = {h5_metadata.Datasets.Name};
h5_struct = struct();
for ii = 1:numel(dataset_names)
    dataset = dataset_names{ii};
    h5_struct.(dataset) = h5read(test_dir + test_h5, "/" + dataset);
end
