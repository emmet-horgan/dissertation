clear;
% Directory Names
FYP = "fyp";
FYP_DATA = "fyp-data";
FYP_LIB = "fyp-lib";
CSV_DATA = "csv-data";
CSV_SPLIT = "csv-split";
CSV_HAMPEL = "csv-hampel";
%Drive var is windows specific, remove if using Unix based platform
DRIVE = split(pwd, filesep);
DRIVE = DRIVE(1);

K = 3; % Window length either side of centre. (DEFAULT=3 i.e. length=7)
N_SIGMA = 3; % Scale factor to multiply the MAD. (DEFAULT=3)

% Directory Paths
LIB_DIR = fullfile(DRIVE, getenv("HOMEPATH"), FYP, FYP_LIB);
DATA_DIR = fullfile(DRIVE, getenv("HOMEPATH"), FYP, FYP_DATA);
CSV_DATA_DIR = fullfile(DATA_DIR, CSV_DATA);
CSV_SPLIT_DIR = fullfile(CSV_DATA_DIR, CSV_SPLIT);
CSV_HAMPEL_DIR = fullfile(CSV_DATA_DIR, strcat(CSV_HAMPEL, "-", "K", "-", int2str(K), "-", "N", "-", int2str(N_SIGMA)));
save("files");

if ~strcmp(LIB_DIR, pwd)
    disp("WARNING: Wrong directory, changing to %HOME%/fyp/fyp-lib");
    cd(LIB_DIR);
end 

cd(CSV_DATA_DIR);

if 7 == exist(CSV_HAMPEL_DIR, "dir")
    cd(LIB_DIR);
    error("The hampel filtering result directory already exists");
    
else
    mkdir(CSV_HAMPEL_DIR);
end
cd(CSV_SPLIT_DIR);

subjects = dir;
subjects = subjects(3:end); % Get rid of "." and ".." dirs 

for itr = 1:length(subjects)
    cd(fullfile(subjects(itr).name, "rr_interval"));
    csv_files = dir;
    csv_files = csv_files(3:end);
    
    hampel_data_dir = fullfile(CSV_HAMPEL_DIR, subjects(itr).name, "rr_interval");
    mkdir(hampel_data_dir);
    hampel_peaks_dir = fullfile(CSV_HAMPEL_DIR, subjects(itr).name, "rr_peaks");
    mkdir(hampel_peaks_dir);
    
    for jtr = 1:length(csv_files)
        hampel_result = hampel(csvread(csv_files(jtr).name), K, N_SIGMA);
        here = pwd;
        cd(hampel_data_dir);
        name = split(csv_files(jtr).name, ".");
        name = name(1);
        writematrix(hampel_result, strcat(name, "-", "hampel", "-", "K", int2str(K), "N", int2str(N_SIGMA), ".csv"));
        cd(here);
    end
    copyfile(fullfile(pwd, "..", "rr_peaks"), hampel_peaks_dir);
    cd(CSV_SPLIT_DIR);
end
cd(LIB_DIR);