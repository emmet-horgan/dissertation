function hampel_filter_func(k, sigma, src_path)
k = double(k);
sigma = double(sigma);
load("files");
CSV_HAMPEL_DIR = fullfile(CSV_DATA_DIR, strcat(CSV_HAMPEL, "-", "K", "-", int2str(k), "-", "N", "-", int2str(sigma)));
cd(src_path);

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
        hampel_result = hampel(csvread(csv_files(jtr).name), k, sigma);
        here = pwd;
        cd(hampel_data_dir);
        name = split(csv_files(jtr).name, ".");
        name = name(1);
        writematrix(hampel_result, strcat(name, "-", "hampel", "-", "K", int2str(k), "N", int2str(sigma), ".csv"));
        cd(here);
    end
    copyfile(fullfile(pwd, "..", "rr_peaks"), hampel_peaks_dir);
    cd(CSV_SPLIT_DIR);
end
cd(LIB_DIR);    
end 