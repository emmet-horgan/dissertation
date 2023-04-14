function tfd_func(dopp_win_len, lag_win_len, Ntime, Nfreq, path, k, n)
load("files");
dopp_win_len = double(dopp_win_len);
lag_win_len = double(lag_win_len);
Ntime = double(Ntime);
Nfreq = double(Nfreq);
CSV_TFD_DIR = fullfile(CSV_DATA_DIR, "csv-tfds");

if not(isfolder(CSV_TFD_DIR))
    mkdir(CSV_TFD_DIR);
end

CSV_TFD_DIR = fullfile(CSV_TFD_DIR, strcat("csv-tfd", "-", "K", "-", int2str(k), "-", "N", "-", int2str(n)));
if isfolder(CSV_TFD_DIR)
    exit("TFD ALREADY COMPLETED FOR THIS CONFIG");
end 
mkdir(CSV_TFD_DIR);
cd(path);

subjects = dir;
subjects = subjects(3:end); % Get rid of "." and ".." dirs

for itr = 1:length(subjects)
    
    here = pwd;
    cd(CSV_TFD_DIR);
    mkdir(subjects(itr).name);
    cd(here);
    
    cd(fullfile(subjects(itr).name));
    csv_files = dir;
    csv_files = csv_files(3:end);
    for jtr = 1:length(csv_files)
        tf = full_tfd(csvread(csv_files(jtr).name), "sep", {{dopp_win_len, "hann"}, {lag_win_len, "hann"}}, Ntime, Nfreq);
        here = pwd;
        cd(fullfile(CSV_TFD_DIR, subjects(itr).name))
        writematrix(tf, strcat("segment", int2str(jtr - 1), ".csv"));
        cd(here);
    end 
    cd(path);
end 