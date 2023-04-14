clear
load("files");
load(fullfile(DATA_DIR, "rr_peaks_HIE_grades_anon_v2", "rr_peaks_HIE_grades_anon_v2.mat"));

peaks = rr_peaks_st(109).rr_peaks;
recalculation = {};
for itr = 2:length(peaks)
    recalculation = [recalculation, peaks(itr) - peaks(itr-1)];
end
recalculation = transpose(recalculation);
recalculation = cell2mat(recalculation);

figure();
plot(rr_peaks_st(109).rr_peaks(2:end), rr_peaks_st(109).rr_interval);
title("Original");
figure();
plot(rr_peaks_st(109).rr_peaks(2:end), recalculation);
title("Recalculated");