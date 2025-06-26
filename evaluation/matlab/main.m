% Script for obtaining the required results for IROS'18
base_dir = '/home/noah/tfm/src/btc_descriptor/';
gt_neigh = 40;
compensate = false;

% Configuring subpaths
addpath('AcademicFigures/');

% Obtaining CityCenter results
curr_dir = strcat(base_dir, 'KITTI00/');
[PR_K0, imgvssize_K0, imgvstime_K0] = process(curr_dir, gt_neigh, compensate);
imgvstime_K0.time = smooth(imgvstime_K0.time);

curr_dir = strcat(base_dir, 'KITTI02/');
[PR_K2, imgvssize_K2, imgvstime_K2] = process(curr_dir, gt_neigh, compensate);
imgvstime_K2.time = smooth(imgvstime_K2.time);

curr_dir = strcat(base_dir, 'KITTI05/');
[PR_K5, imgvssize_K5, imgvstime_K5] = process(curr_dir, gt_neigh, compensate);
imgvstime_K5.time = smooth(imgvstime_K5.time);

curr_dir = strcat(base_dir, 'KITTI06/');
[PR_K6, imgvssize_K6, imgvstime_K6] = process(curr_dir, gt_neigh, compensate);
imgvstime_K6.time = smooth(imgvstime_K6.time);

curr_dir = strcat(base_dir, 'KITTI07/');
[PR_K7, imgvssize_K7, imgvstime_K7] = process(curr_dir, gt_neigh, compensate);
imgvstime_K7.time = smooth(imgvstime_K7.time);

curr_dir = strcat(base_dir, 'KITTI08/');
[PR_K8, imgvssize_K8, imgvstime_K8] = process(curr_dir, gt_neigh, compensate);
imgvstime_K8.time = smooth(imgvstime_K8.time);

% P/R curves
afigure;
hold on;
plot(PR_K0.R, PR_K0.P, '-d', 'MarkerIndices', length(PR_K0.P));
plot(PR_K2.R, PR_K2.P, '-o', 'MarkerIndices', length(PR_K2.P));
plot(PR_K5.R, PR_K5.P, '--^', 'MarkerIndices', length(PR_K5.P));
plot(PR_K6.R, PR_K6.P, '--p', 'MarkerIndices', length(PR_K6.P));
plot(PR_K7.R, PR_K7.P, '-*', 'MarkerIndices', length(PR_K7.P));
plot(PR_K8.R, PR_K8.P, '-x', 'MarkerIndices', length(PR_K8.P));

xlabel('Recall');
ylabel('Precision');
xlim([0.7, 1.02]);
ylim([0.4, 1.02]);
% legend('CC', 'NC', 'L6I', 'L6O', 'K00', 'K05', 'K06', 'Location', 'SouthWest');
legend('K00', 'K02', 'K05', 'K06', 'K07', 'K08');
hold off;
print('-depsc', strcat(base_dir, 'PR_curves'));

% Images vs Size
afigure;
hold on;
%plot(imgvssize_CC.img, imgvssize_CC.size, '-o', 'MarkerIndices', length(imgvssize_CC.size));
%plot(imgvssize_NC.img, imgvssize_NC.size, '-*', 'MarkerIndices', length(imgvssize_NC.size));
%plot(imgvssize_L6I.img, imgvssize_L6I.size, '-*', 'MarkerIndices', length(imgvssize_L6I.size));
%plot(imgvssize_L6O.img, imgvssize_L6O.size, '--s', 'MarkerIndices', length(imgvssize_L6O.size));
%plot(imgvssize_K0.img, imgvssize_K0.size, '-d', 'MarkerIndices', length(imgvssize_K0.size));
%plot(imgvssize_K5.img, imgvssize_K5.size, '--^', 'MarkerIndices', length(imgvssize_K5.size));
%plot(imgvssize_K6.img, imgvssize_K6.size, '--p', 'MarkerIndices', length(imgvssize_K6.size));

plot(imgvssize_K0.img(80:end), imgvssize_K0.size(80:end), '-d', 'MarkerIndices', length(imgvssize_K0.size(80:end)));
xlabel('Time Index');
ylabel('Vocabulary Size (Words)');
xlim([80, length(imgvssize_K0.size(80:end))]);
%legend('CC', 'NC', 'L6I', 'L6O', 'K00', 'K05', 'K06', 'Location', 'NorthWest');
hold off;
print('-depsc', strcat(base_dir, 'imgs_vs_size'));

% Images vs Time
afigure;
hold on;
%plot(imgvstime_CC.img, imgvstime_CC.time, '-o', 'MarkerIndices', length(imgvstime_CC.time));
%plot(imgvstime_NC.img, imgvstime_NC.time, '-*', 'MarkerIndices', length(imgvstime_NC.time));
%plot(imgvstime_L6I.img, imgvstime_L6I.time, '-*', 'MarkerIndices', length(imgvstime_L6I.time));
%plot(imgvstime_L6O.img, imgvstime_L6O.time, '--s', 'MarkerIndices', length(imgvstime_L6O.time));
% plot(imgvstime_K0.img, imgvstime_K0.time, '-d', 'MarkerIndices', length(imgvstime_K0.time));
%plot(imgvstime_K5.img, imgvstime_K5.time, '--^', 'MarkerIndices', length(imgvstime_K5.time));
%plot(imgvstime_K6.img, imgvstime_K6.time, '--p', 'MarkerIndices', length(imgvstime_K6.time));

plot(imgvstime_K0.img(80:end), imgvstime_K0.time(80:end), '-d', 'MarkerIndices', length(imgvstime_K0.time(80:end)));
xlabel('Time Index');
ylabel('Avg. Time (ms)');
xlim([80, length(imgvstime_K0.time(80:end))]);
%legend('CC', 'NC', 'L6I', 'L6O', 'K00', 'K05', 'K06', 'Location', 'NorthWest');
hold off;
print('-depsc', strcat(base_dir, 'imgs_vs_time'));

% Showing summaries
disp('----- Summary KITTI 00 -----');
disp(['Max P: ', num2str(PR_K0.P_max)]);
disp(['Max R: ', num2str(PR_K0.R_max)]);
disp(['Max VWords: ', num2str(imgvssize_K0.size(end))]);
disp(['Avg. Time: ', num2str(mean(imgvstime_K0.time))]);
disp(['Std. Time: ', num2str(std(imgvstime_K0.time))]);

disp('----- Summary KITTI 02 -----');
disp(['Max P: ', num2str(PR_K2.P_max)]);
disp(['Max R: ', num2str(PR_K2.R_max)]);
disp(['Max VWords: ', num2str(imgvssize_K2.size(end))]);
disp(['Avg. Time: ', num2str(mean(imgvstime_K2.time))]);
disp(['Std. Time: ', num2str(std(imgvstime_K2.time))]);

disp('----- Summary KITTI 05 -----');
disp(['Max P: ', num2str(PR_K5.P_max)]);
disp(['Max R: ', num2str(PR_K5.R_max)]);
disp(['Max VWords: ', num2str(imgvssize_K5.size(end))]);
disp(['Avg. Time: ', num2str(mean(imgvstime_K5.time))]);
disp(['Std. Time: ', num2str(std(imgvstime_K5.time))]);

disp('----- Summary KITTI 06 -----');
disp(['Max P: ', num2str(PR_K6.P_max)]);
disp(['Max R: ', num2str(PR_K6.R_max)]);
disp(['Max VWords: ', num2str(imgvssize_K6.size(end))]);
disp(['Avg. Time: ', num2str(mean(imgvstime_K6.time))]);
disp(['Std. Time: ', num2str(std(imgvstime_K6.time))]);

disp('----- Summary KITTI 07 -----');
disp(['Max P: ', num2str(PR_K7.P_max)]);
disp(['Max R: ', num2str(PR_K7.R_max)]);
disp(['Max VWords: ', num2str(imgvssize_K7.size(end))]);
disp(['Avg. Time: ', num2str(mean(imgvstime_K7.time))]);
disp(['Std. Time: ', num2str(std(imgvstime_K7.time))]);

disp('----- Summary KITTI 08 -----');
disp(['Max P: ', num2str(PR_K8.P_max)]);
disp(['Max R: ', num2str(PR_K8.R_max)]);
disp(['Max VWords: ', num2str(imgvssize_K8.size(end))]);
disp(['Avg. Time: ', num2str(mean(imgvstime_K8.time))]);
disp(['Std. Time: ', num2str(std(imgvstime_K8.time))]);

close all;