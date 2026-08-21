%% ==========================================================
% IMPROVED GA + LSTM MODEL FOR 1-DAY WIND FORECASTING
%
% Dataset: Bhubaneswar Weather Data
% Period  : Dec 2018 – Mar 2026
%
% Goal:
% Predict next-day:
%   1. Mean wind speed
%   2. Minimum wind speed
%   3. Maximum wind speed
%   4. Wind direction
%
% Core model:
%   ANN baseline
%   GA-optimized LSTM main model
%
% Major upgrades:
%   1. 1-day-ahead forecasting only
%   2. Full-date plots across whole dataset
%   3. Validation-based GA optimization
%   4. Training-only normalization
%   5. Final performance report for training/validation/test
% ==========================================================

clc
clear
close all

rng(42)

%% ==========================================================
% STEP 1 — LOAD DATASET
% ==========================================================

data = readtable('BBSR_wind_solar_2018dec_2026mar.xlsx');
data = sortrows(data,'datetime');
fprintf('First date in raw file: %s\n', datestr(min(data.datetime)))
fprintf('Last date in raw file : %s\n', datestr(max(data.datetime)))
fprintf('Total rows loaded     : %d\n', height(data))

%% ==========================================================
% STEP 2 — SELECT VARIABLES
% ==========================================================

data = data(:,{'datetime',...
'temp','humidity','dew','precip',...
'windgust','windspeed','windspeedmax','windspeedmean','windspeedmin','winddir',...
'sealevelpressure','cloudcover','solarradiation'});

%% ==========================================================
% STEP 3 — BASIC CLEANING
% ==========================================================

data = rmmissing(data);

numericVars = varfun(@isnumeric,data,'OutputFormat','uniform');
data{:,numericVars} = fillmissing(data{:,numericVars},'linear');
data{:,numericVars} = fillmissing(data{:,numericVars},'nearest');

%% ==========================================================
% STEP 4 — FEATURE ENGINEERING
% ==========================================================

data.datetime = datetime(data.datetime);
time = data.datetime;

data.day       = day(data.datetime);
data.month     = month(data.datetime);
data.dayofyear = day(data.datetime,'dayofyear');

%% ==========================================================
% STEP 5 — WIND DIRECTION CIRCULAR ENCODING
% ==========================================================

data.winddir_sin = sind(data.winddir);
data.winddir_cos = cosd(data.winddir);

data.winddir = [];
data.datetime = [];

%% ==========================================================
% STEP 6 — TABLE TO NUMERIC MATRIX
% ==========================================================

dataset = table2array(data);
varNames = data.Properties.VariableNames;

dataset(isinf(dataset)) = NaN;
dataset = fillmissing(dataset,'linear');
dataset = fillmissing(dataset,'nearest');

sigma_tmp = std(dataset,0,1);
zeroVarCols = sigma_tmp == 0;

dataset(:,zeroVarCols) = [];
varNames(zeroVarCols)  = [];

if any(isnan(dataset),'all')
    error('Dataset still contains NaN after cleaning.')
end

%% ==========================================================
% STEP 7 — BUILD 1-DAY-AHEAD INPUTS AND TARGETS
% ==========================================================

idx_mean = find(strcmp(varNames,'windspeedmean'));
idx_min  = find(strcmp(varNames,'windspeedmin'));
idx_max  = find(strcmp(varNames,'windspeedmax'));
idx_sin  = find(strcmp(varNames,'winddir_sin'));
idx_cos  = find(strcmp(varNames,'winddir_cos'));

if isempty(idx_mean) || isempty(idx_min) || isempty(idx_max) || isempty(idx_sin) || isempty(idx_cos)
    error('One or more required target columns were not found.')
end

% X(t) = current-day weather features
% Y(t) = next-day wind outputs
X_raw = dataset(1:end-1,:);
Y_raw = [ ...
    dataset(2:end,idx_mean), ...
    dataset(2:end,idx_min), ...
    dataset(2:end,idx_max), ...
    dataset(2:end,idx_sin), ...
    dataset(2:end,idx_cos) ...
];

targetTime = time(2:end);

%% ==========================================================
% STEP 8 — TRAIN / TEST SPLIT
% ==========================================================

n = size(X_raw,1);
train_ratio = 0.8;
train_size = round(train_ratio * n);

Xtrain_raw = X_raw(1:train_size,:);
Ytrain_raw = Y_raw(1:train_size,:);

Xtest_raw  = X_raw(train_size+1:end,:);
Ytest_raw  = Y_raw(train_size+1:end,:);

time_train_target = targetTime(1:train_size);
time_test_target  = targetTime(train_size+1:end);

%% ==========================================================
% STEP 9 — TRAINING-ONLY NORMALIZATION
% ==========================================================

muX = mean(Xtrain_raw,1);
sigmaX = std(Xtrain_raw,0,1);
sigmaX(sigmaX == 0) = 1;

muY = mean(Ytrain_raw,1);
sigmaY = std(Ytrain_raw,0,1);
sigmaY(sigmaY == 0) = 1;

Xtrain = (Xtrain_raw - muX) ./ sigmaX;
Xtest  = (Xtest_raw  - muX) ./ sigmaX;

Ytrain = (Ytrain_raw - muY) ./ sigmaY;
Ytest  = (Ytest_raw  - muY) ./ sigmaY;

Xall_norm = (X_raw - muX) ./ sigmaX;
Yall_norm = (Y_raw - muY) ./ sigmaY;

%% ==========================================================
% STEP 10 — ANN BASELINE
% ==========================================================

hiddenLayerSize = [30 20];

net_ann = feedforwardnet(hiddenLayerSize);
net_ann.trainParam.showWindow = true;

net_ann = train(net_ann, Xtrain', Ytrain');
Ypred_ann = net_ann(Xtest')';

% ---- ANN RMSE based on mean wind speed only
pred_ann_mean = Ypred_ann(:,1) * sigmaY(1) + muY(1);
true_ann_mean = Ytest(:,1)     * sigmaY(1) + muY(1);

rmse_ann = sqrt(mean((pred_ann_mean - true_ann_mean).^2));

smape_ann = mean( ...
    2 * abs(pred_ann_mean - true_ann_mean) ./ ...
    (abs(pred_ann_mean) + abs(true_ann_mean) + eps) ...
) * 100;

eff_ann = 100 - smape_ann;

disp("ANN RMSE (mean wind speed only):")
disp(rmse_ann)
fprintf("ANN Average Percentage Error: %.2f %%\n", smape_ann);
fprintf("ANN Efficiency: %.2f %%\n", eff_ann);

% ---- ANN 1-day forecast
ann_pred1 = net_ann(Xall_norm(end,:)')';

ann_meanW = ann_pred1(1) * sigmaY(1) + muY(1);
ann_minW  = ann_pred1(2) * sigmaY(2) + muY(2);
ann_maxW  = ann_pred1(3) * sigmaY(3) + muY(3);

ann_sinD  = ann_pred1(4) * sigmaY(4) + muY(4);
ann_cosD  = ann_pred1(5) * sigmaY(5) + muY(5);

ann_dir = atan2d(ann_sinD, ann_cosD);
if ann_dir < 0
    ann_dir = ann_dir + 360;
end

disp(" ")
disp("======================================")
disp(" ANN BASELINE — 1 DAY FORECAST ")
disp("======================================")
fprintf('Mean Wind Speed  : %.2f km/h\n', ann_meanW);
fprintf('Min Wind Speed   : %.2f km/h\n', ann_minW);
fprintf('Max Wind Speed   : %.2f km/h\n', ann_maxW);
fprintf('Wind Direction   : %.1f degrees\n', ann_dir);
fprintf('RMSE of mean wind speed only : %.4f\n', rmse_ann);
fprintf('--------------------------------------\n');


%% ==========================================================
% STEP 11 — PREPARE LSTM SEQUENCES
% ==========================================================

window = 60;

numTrainSeq = size(Xtrain,1) - window + 1;

Xtrain_lstm = cell(numTrainSeq,1);
Ytrain_lstm = zeros(numTrainSeq,size(Ytrain,2));
time_train_seq = NaT(numTrainSeq,1);

for i = 1:numTrainSeq
    Xtrain_lstm{i} = Xtrain(i:i+window-1,:)';
    Ytrain_lstm(i,:) = Ytrain(i+window-1,:);
    time_train_seq(i) = time_train_target(i+window-1);
end

%% ==========================================================
% STEP 11A — TRAIN / VALIDATION SPLIT
% ==========================================================

val_ratio = 0.2;
val_size = round(val_ratio * numTrainSeq);

Xval_lstm = Xtrain_lstm(end-val_size+1:end);
Yval_lstm = Ytrain_lstm(end-val_size+1:end,:);
time_val_seq = time_train_seq(end-val_size+1:end);

Xtrain_lstm_final = Xtrain_lstm(1:end-val_size);
Ytrain_lstm_final = Ytrain_lstm(1:end-val_size,:);
time_train_seq_final = time_train_seq(1:end-val_size);

%% ==========================================================
% STEP 12 — GA OPTIMIZATION (FAST SEARCH)
% ==========================================================

% GA optimizes:
% x(1) = hidden units
% x(2) = epochs
% x(3) = dropout rate
% x(4) = learning rate

fitnessFunction = @(x) galstmFitness( ...
    x, Xtrain_lstm_final, Ytrain_lstm_final, Xval_lstm, Yval_lstm);

nvars = 4;

% narrower bounds = faster and more stable search
lb = [48 18 0.03 2e-4];
ub = [96 40 0.12 8e-4];

optionsGA = optimoptions('ga', ...
    'PopulationSize',10, ...
    'MaxGenerations',15, ...
    'Display','iter');

best_params = ga(fitnessFunction,nvars,[],[],[],[],lb,ub,[],optionsGA);

disp("Best GA Parameters:")
disp(best_params)

%% ==========================================================
% STEP 13 — FINAL GA-LSTM MODEL (FULL CYCLE + VALIDATION CURVE)
% ==========================================================

numHiddenUnits = round(best_params(1));
maxEpochs      = round(best_params(2));
dropRate       = min(best_params(3), 0.10);
learnRate      = best_params(4);
inputSize = size(Xtrain,2);
numResponses = size(Ytrain,2);

% ----------------------------------------------------------
% STORE TRAINING HISTORY
% ----------------------------------------------------------
trainHistory = struct();
trainHistory.iteration      = [];
trainHistory.epoch          = [];
trainHistory.trainingLoss   = [];
trainHistory.trainingRMSE   = [];
trainHistory.validationLoss = [];
trainHistory.validationRMSE = [];

setappdata(0,'trainHistory',trainHistory);

% ----------------------------------------------------------
% DEFINE NETWORK
% ----------------------------------------------------------
layers = [
    sequenceInputLayer(inputSize)

    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    dropoutLayer(dropRate)

    lstmLayer(max(12,round(numHiddenUnits/2)),'OutputMode','last')
    dropoutLayer(dropRate)

    fullyConnectedLayer(48)
    reluLayer

    fullyConnectedLayer(numResponses)
    regressionLayer
];

% ----------------------------------------------------------
% CALCULATED PATIENCE TO ALLOW FULL TRAINING
% ----------------------------------------------------------
miniBatchSize = 10;
validationFrequency = 10;

iterationsPerEpoch = ceil(numel(Xtrain_lstm_final) / miniBatchSize);
maxIterations = maxEpochs * iterationsPerEpoch;

% number of validation checks expected during full training
numValidationChecks = ceil(maxIterations / validationFrequency);

% set patience slightly larger than required number of checks
validationPatienceFull = numValidationChecks + 5;

% ----------------------------------------------------------
% TRAINING OPTIONS
% ----------------------------------------------------------
options = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',learnRate, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',8, ...
    'L2Regularization',1e-4, ...
    'Shuffle','never', ...
    'ValidationData',{Xval_lstm,Yval_lstm}, ...
    'ValidationFrequency',validationFrequency, ...
    'ValidationPatience',validationPatienceFull, ...
    'Plots','training-progress', ...
    'OutputFcn',@captureTrainingInfo, ...
    'ExecutionEnvironment','cpu', ...
    'Verbose',0);

% ----------------------------------------------------------
% TRAIN FINAL MODEL
% ----------------------------------------------------------
net_lstm = trainNetwork(Xtrain_lstm_final, Ytrain_lstm_final, layers, options);

trainHistory = getappdata(0,'trainHistory');


%% ==========================================================
% STEP 14 — BUILD TEST SEQUENCES
% ==========================================================

numTestSeq = size(Xtest,1) - window + 1;

Xtest_lstm = cell(numTestSeq,1);
Ytest_seq  = zeros(numTestSeq,size(Ytest,2));
time_test_seq = NaT(numTestSeq,1);

for i = 1:numTestSeq
    Xtest_lstm{i} = Xtest(i:i+window-1,:)';
    Ytest_seq(i,:) = Ytest(i+window-1,:);
    time_test_seq(i) = time_test_target(i+window-1);
end


%% ==========================================================
% STEP 14A — PLAIN LSTM MODEL (NO GA OPTIMIZATION)
% ==========================================================

plainHiddenUnits = 64;
plainEpochs      = 20;
plainDropRate    = 0.05;
plainLearnRate   = 5e-4;

plainLayers = [
    sequenceInputLayer(inputSize)

    lstmLayer(plainHiddenUnits,'OutputMode','sequence')
    dropoutLayer(plainDropRate)

    lstmLayer(max(12,round(plainHiddenUnits/2)),'OutputMode','last')
    dropoutLayer(plainDropRate)

    fullyConnectedLayer(48)
    reluLayer

    fullyConnectedLayer(numResponses)
    regressionLayer
];

plainOptions = trainingOptions('adam', ...
    'MaxEpochs',plainEpochs, ...
    'MiniBatchSize',10, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',plainLearnRate, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',8, ...
    'L2Regularization',1e-4, ...
    'Shuffle','never', ...
    'ValidationData',{Xval_lstm,Yval_lstm}, ...
    'ValidationFrequency',10, ...
    'ValidationPatience',50, ...
    'ExecutionEnvironment','cpu', ...
    'Verbose',0, ...
    'Plots','none');

net_lstm_plain = trainNetwork(Xtrain_lstm_final, Ytrain_lstm_final, plainLayers, plainOptions);

% ---- Plain LSTM test prediction
Ypred_plain = predict(net_lstm_plain, Xtest_lstm);

pred_plain_mean = Ypred_plain(:,1) * sigmaY(1) + muY(1);
true_plain_mean = Ytest_seq(:,1)   * sigmaY(1) + muY(1);

rmse_plain = sqrt(mean((pred_plain_mean - true_plain_mean).^2));

smape_plain = mean( ...
    2 * abs(pred_plain_mean - true_plain_mean) ./ ...
    (abs(pred_plain_mean) + abs(true_plain_mean) + eps) ...
) * 100;

eff_plain = 100 - smape_plain;

disp(" ")
disp("PLAIN LSTM RMSE (mean wind speed only):")
disp(rmse_plain)
fprintf("PLAIN LSTM Average Percentage Error: %.2f %%\n", smape_plain);
fprintf("PLAIN LSTM Efficiency: %.2f %%\n", eff_plain);

% ---- Plain LSTM 1-day forecast
plain_pred1 = predict(net_lstm_plain, {Xall_norm(end-window+1:end,:)'});

plain_meanW = plain_pred1(1) * sigmaY(1) + muY(1);
plain_minW  = plain_pred1(2) * sigmaY(2) + muY(2);
plain_maxW  = plain_pred1(3) * sigmaY(3) + muY(3);

plain_sinD = plain_pred1(4) * sigmaY(4) + muY(4);
plain_cosD = plain_pred1(5) * sigmaY(5) + muY(5);

plain_dir = atan2d(plain_sinD, plain_cosD);
if plain_dir < 0
    plain_dir = plain_dir + 360;
end

disp(" ")
disp("======================================")
disp(" PLAIN LSTM — 1 DAY FORECAST ")
disp("======================================")
fprintf('Mean Wind Speed  : %.2f km/h\n', plain_meanW);
fprintf('Min Wind Speed   : %.2f km/h\n', plain_minW);
fprintf('Max Wind Speed   : %.2f km/h\n', plain_maxW);
fprintf('Wind Direction   : %.1f degrees\n', plain_dir);
fprintf('RMSE of mean wind speed only : %.4f\n', rmse_plain);
fprintf('--------------------------------------\n');


%% ==========================================================
% STEP 15 — TRAIN / VALIDATION / TEST METRICS
% ==========================================================

% ---- Training prediction
Ypred_train = predict(net_lstm,Xtrain_lstm_final);

pred_train_mean = Ypred_train(:,1) * sigmaY(1) + muY(1);
true_train_mean = Ytrain_lstm_final(:,1) * sigmaY(1) + muY(1);

rmse_train = sqrt(mean((pred_train_mean - true_train_mean).^2));

smape_train = mean( ...
    2 * abs(pred_train_mean - true_train_mean) ./ ...
    (abs(pred_train_mean) + abs(true_train_mean) + eps) ...
) * 100;

eff_train = 100 - smape_train;

% ---- Validation prediction
Ypred_val = predict(net_lstm,Xval_lstm);

pred_val_mean = Ypred_val(:,1) * sigmaY(1) + muY(1);
true_val_mean = Yval_lstm(:,1) * sigmaY(1) + muY(1);

rmse_val = sqrt(mean((pred_val_mean - true_val_mean).^2));

smape_val = mean( ...
    2 * abs(pred_val_mean - true_val_mean) ./ ...
    (abs(pred_val_mean) + abs(true_val_mean) + eps) ...
) * 100;

eff_val = 100 - smape_val;

% ---- Test prediction
Ypred_lstm = predict(net_lstm,Xtest_lstm);

pred_test_mean = Ypred_lstm(:,1) * sigmaY(1) + muY(1);
true_test_mean = Ytest_seq(:,1)  * sigmaY(1) + muY(1);

rmse_lstm = sqrt(mean((pred_test_mean - true_test_mean).^2));

smape_lstm = mean( ...
    2 * abs(pred_test_mean - true_test_mean) ./ ...
    (abs(pred_test_mean) + abs(true_test_mean) + eps) ...
) * 100;

eff_lstm = 100 - smape_lstm;

fprintf('\n======================================\n');
fprintf('FINAL PERFORMANCE REPORT\n');
fprintf('======================================\n');
fprintf('ANN RMSE (mean only)              : %.4f\n', rmse_ann);
fprintf('ANN Percentage Error              : %.2f %%\n', smape_ann);
fprintf('ANN Efficiency                    : %.2f %%\n', eff_ann);
fprintf('--------------------------------------\n');
fprintf('PLAIN LSTM RMSE (mean only)       : %.4f\n', rmse_plain);
fprintf('PLAIN LSTM Percentage Error       : %.2f %%\n', smape_plain);
fprintf('PLAIN LSTM Efficiency             : %.2f %%\n', eff_plain);
fprintf('--------------------------------------\n');
fprintf('GA-LSTM Training RMSE (mean only) : %.4f\n', rmse_train);
fprintf('GA-LSTM Training Error            : %.2f %%\n', smape_train);
fprintf('GA-LSTM Training Efficiency       : %.2f %%\n', eff_train);
fprintf('GA-LSTM Validation RMSE (mean only): %.4f\n', rmse_val);
fprintf('GA-LSTM Validation Error          : %.2f %%\n', smape_val);
fprintf('GA-LSTM Validation Efficiency     : %.2f %%\n', eff_val);
fprintf('GA-LSTM Test RMSE (mean only)     : %.4f\n', rmse_lstm);
fprintf('GA-LSTM Test Percentage Error     : %.2f %%\n', smape_lstm);
fprintf('GA-LSTM Test Efficiency           : %.2f %%\n', eff_lstm);
fprintf('======================================\n');


%% ==========================================================
% STEP 16 — ENHANCED TRAINING PROGRESS ANALYSIS
% ==========================================================

validTrainLoss = ~isnan(trainHistory.trainingLoss);
validTrainRMSE = ~isnan(trainHistory.trainingRMSE);
validValLoss   = ~isnan(trainHistory.validationLoss);
validValRMSE   = ~isnan(trainHistory.validationRMSE);

itTrainLoss = trainHistory.iteration(validTrainLoss);
itTrainRMSE = trainHistory.iteration(validTrainRMSE);
itValLoss   = trainHistory.iteration(validValLoss);
itValRMSE   = trainHistory.iteration(validValRMSE);

trainLossCurve = trainHistory.trainingLoss(validTrainLoss);
trainRMSECurve = trainHistory.trainingRMSE(validTrainRMSE);
valLossCurve   = trainHistory.validationLoss(validValLoss);
valRMSECurve   = trainHistory.validationRMSE(validValRMSE);

% Smoothed curves for presentation
trainLossSmooth = movmean(trainLossCurve,25);
trainRMSESmooth = movmean(trainRMSECurve,25);

% Best validation points
[minValLoss, idxBestValLoss] = min(valLossCurve);
bestValLossIter = itValLoss(idxBestValLoss);

if ~isempty(valRMSECurve)
    [minValRMSE, idxBestValRMSE] = min(valRMSECurve);
    bestValRMSEIter = itValRMSE(idxBestValRMSE);
else
    minValRMSE = rmse_val;
    bestValRMSEIter = NaN;
end

% ----------------------------------------------------------
% FIGURE 1 — TRAINING RMSE + VALIDATION RMSE
% ----------------------------------------------------------
figure('Name','Training RMSE vs Iteration','NumberTitle','off')
plot(itTrainRMSE, trainRMSECurve,'Color',[0.75 0.85 1],'LineWidth',0.8)
hold on
plot(itTrainRMSE, trainRMSESmooth,'b','LineWidth',2)

if ~isempty(valRMSECurve)
    plot(itValRMSE, valRMSECurve,'ko-','LineWidth',1.2,'MarkerSize',4)
    plot(bestValRMSEIter, minValRMSE,'rs','MarkerSize',8,'MarkerFaceColor','r')
end

title('Training RMSE vs Iteration')
xlabel('Iteration')
ylabel('RMSE')
grid on
legend('Training RMSE','Smoothed Training RMSE','Validation RMSE','Best Validation RMSE','Location','best')


summaryText1 = sprintf([ ...
    'Best GA Parameters\n' ...
    'Hidden Units = %d\n' ...
    'Epochs = %d\n' ...
    'Dropout = %.3f\n' ...
    'Learn Rate = %.5f\n\n' ...
    'Train RMSE = %.4f\n' ...
    'Val RMSE = %.4f\n' ...
    'Test RMSE = %.4f\n' ...
    'ANN RMSE = %.4f\n\n' ...
    'Train Error = %.2f%%\n' ...
    'Val Error = %.2f%%\n' ...
    'Test Error = %.2f%%\n' ...
    'ANN Error = %.2f%%\n\n' ...
    'Train Eff. = %.2f%%\n' ...
    'Val Eff. = %.2f%%\n' ...
    'Test Eff. = %.2f%%\n' ...
    'ANN Eff. = %.2f%%'], ...
    numHiddenUnits, maxEpochs, dropRate, learnRate, ...
    rmse_train, rmse_val, rmse_lstm, rmse_ann, ...
    smape_train, smape_val, smape_lstm, smape_ann, ...
    eff_train, eff_val, eff_lstm, eff_ann);

annotation('textbox',[0.62 0.48 0.28 0.38], ...
    'String',summaryText1, ...
    'FitBoxToText','on', ...
    'BackgroundColor','w', ...
    'EdgeColor','k');

% ----------------------------------------------------------
% FIGURE 2 — TRAINING LOSS + VALIDATION LOSS
% ----------------------------------------------------------
figure('Name','Training and Validation Loss vs Iteration','NumberTitle','off')
plot(itTrainLoss, trainLossCurve,'Color',[1 0.85 0.75],'LineWidth',0.8)
hold on
plot(itTrainLoss, trainLossSmooth,'r','LineWidth',2)
plot(itValLoss, valLossCurve,'ko-','LineWidth',1.2,'MarkerSize',4)
plot(bestValLossIter, minValLoss,'bs','MarkerSize',8,'MarkerFaceColor','b')

title('Training and Validation Loss vs Iteration')
xlabel('Iteration')
ylabel('Loss')
grid on
legend('Training Loss','Smoothed Training Loss','Validation Loss','Best Validation Loss','Location','best')


generalizationGap = rmse_val - rmse_train;

summaryText2 = sprintf([ ...
    'Best Validation Loss = %.4f\n' ...
    'Best Validation Iteration = %d\n' ...
    'Generalization Gap (Val - Train RMSE) = %.4f\n' ...
    'Window Size = %d\n' ...
    'Validation Frequency = %d'], ...
    minValLoss, bestValLossIter, generalizationGap, window, 20);

annotation('textbox',[0.62 0.62 0.25 0.18], ...
    'String',summaryText2, ...
    'FitBoxToText','on', ...
    'BackgroundColor','w', ...
    'EdgeColor','k');

% ----------------------------------------------------------
% FIGURE 3 — VALIDATION LOSS ONLY
% ----------------------------------------------------------
figure('Name','Validation Loss vs Iteration','NumberTitle','off')
plot(itValLoss, valLossCurve,'b-o','LineWidth',1.5,'MarkerSize',4)
hold on
plot(bestValLossIter, minValLoss,'rs','MarkerSize',8,'MarkerFaceColor','r')

title('Validation Loss vs Iteration')
xlabel('Iteration')
ylabel('Validation Loss')
grid on

legend('Validation Loss','Best Validation Loss','Location','best')
text(itValLoss(round(end*0.65)), max(valLossCurve)*0.95, ...
    sprintf('Final Val RMSE = %.4f\nVal Error = %.2f%%\nVal Efficiency = %.2f%%', ...
    rmse_val, smape_val, eff_val), ...
    'BackgroundColor','w','EdgeColor','k')

%% ==========================================================
% STEP 16C — VALIDATION LOSS VS ITERATION
% ==========================================================

bestValSoFar = cummin(valLossCurve);

figure('Name','Validation Loss vs Iteration','NumberTitle','off')

plot(itValLoss, valLossCurve,'Color',[0.75 0.75 1],'LineWidth',1.0)
hold on
plot(itValLoss, bestValSoFar,'b','LineWidth',2.5)
plot(bestValLossIter, minValLoss,'rs','MarkerSize',8,'MarkerFaceColor','r')

title('Validation Loss vs Iteration')
xlabel('Iteration')
ylabel('Validation Loss')
grid on

legend('Raw Validation Loss','Best Validation Loss So Far','Best Validation Point','Location','best')

text(itValLoss(round(end*0.65)), max(valLossCurve)*0.95, ...
    sprintf('Final Val RMSE (real scale) = %.4f km/h\nVal Error = %.2f%%\nVal Efficiency = %.2f%%', ...
    rmse_val, smape_val, eff_val), ...
    'BackgroundColor','w','EdgeColor','k')


%% ==========================================================
% STEP 16C — BEST VALIDATION LOSS SO FAR
% ==========================================================

bestValSoFar = cummin(valLossCurve);

figure('Name','Validation Loss vs Iteration','NumberTitle','off')
plot(itValLoss, bestValSoFar,'b','LineWidth',2.5)
hold on
plot(bestValLossIter, minValLoss,'rs','MarkerSize',8,'MarkerFaceColor','r')

title('Validation Loss vs Iteration')
xlabel('Iteration')
ylabel('Validation Loss')
grid on
legend('Best Validation Loss So Far','Best Validation Point','Location','best')


text(itValLoss(round(end*0.65)), max(bestValSoFar)*0.98, ...
    sprintf('Best Val RMSE (real scale) = %.4f km/h\nVal Error = %.2f%%\nVal Efficiency = %.2f%%', ...
    rmse_val, smape_val, eff_val), ...
    'BackgroundColor','w','EdgeColor','k')


%% ==========================================================
% STEP 17 — FULL DATASET PREDICTION FOR FULL-DATE PLOTS
% ==========================================================

numAllSeq = size(Xall_norm,1) - window + 1;

Xall_lstm = cell(numAllSeq,1);
Yall_seq  = zeros(numAllSeq,size(Yall_norm,2));
time_all_seq = NaT(numAllSeq,1);

for i = 1:numAllSeq
    Xall_lstm{i} = Xall_norm(i:i+window-1,:)';
    Yall_seq(i,:) = Yall_norm(i+window-1,:);
    time_all_seq(i) = targetTime(i+window-1);
end

Ypred_all = predict(net_lstm,Xall_lstm);

actual_full_mean = Yall_seq(:,1) * sigmaY(1) + muY(1);
pred_full_mean   = Ypred_all(:,1) * sigmaY(1) + muY(1);


%% ==========================================================
% STEP 17A — FULL-RANGE TEST SERIES FOR PLOTTING
% ==========================================================

% Full timeline corresponding to all supervised targets
full_time_axis = targetTime;

% Create full-length NaN series
true_test_full = NaN(length(full_time_axis),1);
pred_test_full = NaN(length(full_time_axis),1);
error_full     = NaN(length(full_time_axis),1);
abs_error_full = NaN(length(full_time_axis),1);

% Find where test-sequence dates sit inside full_time_axis
[~, test_idx_in_full] = ismember(time_test_seq, full_time_axis);

% Insert test predictions into full-length vectors
true_test_full(test_idx_in_full) = true_test_mean;
pred_test_full(test_idx_in_full) = pred_test_mean;

error_series = true_test_mean - pred_test_mean;
abs_error    = abs(error_series);

error_full(test_idx_in_full)     = error_series;
abs_error_full(test_idx_in_full) = abs_error;


%% ==========================================================
% STEP 18 — RETRAIN FINAL MODEL ON FULL DATA FOR DEPLOYMENT
% ==========================================================

% Build full sequences using all normalized data
numAllSeq_full = size(Xall_norm,1) - window + 1;

Xfull_lstm = cell(numAllSeq_full,1);
Yfull_lstm = zeros(numAllSeq_full,size(Yall_norm,2));

for i = 1:numAllSeq_full
    Xfull_lstm{i} = Xall_norm(i:i+window-1,:)';
    Yfull_lstm(i,:) = Yall_norm(i+window-1,:);
end

% Use best GA parameters already found earlier
numHiddenUnits_full = round(best_params(1));
maxEpochs_full      = round(best_params(2));
dropRate_full       = best_params(3);
learnRate_full      = best_params(4);

inputSize = size(Xtrain,2);
numResponses = size(Ytrain,2);

layers_full = [
    sequenceInputLayer(inputSize)

    lstmLayer(numHiddenUnits_full,'OutputMode','sequence')
    dropoutLayer(dropRate_full)

    lstmLayer(max(12,round(numHiddenUnits_full/2)),'OutputMode','last')
    dropoutLayer(dropRate_full)

    fullyConnectedLayer(48)
    reluLayer

    fullyConnectedLayer(numResponses)
    regressionLayer
];

options_full = trainingOptions('adam', ...
    'MaxEpochs',maxEpochs_full, ...
    'MiniBatchSize',10, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',learnRate_full, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.5, ...
    'LearnRateDropPeriod',8, ...
    'L2Regularization',1e-4, ...
    'Shuffle','never', ...
    'ExecutionEnvironment','cpu', ...
    'Verbose',0, ...
    'Plots','training-progress');

net_lstm_full = trainNetwork(Xfull_lstm, Yfull_lstm, layers_full, options_full);


%% ==========================================================
% STEP 19 — 1-DAY FORECAST ONLY (FULL-DATA DEPLOYMENT MODEL)
% ==========================================================

sequence = Xall_norm(end-window+1:end,:);
pred1 = predict(net_lstm_full,{sequence'});

forecastDate = time(end) + caldays(1);

meanW = pred1(1) * sigmaY(1) + muY(1);
minW  = pred1(2) * sigmaY(2) + muY(2);
maxW  = pred1(3) * sigmaY(3) + muY(3);

% De-normalize sin/cos correctly
sinD = pred1(4) * sigmaY(4) + muY(4);
cosD = pred1(5) * sigmaY(5) + muY(5);

dir = atan2d(sinD,cosD);
if dir < 0
    dir = dir + 360;
end

disp(" ")
disp("======================================")
disp(" GA-OPTIMIZED LSTM — 1 DAY FORECAST ")
disp("======================================")
fprintf('Forecast Date    : %s\n', datestr(forecastDate))
fprintf('Mean Wind Speed  : %.2f km/h\n', meanW)
fprintf('Min Wind Speed   : %.2f km/h\n', minW)
fprintf('Max Wind Speed   : %.2f km/h\n', maxW)
fprintf('Wind Direction   : %.1f degrees\n', dir)
fprintf('RMSE of mean wind speed only : %.4f\n', rmse_lstm)
fprintf('--------------------------------------\n')

%% ==========================================================
% STEP 19A — LAST 10 DAYS ACTUAL + PREDICTED + 1 DAY FORECAST
% ==========================================================

lastN = 10;

actual_last10 = Y_raw(end-lastN+1:end,1);
time_last10   = targetTime(end-lastN+1:end);

pred_last10   = pred_full_mean(end-lastN+1:end);

future_time = forecastDate;
future_pred = meanW;

figure('Name','Last 10 Days Actual vs Predicted + 1 Day Forecast','NumberTitle','off')
plot(time_last10, actual_last10,'b-o','LineWidth',1.5)
hold on
plot(time_last10, pred_last10,'r-o','LineWidth',1.5)
plot(future_time, future_pred,'ks','MarkerSize',9,'MarkerFaceColor','y')

legend('Actual Mean Wind','Predicted Mean Wind','1-Day Forecast','Location','best')
title('Actual vs Predicted Mean Wind Speed (Last 10 Days + 1 Day Forecast)')
xlabel('Date')
ylabel('Wind Speed (km/h)')
grid on
xlim([time_last10(1) future_time])


%% ==========================================================
% STEP 19B — RMSE COMPARISON
% ==========================================================

figure('Name','RMSE Comparison of Models','NumberTitle','off')
bar([rmse_ann, rmse_plain, rmse_lstm])
set(gca,'XTickLabel',{'ANN','LSTM','GA-LSTM'})
ylabel('RMSE (Mean Wind Speed Only)')
title('RMSE Comparison of ANN, LSTM, and GA-LSTM')
grid on


%% ==========================================================
% STEP 19C — 1-DAY MEAN WIND COMPARISON
% ==========================================================

actual_bhubaneswar_mean = 13.3 ;   % user-provided actual value

figure('Name','Mean Wind Prediction Comparison','NumberTitle','off')
bar([actual_bhubaneswar_mean, ann_meanW, plain_meanW, meanW])
set(gca,'XTickLabel',{'Actual','ANN','LSTM','GA-LSTM'})
ylabel('Mean Wind Speed (km/h)')
title('1-Day Mean Wind Prediction Comparison')
grid on


%% ==========================================================
% STEP 20 — ACTUAL VS PREDICTED MEAN WIND (FULL DATASET)
% ==========================================================

figure('Name','Actual vs Predicted Mean Wind Speed (Full Dataset)','NumberTitle','off')
plot(targetTime, Y_raw(:,1),'Color',[0.85 0.85 0.85],'LineWidth',1.0)
hold on
plot(time_all_seq, pred_full_mean,'r','LineWidth',1.4)
legend('Actual Mean Wind','Predicted Mean Wind','Location','best')
title('Actual vs Predicted Mean Wind Speed (Full Dataset)')
xlabel('Date')
ylabel('Wind Speed (km/h)')
grid on
xlim([time(1) time(end)])

%% ==========================================================
% STEP 21 — ACTUAL VS PREDICTED MEAN WIND (TEST SET, FULL X-AXIS)
% ==========================================================

figure('Name','Actual vs Predicted Mean Wind Speed Test','NumberTitle','off')
plot(full_time_axis, true_test_full,'b','LineWidth',1.5)
hold on
plot(full_time_axis, pred_test_full,'r','LineWidth',1.5)
legend('Actual Wind','Predicted Wind','Location','best')
title('Actual vs Predicted Mean Wind Speed (Test Set)')
xlabel('Date')
ylabel('Wind Speed (km/h)')
grid on
xlim([time(1) time(end)])


%% ==========================================================
% STEP 22 — WIND TREND ANALYSIS
% ==========================================================

%figure('Name','Wind Trend Analysis','NumberTitle','off')

%windTrend = movmean(actual_full_mean,30);

%plot(time_all_seq, actual_full_mean,'Color',[0.75 0.75 0.75],'LineWidth',1.0)
%hold on
%plot(time_all_seq, windTrend,'r','LineWidth',2)

%legend('Raw Wind','30-Day Moving Average','Location','best')
%title('Wind Trend Analysis')
%xlabel('Date')
%ylabel('Wind Speed (km/h)')
%grid on
%xlim([time(1) time(end)])

%% ==========================================================
% STEP 23 — FREQUENCY SPECTRUM (FFT)
% ==========================================================

%wind_signal = actual_full_mean;
%N = length(wind_signal);

%fft_result = abs(fft(wind_signal));
%freq = (0:N-1)/N;

%figure('Name','Frequency Spectrum (FFT)','NumberTitle','off')
%plot(freq(1:floor(N/2)), fft_result(1:floor(N/2)),'LineWidth',1.5)

%title('Frequency Spectrum (FFT)')
%xlabel('Normalized Frequency')
%ylabel('Amplitude')
%grid on

%% ==========================================================
% STEP 24 — ACTUAL VS PREDICTED WIND SPEED SCATTER PLOT
% ==========================================================

%figure('Name','Actual vs Predicted Wind Speed Scatter Plot','NumberTitle','off')

%scatter(true_test_mean, pred_test_mean, 18, 'filled')
%hold on

%minVal = min([true_test_mean; pred_test_mean]);
%maxVal = max([true_test_mean; pred_test_mean]);

%plot([minVal maxVal],[minVal maxVal],'r--','LineWidth',1.5)

%title('Actual vs Predicted Wind Speed Scatter Plot')
%xlabel('Actual Mean Wind Speed (km/h)')
%ylabel('Predicted Mean Wind Speed (km/h)')
%legend('Predicted Points','Ideal 1:1 Line','Location','best')
%grid on
%axis equal
%xlim([minVal maxVal])
%ylim([minVal maxVal])

%% ==========================================================
% STEP 25A — FULL-HISTORY RESIDUAL ERROR PREDICTION ERROR
% ==========================================================

error_full_history = actual_full_mean - pred_full_mean;

figure('Name','Prediction Error Over Full Dataset','NumberTitle','off')
plot(time_all_seq, error_full_history,'k','LineWidth',1.0)
hold on
yline(0,'r--','LineWidth',1.3)

legend('Prediction Error','Zero Error Reference','Location','best')
title('Prediction Error Over Full Dataset')
xlabel('Date')
ylabel('Error (km/h)')
grid on
xlim([time_all_seq(1) time_all_seq(end)])


%% ==========================================================
% STEP 26A — FULL-HISTORY ABSOLUTE ERROR AND SMOOTHED ERROR
% ==========================================================

%abs_error_full_history = abs(error_full_history);
%smooth_abs_error_full_history = movmean(abs_error_full_history,30);

%figure('Name','Full-History Absolute Error and Smoothed Error','NumberTitle','off')
%plot(time_all_seq, abs_error_full_history,'Color',[0.8 0.8 0.8])
%hold on
%plot(time_all_seq, smooth_abs_error_full_history,'r','LineWidth',2)

%legend('Daily Absolute Error','30-Day Moving Average','Location','best')
%title('Full-History Absolute Error and Smoothed Error Trend')
%xlabel('Date')
%ylabel('Absolute Error (km/h)')
%grid on
%xlim([time_all_seq(1) time_all_seq(end)])



%% ==========================================================
% STEP 27A — FULL-HISTORY ROLLING RMSE
% ==========================================================

roll_window_full = 30;

rmse_roll_full_history = zeros(length(actual_full_mean)-roll_window_full+1,1);
time_rmse_full_history = time_all_seq(roll_window_full:end);

for i = 1:length(rmse_roll_full_history)
    seg_true = actual_full_mean(i:i+roll_window_full-1);
    seg_pred = pred_full_mean(i:i+roll_window_full-1);

    rmse_roll_full_history(i) = sqrt(mean((seg_true - seg_pred).^2));
end

figure('Name','Rolling RMSE Over Full Dataset','NumberTitle','off')
plot(time_rmse_full_history, rmse_roll_full_history,'b','LineWidth',1.4)

title('Rolling RMSE Over Time')
xlabel('Date')
ylabel('RMSE (km/h)')
grid on
xlim([time_rmse_full_history(1) time_rmse_full_history(end)])



%% ==========================================================
% STEP 28 — ACTUAL VS PREDICTED WIND DIRECTION
% ==========================================================

%actual_dir_full = atan2d(Yall_seq(:,4), Yall_seq(:,5));
%actual_dir_full(actual_dir_full < 0) = actual_dir_full(actual_dir_full < 0) + 360;

%pred_dir_full = atan2d(Ypred_all(:,4), Ypred_all(:,5));
%pred_dir_full(pred_dir_full < 0) = pred_dir_full(pred_dir_full < 0) + 360;

%figure('Name','Actual vs Predicted Wind Direction','NumberTitle','off')
%plot(time_all_seq, actual_dir_full,'b','LineWidth',1.0)
%hold on
%plot(time_all_seq, pred_dir_full,'r','LineWidth',1.0)
%plot(forecastDate, dir,'ko','MarkerFaceColor','y','MarkerSize',7)
%legend('Actual Direction','Predicted Direction','1-Day Forecast','Location','best')
%title('Actual vs Predicted Wind Direction (Full Dataset)')
%xlabel('Date')
%ylabel('Wind Direction (degrees)')
%grid on
%xlim([time(1) time(end)])

%% ==========================================================
% STEP 29 — WIND DIRECTION DISTRIBUTION
% ==========================================================

%figure('Name','Wind Direction Distribution','NumberTitle','off')
%polarhistogram(deg2rad(actual_dir_full),16)
%title('Wind Direction Distribution')

%% ==========================================================
% STEP 30 — POLAR FORECAST PLOT
% ==========================================================

figure('Name','1-Day Forecast: Wind Direction vs Mean Wind Speed','NumberTitle','off')
polarplot(deg2rad(dir), meanW, 'ro','LineWidth',2,'MarkerSize',8)
title('1-Day Forecast: Wind Direction vs Mean Wind Speed')

%% ==========================================================
% STEP 31 — POLAR ACTUAL VS PREDICTED SAMPLE
% ==========================================================

sampleStep = max(1, floor(length(actual_dir_full)/200));

figure('Name','Polar Actual vs Predicted Sample','NumberTitle','off')
polarplot(deg2rad(actual_dir_full(1:sampleStep:end)), actual_full_mean(1:sampleStep:end), 'b.')
hold on
polarplot(deg2rad(pred_dir_full(1:sampleStep:end)), pred_full_mean(1:sampleStep:end), 'r.')
legend('Actual','Predicted','Location','best')
title('Polar Actual vs Predicted Wind Pattern Sample')

%% ==========================================================
% STEP 32 — CLEANUP
% ==========================================================

if isappdata(0,'trainHistory')
    rmappdata(0,'trainHistory');
end



%% ==========================================================
% FUNCTION — GA FITNESS (FAST VALIDATION SCORING)
% ==========================================================
function errorVal = galstmFitness(params, Xtrain_lstm, Ytrain_lstm, Xval_lstm, Yval_lstm)

try
    numHiddenUnits = min(max(48, round(params(1))), 96);
    maxEpochs      = min(max(18, round(params(2))), 40);
    dropRate       = min(max(params(3),0.03),0.12);
    learnRate      = min(max(params(4),2e-4),7e-4);


    inputSize = size(Xtrain_lstm{1},1);
    numResponses = size(Ytrain_lstm,2);

    layers = [
        sequenceInputLayer(inputSize)

        lstmLayer(numHiddenUnits,'OutputMode','sequence')
        dropoutLayer(dropRate)

        lstmLayer(max(12,round(numHiddenUnits/2)),'OutputMode','last')
        dropoutLayer(dropRate)

        fullyConnectedLayer(48)
        reluLayer

        fullyConnectedLayer(numResponses)
        regressionLayer
    ];

    % FAST TRAINING FOR GA ONLY
    % no validation during temporary GA training
    options = trainingOptions('adam', ...
        'MaxEpochs',maxEpochs, ...
        'MiniBatchSize',32, ...
        'GradientThreshold',1, ...
        'InitialLearnRate',learnRate, ...
        'LearnRateSchedule','piecewise', ...
        'LearnRateDropFactor',0.5, ...
        'LearnRateDropPeriod',6, ...
        'L2Regularization',1e-4, ...
        'Shuffle','never', ...
        'ExecutionEnvironment','cpu', ...
        'Verbose',0);

    net = trainNetwork(Xtrain_lstm, Ytrain_lstm, layers, options);

    % evaluate on validation set after training
    Ypred_val = predict(net, Xval_lstm);

if any(isnan(Ypred_val),'all') || any(isinf(Ypred_val),'all')
    errorVal = 1e6;
else
    pred_mean_val = Ypred_val(:,1);
    true_mean_val = Yval_lstm(:,1);
    errorVal = sqrt(mean((pred_mean_val - true_mean_val).^2));
end

    catch
    errorVal = 1e6;
end

end

%% ==========================================================
% FUNCTION — CAPTURE TRAINING HISTORY
% ==========================================================
function stop = captureTrainingInfo(info)

stop = false;

if ~isappdata(0,'trainHistory')
    return
end

trainHistory = getappdata(0,'trainHistory');

if strcmp(info.State,"iteration")

    trainHistory.iteration(end+1,1) = info.Iteration;
    trainHistory.epoch(end+1,1)     = info.Epoch;

    % Training loss
    if isfield(info,'TrainingLoss') && ~isempty(info.TrainingLoss)
        trainHistory.trainingLoss(end+1,1) = info.TrainingLoss;
    else
        trainHistory.trainingLoss(end+1,1) = NaN;
    end

    % Training RMSE
    if isfield(info,'TrainingRMSE') && ~isempty(info.TrainingRMSE)
        trainHistory.trainingRMSE(end+1,1) = info.TrainingRMSE;
    elseif isfield(info,'TrainingLoss') && ~isempty(info.TrainingLoss)
        trainHistory.trainingRMSE(end+1,1) = sqrt(max(info.TrainingLoss,0));
    else
        trainHistory.trainingRMSE(end+1,1) = NaN;
    end

    % Validation loss
    if isfield(info,'ValidationLoss') && ~isempty(info.ValidationLoss)
        trainHistory.validationLoss(end+1,1) = info.ValidationLoss;
    else
        trainHistory.validationLoss(end+1,1) = NaN;
    end

    % Validation RMSE
    if isfield(info,'ValidationRMSE') && ~isempty(info.ValidationRMSE)
        trainHistory.validationRMSE(end+1,1) = info.ValidationRMSE;
    else
        trainHistory.validationRMSE(end+1,1) = NaN;
    end

    setappdata(0,'trainHistory',trainHistory);
end

end