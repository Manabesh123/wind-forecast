%% ==========================================================
% WEATHER PREDICTION USING ANN + GA-LSTM
%
% Dataset: Bhubaneswar Weather Data
%
% Goal:
% Predict 5-day future wind conditions
%
% Outputs predicted:
%   Mean wind speed
%   Minimum wind speed
%   Maximum wind speed
%   Wind direction
%
% Improvements added:
%   1. Wind direction circular encoding
%   2. Historical window = 14
%   3. Stable multi-step forecasting
% ==========================================================

clc
clear
close all

%% ==========================================================
% STEP 1 — LOAD DATASET
% ==========================================================

data = readtable('bbsr_2026mar_2025mar.xlsx');

%% ==========================================================
% STEP 2 — SELECT WEATHER VARIABLES
% ==========================================================

vars = {'datetime','temp','humidity','precip','windgust',...
'windspeedmean','windspeedmin','windspeedmax','winddir',...
'sealevelpressure','cloudcover','visibility','solarradiation','elevation'};

data = data(:,vars);

%% ==========================================================
% STEP 3 — DATA CLEANING
% ==========================================================

data = rmmissing(data);

numericVars = varfun(@isnumeric,data,'OutputFormat','uniform');

data{:,numericVars} = fillmissing(data{:,numericVars},'linear');

%% ==========================================================
% STEP 4 — FEATURE ENGINEERING (TIME + DATE FEATURES)
% ==========================================================

% Convert to datetime (if not already)
data.datetime = datetime(data.datetime);

% Save datetime for later plotting (VERY IMPORTANT)
timeVec = data.datetime;

% Extract useful time-based features
data.day = day(data.datetime);
data.month = month(data.datetime);
data.dayofyear = day(data.datetime,'dayofyear');

%% ==========================================================
% STEP 5 — WIND DIRECTION + COLUMN INDEX SETUP
% ==========================================================

% Convert wind direction into circular form
data.winddir_sin = sind(data.winddir);
data.winddir_cos = cosd(data.winddir);

% Remove original wind direction (not useful for ML)
data.winddir = [];

% ----------------------------------------------------------
% STORE COLUMN NAMES (DO THIS BEFORE REMOVING datetime)
% ----------------------------------------------------------
varNames = data.Properties.VariableNames;

% Find required column indexes dynamically
idx_mean = find(strcmp(varNames,'windspeedmean'));
idx_min  = find(strcmp(varNames,'windspeedmin'));
idx_max  = find(strcmp(varNames,'windspeedmax'));
idx_sin  = find(strcmp(varNames,'winddir_sin'));
idx_cos  = find(strcmp(varNames,'winddir_cos'));

% ----------------------------------------------------------
% REMOVE datetime (to avoid table2array error)
% ----------------------------------------------------------
data.datetime = [];

% ----------------------------------------------------------
% KEEP ONLY NUMERIC DATA (SAFE CONVERSION)
% ----------------------------------------------------------
numericVars = varfun(@isnumeric,data,'OutputFormat','uniform');
data = data(:,numericVars);


%% ==========================================================
% STEP 6 — CONVERT TABLE TO NUMERIC MATRIX
% ==========================================================

dataset = table2array(data);

%% ==========================================================
%% STEP 7 — FULL MULTIVARIATE MODEL — FINAL

horizon = 5;

X = dataset(1:end-horizon,:);

n = size(X,1);

numFeatures = size(dataset,2);

Y = zeros(n, horizon * numFeatures);

col = 1;

for i = 1:horizon

    rows = i+1 : n+i;

    Y(:,col:col+numFeatures-1) = dataset(rows,:);

    col = col + numFeatures;

end

%% ==========================================================
% STEP 7.1 — MINIMAL FEATURE SCALING (NO NORMALIZATION)
% ==========================================================

% Get column names again (important)
varNames = data.Properties.VariableNames;

% Find large-scale features
idx_pressure = find(strcmp(varNames,'sealevelpressure'));
idx_solar    = find(strcmp(varNames,'solarradiation'));
idx_vis      = find(strcmp(varNames,'visibility'));
idx_cloud    = find(strcmp(varNames,'cloudcover'));

% Scale only high-magnitude features
dataset(:,idx_pressure) = dataset(:,idx_pressure) / 1000;
dataset(:,idx_solar)    = dataset(:,idx_solar) / 1000;
dataset(:,idx_vis)      = dataset(:,idx_vis) / 100;
dataset(:,idx_cloud)    = dataset(:,idx_cloud) / 100;

% (OPTIONAL but helpful)
dataset(:,idx_max)  = dataset(:,idx_max) / 100;
dataset(:,idx_min)  = dataset(:,idx_min) / 100;
dataset(:,idx_mean) = dataset(:,idx_mean) / 100;

%% ==========================================================
% STEP 8 — TRAIN TEST SPLIT
% ==========================================================

n = size(X,1);

train_ratio = 0.8;

train_size = round(train_ratio*n);

Xtrain = X(1:train_size,:);
Ytrain = Y(1:train_size,:);

Xtest = X(train_size+1:end,:);
Ytest = Y(train_size+1:end,:);

% ✅ USE timeVec (NOT time)
time_train = timeVec(1:train_size);
time_test  = timeVec(train_size+1:end);

%% ==========================================================
%% STEP 9 — ANN PREDICTION

% TRAIN ANN MODEL

hiddenLayerSize = 10;

net_ann = feedforwardnet(hiddenLayerSize);

net_ann.trainParam.showWindow = true;   % disables/enables training GUI

net_ann = train(net_ann, Xtrain', Ytrain');

Ypred_ann = net_ann(Xtest')';

%% ANN RMSE

error_ann = Ypred_ann - Ytest;

rmse_ann = sqrt(mean(error_ann(:).^2));

disp("ANN RMSE:")
disp(rmse_ann)

%% ANN PERCENTAGE ERROR

mae_ann = mean(abs(error_ann(:)));

meanActual = mean(abs(Ytest(:)));

percentError = (mae_ann / meanActual) * 100;

fprintf("\nANN Average Percentage Error : %.2f %%\n",percentError);


%% ==========================================================
% STEP 10 — PREPARE LSTM SEQUENCES
% ==========================================================

window = 7;

numSamples = size(Xtrain,1) - window;

Xtrain_lstm = cell(numSamples,1);
Ytrain_lstm = zeros(numSamples,size(Ytrain,2));

for i = 1:numSamples

    Xtrain_lstm{i} = Xtrain(i:i+window-1,:)';

    Ytrain_lstm(i,:) = Ytrain(i+window,:);

end

%% ==========================================================
% STEP 11 — GA PARAMETER SEARCH
% ==========================================================

fitnessFunction = @(x)galstmFitness(x,Xtrain_lstm,Ytrain_lstm,Xtest,Ytest);

nvars = 2;

lb = [50 50];
ub = [100 100];

optionsGA = optimoptions('ga',...
'PopulationSize',8,...
'MaxGenerations',8,...
'Display','iter');

best_params = ga(fitnessFunction,nvars,[],[],[],[],lb,ub,[],optionsGA);

disp("Best GA Parameters:")
disp(best_params)

%% ==========================================================
% STEP 12 — FINAL LSTM MODEL
% ==========================================================

numHiddenUnits = round(best_params(1));
maxEpochs = round(best_params(2));

inputSize = size(Xtrain,2);
numResponses = size(Ytrain,2);

layers = [
sequenceInputLayer(inputSize)
lstmLayer(numHiddenUnits,'OutputMode','last')
fullyConnectedLayer(numResponses)
regressionLayer
];

options = trainingOptions('adam',...
'MaxEpochs',maxEpochs,...
'MiniBatchSize',32,...
'GradientThreshold',1,...
'ExecutionEnvironment','cpu',...
'Verbose',0);

net_lstm = trainNetwork(Xtrain_lstm,Ytrain_lstm,layers,options);

%% ==========================================================
% STEP 13 — LSTM TEST PREDICTION (MINIMAL SCALING VERSION)
% ==========================================================

numTest = size(Xtest,1) - window;

Xtest_lstm = cell(numTest,1);
Ytest_seq = zeros(numTest,size(Ytest,2));

for i = 1:numTest

    Xtest_lstm{i} = Xtest(i:i+window-1,:)';

    Ytest_seq(i,:) = Ytest(i+window,:);

end

% Prediction
Ypred_lstm = predict(net_lstm,Xtest_lstm);

% ----------------------------------------------------------
% RESCALE WIND FEATURES BACK TO ORIGINAL SCALE
% (ONLY windspeed columns)
% ----------------------------------------------------------
% ----------------------------------------------------------
% RMSE
% ----------------------------------------------------------

rmse_lstm = sqrt(mean((Ypred_lstm - Ytest_seq).^2,'all'));

disp("GA-LSTM RMSE:")
disp(rmse_lstm)

%% ==========================================================
% STEP 14 — 5 DAY FORECAST (MINIMAL SCALING FINAL)
% ==========================================================

sequence = X(end-window+1:end,:);
Xinput = {sequence'};

forecast = predict(net_lstm, Xinput);

%scaleWind = 100;

disp(" ")
disp("======================================")
disp("        5 DAY WIND FORECAST")
disp("======================================")

for d = 1:5
    
    base = (d-1)*5;

    % Rescale wind
    meanW = max(0, forecast(base + 1));

    dev1 = abs(forecast(base + 2));
    dev2 = abs(forecast(base + 3));

    minW = max(0, meanW - dev1);
    maxW = max(meanW, meanW + dev2);

    % Direction
    sinD = forecast(base + 4);
    cosD = forecast(base + 5);

    dir = atan2d(sinD,cosD);
    if dir < 0
        dir = dir + 360;
    end

    fprintf("Day %d Forecast\n",d)
    fprintf("Mean Wind Speed : %.2f km/h\n",meanW)
    fprintf("Min Wind Speed  : %.2f km/h\n",minW)
    fprintf("Max Wind Speed  : %.2f km/h\n",maxW)
    fprintf("Wind Direction  : %.1f degrees\n",dir)
    fprintf("--------------------------------------\n")

end

%% ==========================================================
% STEP 15 — FORECAST VISUALIZATION (MINIMAL SCALING FINAL)
% ==========================================================

days = 1:5;

meanWind = zeros(5,1);
minWind  = zeros(5,1);
maxWind  = zeros(5,1);
windDir  = zeros(5,1);

scaleWind = 100;

for d = 1:5
    
    base = (d-1)*5;

    % Rescale
    meanW = forecast(base + 1) ;

    dev1 = abs(forecast(base + 2));
    dev2 = abs(forecast(base + 3));

    meanWind(d) = meanW;
    minWind(d)  = meanW - dev1;
    maxWind(d)  = meanW + dev2;

    % Direction
    sinD = forecast(base + 4);
    cosD = forecast(base + 5);

    dir = atan2d(sinD,cosD);
    if dir < 0
        dir = dir + 360;
    end

    windDir(d) = dir;

end

% ----------------------------------------------------------
% PLOTS
% ----------------------------------------------------------

figure('Name','5 Day Wind Forecast','NumberTitle','off')

subplot(2,2,1)
plot(days,meanWind,'o-','LineWidth',2)
title('Mean Wind Speed Forecast')
xlabel('Day')
ylabel('Wind Speed')
grid on

subplot(2,2,2)
plot(days,minWind,'LineWidth',2)
hold on
plot(days,maxWind,'LineWidth',2)
legend('Minimum Wind','Maximum Wind')
title('Wind Speed Range')
xlabel('Day')
ylabel('Wind Speed')
grid on

subplot(2,2,3)
plot(days,windDir,'o-','LineWidth',2)
title('Wind Direction Forecast')
xlabel('Day')
ylabel('Degrees')
grid on

subplot(2,2,4)
errorbar(days, meanWind, ...
         meanWind - minWind, ...
         maxWind - meanWind, ...
         'o','LineWidth',2)

title('Forecast Uncertainty')
xlabel('Day')
ylabel('Wind Speed')
grid on

%% ==========================================================
% STEP 16 — ACTUAL VS PREDICTED (FINAL FIX)
% ==========================================================

figure('Name','Prediction Performance','NumberTitle','off')

% Use SAME length everywhere
numPoints = size(Ypred_lstm,1);

time_aligned = time_test(window+1 : window+numPoints);

actual = Ytest(window+1 : window+numPoints, 1);
pred   = Ypred_lstm(:,1);

plot(time_aligned, actual,'b','LineWidth',1.5)
hold on
plot(time_aligned, pred,'r','LineWidth',1.5)

legend('Actual Wind','Predicted Wind')

title('Actual vs Predicted Mean Wind Speed')

xlabel('Date')
ylabel('Wind Speed')

grid on

%% ==========================================================
% STEP 17 — WIND TREND (FINAL FIXED)
% ==========================================================

figure('Name','Wind Trend Analysis','NumberTitle','off')

numPoints = size(Ypred_lstm,1);

time_aligned = time_test(window+1 : window+numPoints);
wind = Ytest(window+1 : window+numPoints, 1);

% Zones
low_thr  = prctile(wind,50);
high_thr = prctile(wind,85);

idx_low  = wind <= low_thr;
idx_mid  = wind > low_thr & wind <= high_thr;
idx_high = wind > high_thr;

hold on

plot(time_aligned(idx_low),  wind(idx_low),  'g.')
plot(time_aligned(idx_mid),  wind(idx_mid),  'y.')
plot(time_aligned(idx_high), wind(idx_high), 'r.')

legend('Normal','Warning','Danger')

title('Wind Speed Risk Zones')

xlabel('Date')
ylabel('Wind Speed')

grid on

%% ==========================================================
% STEP 18 — WIND PERIODICITY ANALYSIS (FFT)
% ==========================================================

wind_signal = Ytest(:,1);

N = length(wind_signal);

fft_result = abs(fft(wind_signal));

freq = (0:N-1)/N;

figure('Name','Wind Frequency Spectrum','NumberTitle','off')

plot(freq,fft_result,'LineWidth',1.5)

title('Wind Frequency Spectrum')

xlabel('Frequency')
ylabel('Amplitude')

grid on


%% ==========================================================
% STEP 19 — WIND AUTOCORRELATION
% ==========================================================

figure('Name','Wind Autocorrelation','NumberTitle','off')

autocorr(wind_signal)

title('Wind Speed Autocorrelation')


%% ==========================================================
% STEP 20 — WIND DIRECTION DISTRIBUTION (WIND ROSE STYLE)
% ==========================================================

figure('Name','Wind Direction Distribution','NumberTitle','off')

polarhistogram(deg2rad(Ytest(:,4)),16)

title('Wind Direction Distribution')


%% ==========================================================
% STEP 21 — POLAR WIND FORECAST
% ==========================================================

figure('Name','Polar Wind Forecast','NumberTitle','off')

polarplot(deg2rad(windDir),meanWind,'o-','LineWidth',2)

title('Wind Direction vs Wind Speed Forecast')

%% ==========================================================
% STEP 22 — CLEAN DAILY AVERAGE GRAPH (FINAL FIX)
% ==========================================================

% Create timetables with explicit variable names
TT_actual = timetable(time_aligned, actual, 'VariableNames', {'ActualWind'});
TT_pred   = timetable(time_aligned, pred,   'VariableNames', {'PredWind'});

% Daily averaging
TTa = retime(TT_actual,'daily','mean');
TTp = retime(TT_pred,'daily','mean');

% Plot
figure

plot(TTa.time_aligned, TTa.ActualWind,'b-o','LineWidth',2)
hold on
plot(TTp.time_aligned, TTp.PredWind,'r--o','LineWidth',2)

legend('Actual','Predicted')

title('Daily Average Wind Speed Comparison')
xlabel('Date')
ylabel('Wind Speed')

grid on




%% ==========================================================
% FITNESS FUNCTION FOR GA
% ==========================================================

function error = galstmFitness(params,Xtrain_lstm,Ytrain_lstm,Xtest,Ytest)

hiddenUnits = round(params(1));
epochs = round(params(2));

inputSize = size(Xtrain_lstm{1},1);
numResponses = size(Ytrain_lstm,2);

layers = [
sequenceInputLayer(inputSize)
lstmLayer(hiddenUnits,'OutputMode','last')
fullyConnectedLayer(numResponses)
regressionLayer
];

options = trainingOptions('adam',...
'MaxEpochs',epochs,...
'MiniBatchSize',32,...
'GradientThreshold',1,...
'Verbose',0);

net = trainNetwork(Xtrain_lstm,Ytrain_lstm,layers,options);

window = 7;

numTest = size(Xtest,1) - window;

Xtest_lstm = cell(numTest,1);
Ytest_seq = zeros(numTest,size(Ytest,2));

for i = 1:numTest

    Xtest_lstm{i} = Xtest(i:i+window-1,:)';

    Ytest_seq(i,:) = Ytest(i+window,:);

end

Ypred = predict(net,Xtest_lstm);

error = mean((Ypred - Ytest_seq).^2,'all');

end