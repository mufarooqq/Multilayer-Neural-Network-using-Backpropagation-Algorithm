% Loading and Initialization of parameters 
clc;
clear all;
close all;

load('TrainData.mat');
load('TestData.mat');
load('GroundTruth.mat');
load('TrueClass.mat');

alpha = 1;
eta = 0.09;
epochs = 50;
predict = zeros(10000,1);

I = 785;
H1 = ('Enter the number of Hidden layer 1 units: ');
H1 =  input(H1);

H2 = ('Enter the number of Hidden layer 2 units: ');
H2 =  input(H2);

O = ('Enter the number of Output layer units: ');
O =  input(O);

% H1 = 75;
% H2 = 50;
% O = 10;

wH = randn(I,H1);
wH2 = randn(H1,H2);
wO = randn(H2,O);

wH = wH./sqrt(H1);
wH2 = wH2./sqrt(H2);
wO = wO./sqrt(O);

E = zeros(1,epochs);
tic;
disp('Training the Neural Network...');

for t = 1:epochs   
    X=['Epoch#',num2str(t)];
    disp(X);
    for k = 1:length(TrainData)

            % Random dataset row selection parameter
            r = round(length(TrainData).*rand(1,1));
            if r == 0
                r = r+1;
            end

            % FORWARD PASS
            v1 = TrainData(r,:)*wH;
            y1 = logsig(v1);
            v2 = y1*wH2;
            y2 = logsig(v2);    
            v3 = y2*wO;
            y3 = logsig(v3);
            
            % ERROR CALCULATION
            e = GroundTruth(r,:)-y3;
            
            % BACKWARD PASS
            dO  = dlogsig(v3,y3).*(e);
            dH2 = diag(dlogsig(v2,y2))*(wO*dO');
            dH1  = diag(dlogsig(v1,y1))*(wH2*dH2);
            
            wO  = alpha.*wO +(eta.*dO'*y2)';
            wH2 = alpha.*wH2 +(eta.*dH2*y1)';
            wH  = alpha.*wH +(eta.*dH1*TrainData(r,:))';
    end
    
    % Error after each epoch
    E(t) = sum(e.^2)/10;
    % Testing the TestData using the updated weight matrices
    v1T = TestData*wH;
    y1T = logsig(v1T);
    v2T = y1T*wH2;
    y2T = logsig(v2T);    
    v3T = y2T*wO;
    y3T = logsig(v3T);
    
    % Predicting the values for each TestData input
    for z = 1:length(TestData)
        [val, col] = max(y3T(z,:));
        predict(z,:) = col-1;
    end
    % Calculating the Classification Rate in percentage
    CR(t) = (sum(predict==TrueClass)/length(TestData))*100;
end
toc;

figure;
plot(1:t,CR,'b--o');
title('Recognition Curve');
xlabel('Epochs');
ylabel('Classification Rate (%)');

figure;
plot(1:t,E,'b--o')
title('Error Curve');
xlabel('Epochs');
ylabel('Squared Error Sum');



