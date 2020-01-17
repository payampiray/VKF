function [predictions, signals] = vkf(outcomes,lambda,v0,sigma2)
% Volatile Kalman Filter (VKF) for continuous outcomes
% [predictions, signals] = vkf(lambda,sigma2,v0,outcomes)
% Inputs: 
%       outcomes: column-vector of outcomes
%       0<lambda<1, volatility learning rate 
%       v0>0, initial volatility
%       sigma2>0, outcome noise
% Outputs:
%       predictions: predicted state
%       signals: a struct that contains signals that might be useful:
%               predictions
%               volatility
%               learning rate
%               prediction error
%               volatility prediction error
% 
% Note: outputs of VKF also depends on initial variance (w0), which is
% assumed here w0 = sigma2
% 
% See the following paper and equations therein
% Piray and Daw, "A simple model for learning in volatile environments"
% https://doi.org/10.1101/701466

if lambda<=0 || lambda>=1
    error('lambda should be in the unit range');
end
if sigma2<=0
    error('sigma2 should be positive');
end
if v0<=0
    error('v0 should be positive');
end
    

w0      = sigma2;
[T,C] = size(outcomes);
% T: number of trials
% C: number of cues

m       = zeros(1,C);
w       = w0*ones(1,C);
v       = v0*ones(1,C);

predictions = nan(T,C);
learning_rate = nan(T,C);
volatility = nan(T,C);
prediction_error = nan(T,C);
volatility_error = nan(T,C);

for t  = 1:T      
    o = outcomes(t,:);
    predictions(t,:) = m;    
    volatility(t,:) = v;
        
    mpre        = m;
    wpre        = w;
    
    delta_m     = o - m;
    k           = (w+v)./(w+v + sigma2);                            % Eq 9
    m           = m + k.*delta_m;                                   % Eq 10
    w           = (1-k).*(w+v);                                     % Eq 11
    
    wcov        = (1-k).*wpre;                                      % Eq 12
    delta_v     = (m-mpre).^2 + w + wpre - 2*wcov - v;
    v           = v +lambda.*delta_v;                               % Eq 13
    
    learning_rate(t,:) = k;
    prediction_error(t,:) = delta_m;
    volatility_error(t,:) = delta_v;
end

signals = struct('predictions',predictions,'volatility',volatility,'learning_rate',learning_rate,...
                 'prediction_error',prediction_error,'volatility_prediction_error',volatility_error);

end
