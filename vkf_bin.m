function [predictions, signals] = vkf_bin(outcomes,lambda,v0,omega)
% Volatile Kalman Filter for binary outcomes (binary VKF)
% [predictions, signals] = vkf_bin(lambda,sigma2,v0,outcomes)
% Inputs: 
%       outcomes: column-vector of outcomes
%       0<lambda<1, volatility learning rate
%       v0>0, initial volatility
%       omega>0, noise parameter
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
% assumed here w0 = omega
% 
% See the following paper and equations therein
% Piray and Daw, "A simple model for learning in volatile environments"
% https://doi.org/10.1101/701466

if lambda<=0 || lambda>=1
    error('lambda should be in the unit range');
end
if omega<= 0
    error('omega should be positive');
end
if v0<=0
    error('v0 should be positive');
end  

w0 = omega; 
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

sigmoid = @(x)1./(1+exp(-x));
for t  = 1:T      
    o = outcomes(t,:);
    predictions(t,:) = m;    
    volatility(t,:) = v;    
    
    mpre        = m;
    wpre        = w;
    
    delta_m     = o - sigmoid(m);    
    k           = (w+v)./(w+v+ omega);                              % Eq 14
    alpha       = sqrt(w+v);                                        % Eq 15
    m           = m + alpha.*delta_m;                               % Eq 16
    w           = (1-k).*(w+v);                                     % Eq 17
    
    wcov        = (1-k).*wpre;                                      % Eq 18
    delta_v     = (m-mpre).^2 + w + wpre - 2*wcov - v;    
    v           = v +lambda.*delta_v;                               % Eq 19
    
    learning_rate(t,:) = alpha;
    prediction_error(t,:) = delta_m;
    volatility_error(t,:) = delta_v;    
end


signals = struct('predictions',predictions,'volatility',volatility,'learning_rate',learning_rate,...
                 'prediction_error',prediction_error,'volatility_prediction_error',volatility_error);
end
