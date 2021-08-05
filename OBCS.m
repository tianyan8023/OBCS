function [w] = OBCS(data,y,tau_w,K,l_max) %OBCS algorithm function
% data is the traininng data
% y is the labels corresponding to the data
% tau_w is the step size 
% K is the sparsity of feature selection vector
% l_max is the number of the iterations for obtaining w 
% w is the reconstructed feature selection vector    
    [M,N] = size(data);
    D = data;
    w = zeros(N,1);   

    s_lim = 0;
    s = Inf; 
    i = 0;
    % Function A = sign(D*x)
    A = @(in) sign(D*in); 
    % Iterations for obtaining scattered feature selection vector
    while (s_lim < s) && (i < l_max)
        % Compute Gradient
        g = D'*(A(w) - y);
        % Step
        w0 = w - tau_w.*g;        
        % Best K-term (threshold)
        [trash, aidx] = sort(abs(w0), 'descend');
        w0(aidx(K+1:end)) = 0;
        % Update w      
        w = w0;     
        % Threshold
        s = nnz(y - sign(D*w));
        i = i+1;
    end     
end