function [w] = OBPCS(data,y,tau_w,tau_d,K,l_max,t_max)%OBPCS algorithm function
% data is the traininng data
% y is the label corresponding to the data
% tau_w and tao_d is the step size 
% K is the sparsity of feature selection vector
% l_max and t_max are the number of the iterations for obtaining w and D
% w is the reconstructed feature selection vector
    [M,N] = size(data);
    D = data;

    iterN = 25;
    w = zeros(N,1);

     for iteri = iterN:-1:1
        if iteri > 1    
            s_lim = 0;
            s = Inf;
            s_limE = 0;
            sE = Inf;
            i = 0;
            iE = 0;
            
            % Iterations for obtaining feature selection vector
            while (s_lim < s) && (i < l_max)
                % Compute Gradient
                g = D'*(sign(D*w) - y);
                % Step
                w0 = w - tau_w.*g;
                % Best K-term (threshold)
                [trash, aidx] = sort(abs(w0), 'descend');
                w0(aidx(K+1:end)) = 0;           
                % Update w
                w = w0;
                % Measure hammning
                s = nnz(y - sign(D*w));
                i = i+1; 
            end

            % Iterations for removing noise in D 
            while (s_limE < sE) && (iE < t_max)
                % Compute Gradient
                for rowi=1:M
                    gE = (sign(D(rowi,:)*w) - y(rowi))*w;
                    % Step
                    D(rowi,:) = D(rowi,:) - tau_d.*gE.';
                end
                % Measure hammning
                sE = nnz(y - sign(D*w));
                iE = iE+1;
            end       
       else
            break;
       end
     end
end