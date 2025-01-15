function mypinv(T::DiagonalTensorMap)
    epsilon0 = 1e-12
    epsilon=epsilon0*maximum(abs.(diag(convert(Array,T))))^2
    T_new=deepcopy(T);
    
    dat=(T_new.data)./((T_new.data).^2 .+epsilon);

    return DiagonalTensorMap(dat,T.domain)
end