data=load("test.jld2");
psi_sample=data["psi_sample"];
config=data["config"];
contract_history_=data["contract_history_"]
chi=data["chi"]

contract_history=torus_contract_history(zeros(Int8,L),Matrix{TensorMap}(undef,Lx,Ly));
Norm,trun_errs, _=contract_partial_torus_boundaryMPS(psi_sample,config,contract_history, chi)
@show Norm
Norm,trun_errs, contract_history_new=contract_partial_torus_boundaryMPS(psi_sample,config,contract_history_, chi)
@show Norm
verify_contract_history(psi_sample,contract_history_new, chi);