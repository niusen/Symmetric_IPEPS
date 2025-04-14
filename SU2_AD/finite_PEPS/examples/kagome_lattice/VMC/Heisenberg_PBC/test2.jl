data=load("test.jld2");
sample=data["sample"];
psi_decomposed=data["psi_decomposed"];


A11=sample[1,1];
A21=sample[2,1];
A12=sample[1,2];
A22=sample[2,2];


coe=@tensor A11[2,7,1,5]*A21[1,8,2,6]*A12[4,5,3,7]*A22[3,6,4,8];



A11=psi_decomposed[1,1,1,2,1];
A21=psi_decomposed[2,1,2,1,2];
A12=psi_decomposed[1,2,2,1,2];
A22=psi_decomposed[2,2,1,2,1];


@tensor AAAA[:]:= A11[2,7,1,5,-1]*A21[1,8,2,6,-2]*A12[4,5,3,7,-3]*A22[3,6,4,8,-4];

dense1a=convert(Array,psi_decomposed[1,1,1,2,1])
dense2a=convert(Array,psi_decomposed[2,1,2,1,2])
dense3a=convert(Array,psi_decomposed[1,2,2,1,2])
dense4a=convert(Array,psi_decomposed[2,2,1,2,1])