using Revise, TensorKit, Zygote
using JLD2,ChainRulesCore,MAT
using KrylovKit
using JSON
using Random
using Zygote:@ignore_derivatives
using Dates

cd(@__DIR__)




include("..\\..\\src\\bosonic\\Settings.jl")
include("..\\..\\src\\bosonic\\Settings_cell.jl")
include("..\\..\\src\\bosonic\\iPEPS_ansatz.jl")
include("..\\..\\src\\bosonic\\AD_lib.jl")
include("..\\..\\src\\bosonic\\line_search_lib.jl")
include("..\\..\\src\\bosonic\\line_search_lib_cell.jl")
include("..\\..\\src\\bosonic\\optimkit_lib.jl")
include("..\\..\\src\\bosonic\\CTMRG.jl")
include("..\\..\\src\\fermionic\\Fermionic_CTMRG.jl")
include("..\\..\\src\\fermionic\\Fermionic_CTMRG_unitcell.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_model_cell.jl")
include("..\\..\\src\\fermionic\\square_Hubbard_AD_cell.jl")
include("..\\..\\src\\fermionic\\swap_funs.jl")
include("..\\..\\src\\fermionic\\fermi_permute.jl")
include("..\\..\\src\\fermionic\\mpo_mps_funs.jl")
include("..\\..\\src\\fermionic\\double_layer_funs.jl")
include("..\\..\\src\\fermionic\\triangle_fiPESS_method.jl")
include("..\\..\\src\\fermionic\\simple_update\\fermi_triangle_SimpleUpdate.jl")
include("..\\..\\src\\fermionic\\simple_update\\fermi_triangle_SimpleUpdate_iPESS.jl")

###########################
"""
ABABABAB
CDCDCDCD
ABABABAB
CDCDCDCD
"""
###########################
# let

Mcell=3;#magnetic unit-cell
t1=1;
t2=1;
μ=0.2;
U=0;
parameters=Dict([("t1", t1),("t2", t2), ("μ",  μ), ("U",  U), ("M", Mcell)]);
println("parameters:");
println(parameters);

energy_setting=Triangle_Hofstadter_Hubbard_settings();
energy_setting.model = "Triangle_Hofstadter_spinless";
energy_setting.Magnetic_cell=parameters["M"];
dump(energy_setting);

Lx=Mcell;
if Lx==4
# ex_set=ComplexF64[0.16894647336486587 - 0.000848878596418392im 0.1685898329563909 - 0.000830317421741874im; 0.17518747304925908 - 0.0020528207919036276im 0.17502712495751896 - 0.0020507939514248672im; 0.16895024156077876 - 0.0009039615410462967im 0.16868308960226502 - 0.0008406689153681149im; 0.1751437028705141 - 0.002095084965559436im 0.17522913924107836 - 0.0018810458223066549im]
# ey_set=ComplexF64[-0.17396304402448018 - 0.009803725150885404im -0.17387513784108521 - 0.00982656503457158im; -0.003942705897364502 + 0.17294548465620596im -0.003986412316409845 + 0.173080716836371im; 0.17397569139812746 + 0.009883256485923921im 0.1739765247467986 + 0.009953701077383244im; 0.003943191474216484 - 0.17308716580921157im 0.003982991544332689 - 0.17320584184793006im]
# e_t2_set=ComplexF64[0.1189616638789216 + 0.11928716200335808im 0.11920661439367944 + 0.11958791545953783im; 0.1250325572222013 - 0.1228171900668329im 0.125166397824796 - 0.12286192136900656im; -0.11906527876016415 - 0.11930385881304018im -0.1191858071047888 - 0.11969540793235038im; -0.12499566254044743 + 0.12271897536343541im -0.1250633043753007 + 0.12306326050049121im]
# e0_set=ComplexF64[0.6148742516663902 - 4.021703732282307e-13im 0.6150404526131388 + 8.103090046456483e-14im; 0.38475160200917813 + 9.755652793591459e-14im 0.3847715758496374 + 2.6122811654072318e-14im; 0.6150833074193207 - 3.0511216114216327e-13im 0.6151826871191161 - 1.3353905107465732e-13im; 0.3848887657588952 - 5.974666301930537e-13im 0.3848944698075399 - 3.9975726549674524e-13im]
# eU_set=ComplexF64[-0.057437125833194665 + 2.0116732512727347e-13im -0.0575202263065698 - 4.055913620619609e-14im; 0.057624198995412435 - 4.880308918328099e-14im 0.05761421207518141 - 1.3106321618640769e-14im; -0.057541653709656214 + 1.525388553935414e-13im -0.05759134355955785 + 6.672376535062158e-14im; 0.05755561712055245 + 2.9871683542636166e-13im 0.057552765096228115 + 1.9989256140904082e-13im]


ex_set=ComplexF64[0.15368918328559708 + 0.004372810953659442im 0.1394469439916616 + 0.0034912516648183995im; 0.1505612470299327 + 0.006924912986103254im 0.20074238114215323 + 0.012159308874991514im; 0.1519035852231949 + 0.009896230838794935im 0.1573886941675235 - 0.0028552007125725995im; 0.19392151821287923 + 0.015367451886578612im 0.16996421648026186 + 0.0020460659289922374im]
ey_set=ComplexF64[-0.18392825226826603 + 0.0005506135256324846im -0.15245081090171866 + 0.012442499558815538im; 0.009875164637547928 - 0.2002721855703767im -0.007898336140997483 - 0.1310108611596066im; 0.149155419712795 - 0.02192122714931397im 0.175511214939477 + 0.008114877471267466im; 0.016036312012297018 + 0.1464237858678081im -0.003898968456380875 + 0.20499812047825125im]
e_t2_set=ComplexF64[0.13257670727518842 + 0.11814856977035844im 0.11525100468194448 + 0.12249439192283954im; 0.11601529999449921 - 0.1335482387228622im 0.08749815227389086 - 0.10719247878826883im; -0.12029914911661642 - 0.1327963515746769im -0.14329914338427194 - 0.12359608512501355im; -0.1035510782553481 + 0.12408801269112418im -0.11919212832808798 + 0.1400831252008727im]
e0_set=ComplexF64[0.5193013138905143 + 7.921916661209951e-8im 0.47372616295835546 - 6.473497268241574e-8im; 0.5578391884627136 + 1.735160914674674e-7im 0.45690440105199565 + 1.7455168135154725e-7im; 0.4883378585830664 + 3.696195434559907e-7im 0.5196231321545679 + 3.3227878840782736e-7im; 0.44890471972963236 + 2.839509992039199e-7im 0.5353092551106245 - 1.5215417721945552e-7im]
eU_set=ComplexF64[-0.009650656945257347 - 3.960958331484997e-8im 0.01313691852082203 + 3.236748634228542e-8im; -0.028919594231357565 - 8.675804570504096e-8im 0.021547799474001733 - 8.727584067533353e-8im; 0.005831070708467542 - 1.8480977172455727e-7im -0.009811566077284374 - 1.661393941916341e-7im; 0.025547640135182598 - 1.419754995999611e-7im -0.01765462755531213 + 7.607708862287246e-8im]
elseif Lx==3
# ex_set=ComplexF64[0.2138811963573229 + 0.00017679126803742083im 0.2166025573256273 + 0.002989867146285191im; 0.1400779534204635 + 0.05176899191047403im 0.13965952417589783 + 0.05322353717873483im; 0.1468497715809519 - 0.05588740095561432im 0.13852649801581898 - 0.05916300679890127im]
# ey_set=ComplexF64[-0.1062474515239233 + 0.09896771488601294im -0.10298712931502609 + 0.0956488776522325im; 0.20766258417412795 + 0.003977345742191901im 0.20753809510708016 + 0.001016367863467821im; -0.10078995969484504 - 0.10567871020916916im -0.10488814789489749 - 0.10140347841986562im]
# e_t2_set=ComplexF64[0.2177470503400536 + 0.0023636982000705036im 0.22465259921309116 + 0.006277841533625958im; -0.10880642572652587 - 0.08644144471303349im -0.11281273019886451 - 0.08485528481044301im; -0.12060886495743632 + 0.08784480269103379im -0.1233482695478984 + 0.0854283743361416im]
# e0_set=ComplexF64[0.3458369788739935 + 9.819145987845279e-17im 0.33841060394194633 + 6.510312391712222e-17im; 0.3295716556985701 - 3.5725784544953584e-17im 0.33209848874788467 - 7.54281867930738e-18im; 0.331689894265131 + 2.4863539327768854e-17im 0.3228084379416875 + 2.57081397243006e-16im]
# eU_set=ComplexF64[0.07708151056300111 - 3.526905223566855e-17im 0.08079469802902703 - 2.356793760228524e-17im; 0.08521417215071546 + 2.2732500100243875e-17im 0.08395075562605629 - 4.700620653036781e-18im; 0.08415505286743359 - 5.3491113744929403e-17im 0.08859578102915565 - 1.314698234107039e-16im]



ex_set=ComplexF64[0.16973558587499823 - 0.004312512779347184im 0.16088370348012537 + 0.003589917521233868im; 0.18394730455304661 - 0.0034603122226915086im 0.18521414209103346 - 0.0015891277232101973im; 0.19205439430828355 + 0.0008622226110330702im 0.19022699478688096 + 0.00315983492021926im]
ey_set=ComplexF64[-0.10105526761400171 - 0.1528312113186824im -0.09031560616830402 - 0.16301680104769686im; 0.18250049958487038 + 0.0027030389156104707im 0.17722863458158206 + 0.0010952610517675344im; -0.09168949995095725 + 0.16342590202344262im -0.07880260623849626 + 0.15412182878530786im]
e_t2_set=ComplexF64[0.18164149882283562 - 0.007575652070450475im 0.16676945759837478 - 0.0048774532292259086im; -0.09971802868441414 - 0.1543573804695679im -0.08955055765098649 - 0.15584184776274287im; -0.08633615467788498 + 0.1626945374371535im -0.08593408974828663 + 0.1632150468783765im]
e0_set=ComplexF64[0.3261645240694412 - 3.3054840082883713e-16im 0.3228302567919162 + 5.466541233322724e-16im; 0.34432235778076475 - 3.2334850341681427e-15im 0.3171847536879716 + 1.840160554062109e-16im; 0.3521633410719101 - 7.373436041872785e-15im 0.33730648738370345 - 1.8142610405544712e-15im]
eU_set=ComplexF64[0.08691773796527985 + 1.7789176914634814e-16im 0.08858487160404148 - 2.5317277174270065e-16im; 0.07783882110961815 + 1.6614398658167618e-15im 0.09140762315601393 - 6.633555898875764e-17im; 0.07391832946404404 + 3.658459999288441e-15im 0.08134675630814833 + 9.31039845177103e-16im]
end

pasrmeters_site=@ignore_derivatives get_Hofstadter_coefficients(Lx,Ly,parameters,energy_setting);
tx_coe_set=pasrmeters_site["tx_coe_set"];
ty_coe_set=pasrmeters_site["ty_coe_set"];
t2_coe_set=pasrmeters_site["t2_coe_set"];
U_coe_set=pasrmeters_site["U_coe_set"];
μ_coe_set=pasrmeters_site["μ_coe_set"];

E_total=0;
for ca=1:Lx
    for cb=1:Ly
        E_total=E_total+tx_coe_set[ca,cb]*ex_set[ca,cb]+ty_coe_set[ca,cb]*ey_set[ca,cb]+t2_coe_set[ca,cb]*e_t2_set[ca,cb];
        println([tx_coe_set[ca,cb]*ex_set[ca,cb],ty_coe_set[ca,cb]*ey_set[ca,cb],t2_coe_set[ca,cb]*e_t2_set[ca,cb]])
        # println([ex_set[ca,cb],ey_set[ca,cb],e_t2_set[ca,cb]])
        println(imag(log(ex_set[ca,cb]*ey_set[ca,cb]*e_t2_set[ca,cb])))
    end
end
E_total=E_total/Lx/Ly