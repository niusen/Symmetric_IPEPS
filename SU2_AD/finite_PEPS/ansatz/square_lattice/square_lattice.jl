function iPEPS_C4(iPEPS_full::Matrix{TensorMap}, Lx,Ly)
    @assert Lx==Ly;
    @assert mod(Lx,2)==0;

    iPEPS_=iPEPS_full[1:Int(Lx/2),1:Int(Ly/2)];
    return iPEPS_

end

function iPEPS_from_c4_corner(iPEPS_corner::Matrix{TensorMap},Lx,Ly)
    @assert size(iPEPS_corner,1)*2==Lx;
    @assert size(iPEPS_corner,2)*2==Ly;
    @assert Lx==Ly;

    R=[0 1;-1 0];

    psi=Matrix{TensorMap}(undef,Lx,Ly);
    for cx=1:Int(Lx/2)
        for cy=1:Int(Ly/2)

            psi[cx,cy] = iPEPS_corner[cx,cy];

            coord=[cx-Lx/2-0.5,cy-Ly/2-0.5];
            coord_new=R*coord;
            coord_new[1]=coord_new[1]+Lx/2+0.5;
            coord_new[2]=coord_new[2]+Ly/2+0.5;
            psi[Int(coord_new[1]),Int(coord_new[2])] = rotate_90(iPEPS_corner[cx,cy]);

            coord=[cx-Lx/2-0.5,cy-Ly/2-0.5];
            coord_new=R*R*coord;
            coord_new[1]=coord_new[1]+Lx/2+0.5;
            coord_new[2]=coord_new[2]+Ly/2+0.5;
            psi[Int(coord_new[1]),Int(coord_new[2])] = rotate_90(rotate_90(iPEPS_corner[cx,cy]));

            coord=[cx-Lx/2-0.5,cy-Ly/2-0.5];
            coord_new=R*R*R*coord;
            coord_new[1]=coord_new[1]+Lx/2+0.5;
            coord_new[2]=coord_new[2]+Ly/2+0.5;
            psi[Int(coord_new[1]),Int(coord_new[2])] = rotate_90(rotate_90(rotate_90(iPEPS_corner[cx,cy])));

        end
    end
    return psi
end


function rotate_90(A)
    #clockwise rotation
    if isa(A,AbstractTensorMap{GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}, 5,0})
        A_=permute(A,(2,3,4,1,5,));
    # elseif isa(A,AbstractTensorMap{GradedSpace{SU2Irrep, TensorKit.SortedVectorDict{SU2Irrep, Int64}}, 4,0})
    end
    return A_
end