using Printf, TOML

"""Write CTM convergence data for (label, directory, filename) entries."""
function write_spin_environment_tables(io,entries,report_dir)
    f(x)=@sprintf("%.10f",x)
    println(io,"| 构造 | χ | E/N | Mx | My | Mz | Mz/M_sat | 角谱收敛 |")
    println(io,"|---|---:|---:|---:|---:|---:|---:|---|")
    for (family,dir,file) in entries
        r=TOML.parsefile(joinpath(dir,file))
        m=r["magnetization_per_site"]
        println(io,"| $family | $(r["chi"]) | $(f(r["energy_per_site"])) | $(f(m[1])) | $(f(m[2])) | $(f(m[3])) | $(f(r["mz_over_saturation"])) | $(r["converged"]) |")
    end
    println(io,"\n## 各次环境检查\n")
    println(io,"收敛列指角谱是否通过该次运行设定的阈值。有限 χ 误差需另看能量和磁化的变化；未收敛记录保留，不当作高精度结果。\n")
    println(io,"| 构造 | χ | cutoff | 规范平衡轮数 | 更新顺序 | E/N | Mz | 角谱误差 | 收敛 | 数据 |")
    println(io,"|---|---:|---:|---:|---|---:|---:|---:|---|---|")
    for (family,dir,_) in entries
        files=filter(n->startswith(n,"heisenberg_chi") && endswith(n,".toml"),readdir(dir))
        sort!(files;by=n->(TOML.parsefile(joinpath(dir,n))["chi"],n))
        for file in files
            r=TOML.parsefile(joinpath(dir,file))
            link=replace(relpath(joinpath(dir,file),report_dir),'\\'=>'/')
            println(io,"| $family | $(r["chi"]) | $(r["projector_tol"]) | $(get(r,"gauge_sweeps",0)) | $(get(r,"update_method","continuous_update")) | $(f(r["energy_per_site"])) | $(f(r["magnetization_per_site"][3])) | $(r["ctm_error"]) | $(r["converged"]) | [TOML]($link) |")
        end
    end
end

function write_local_spin_tables(io,entries,load_result)
    f(x)=@sprintf("%.10f",x)
    for (family,dir,file) in entries
        r=load_result(joinpath(dir,replace(file,".toml"=>".jld2")))
        println(io,"### $family\n\n| x | 子格点 | Mx | My | Mz |\n|---:|---|---:|---:|---:|")
        for x=1:size(r.magnetization,1),y=1:size(r.magnetization,2),s=1:3
            m=real.(r.magnetization[x,y,s,:])
            println(io,"| $x | $(("a","b","c")[s]) | $(f(m[1])) | $(f(m[2])) | $(f(m[3])) |")
        end
        println(io,"\n平均局域极化长度 meanᵢ|〈Sᵢ〉|=$(f(r.mean_local_magnitude))；均匀磁化长度 |meanᵢ〈Sᵢ〉|=$(f(r.uniform_magnitude))。最大观测量虚部=$(r.max_imag)。\n")
    end
end

function write_projected_energy_tables(io,entries)
    f(x)=@sprintf("%.10f",x)
    println(io,"| 构造/存档 | D | χ | E/N | Mz/Msat | 角谱误差 | 角谱收敛 |")
    println(io,"|---|---:|---:|---:|---:|---:|---|")
    for (label,directory,file) in entries
        r=TOML.parsefile(joinpath(directory,file))
        println(io,"| $label | $(r["D"]) | $(r["chi"]) | $(f(r["energy_per_site"])) | $(f(r["mz_over_saturation"])) | $(r["ctm_error"]) | $(r["converged"]) |")
    end
    println(io,"\n表中的打印小数不代表有效位数。角谱收敛与有限 χ 误差是两项不同检查；下面保留各次运行供比较。\n")
    for (label,directory,file) in entries
        println(io,"## $label\n")
        println(io,"选用结果：`$file`；source 元数据与 SHA256：`$directory/source.json`。\n")
        println(io,"| 结果文件 | χ | E/N | Mz/Msat | 迭代数 | 角谱误差 | 收敛 |")
        println(io,"|---|---:|---:|---:|---:|---:|---|")
        files=filter(n->startswith(n,"heisenberg_")&&endswith(n,".toml"),readdir(directory))
        sort!(files;by=n->(TOML.parsefile(joinpath(directory,n))["chi"],n))
        for name in files
            r=TOML.parsefile(joinpath(directory,name))
            println(io,"| $name | $(r["chi"]) | $(f(r["energy_per_site"])) | $(f(r["mz_over_saturation"])) | $(r["iterations"]) | $(r["ctm_error"]) | $(r["converged"]) |")
        end
        println(io)
    end
end
