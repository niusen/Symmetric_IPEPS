include("projected_energy_report.jl")

function summarize_projected_runs()
    root=normpath(joinpath(@__DIR__,"../../results/gfpeps_validation"))
    base32="heisenberg_chi32_cut1.0e-8_gauge8.toml"
    continuation="heisenberg_chi32_cut1.0e-8_gauge8_continue_together_update.toml"
    latest(directory)=isfile(joinpath(root,directory,continuation)) ? continuation : base32
    entries=[
        ("原生 iPESS D4，seed12102",joinpath(root,"ipess_D4"),"heisenberg_chi32_cut1.0e-10.toml"),
        ("直接 iPEPS D4",joinpath(root,"ipeps_D4"),"heisenberg_chi64_cut1.0e-8_gauge8_continue_together_update.toml"),
        ("原生 iPESS D8，seed12003",joinpath(root,"ipess_D8"),base32),
        ("原生 iPESS D8，seed12103",joinpath(root,"ipess_D8_seed12103"),latest("ipess_D8_seed12103")),
        ("直接 iPEPS D8",joinpath(root,"ipeps_D8"),latest("ipeps_D8"))]
    destination=write_projected_energy_report(entries)
    open(destination,"a") do io
        println(io,"\n## D=8 的精度与磁化检查\n")
        println(io,"| 存档 | χ16→32 的能量变化 | χ32 后续迭代的能量变化 | 最后结果的最大观测量虚部 |")
        println(io,"|---|---:|---:|---:|")
        for (label,directory,selected) in entries[3:end]
            r16=TOML.parsefile(joinpath(directory,"heisenberg_chi16_cut1.0e-8_gauge8.toml"))
            r32=TOML.parsefile(joinpath(directory,base32))
            last=TOML.parsefile(joinpath(directory,selected))
            dchi=abs(r32["energy_per_site"]-r16["energy_per_site"])
            dmore=selected==base32 ? "尚无延续结果" : string(abs(last["energy_per_site"]-r32["energy_per_site"]))
            println(io,"| $label | $dchi | $dmore | $(last["max_imag"]) |")
        end
        println(io,"\n以上变化仅为实际环境对照，不是严格误差上界。未满足角谱阈值的数值仍标为估计，不能按输出小数位数声称高精度。")
        println(io,"原生 D8 seed12103 在源态及九格点精确投影检查中保持目标磁化；其 CTMRG 磁化偏差用于检验有限 χ 的影响。")
        println(io,"测得磁化未通过数值容差本身不能证明输入态混合扇区。seed12103 的残余磁化偏差及观测量虚部在增加 χ 时减小，但在固定 χ 延长迭代后仍存在，须保留有限环境误差的限制。")
        println(io,"另两份 D8 存档本身混合总 Sz 扇区，测得磁化接近 1/9 不能据此称为严格固定扇区。")
        println(io,"D4 的环境检查和全部局域磁化另见 [HEISENBERG_D4.md](HEISENBERG_D4.md)。")
    end
    println(destination)
    for (label,directory,file) in entries
        r=TOML.parsefile(joinpath(directory,file))
        println(label,": E/N=",r["energy_per_site"],", M/Msat=",r["mz_over_saturation"],", χ=",r["chi"],", converged=",r["converged"])
    end
    destination
end

if abspath(PROGRAM_FILE)==@__FILE__
    summarize_projected_runs()
end
