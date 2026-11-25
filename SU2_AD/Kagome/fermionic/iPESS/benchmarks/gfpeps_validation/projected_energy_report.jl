using TOML
if !isdefined(@__MODULE__,:write_projected_energy_tables)
    include(joinpath(@__DIR__,"../../../../../src/fermionic/kagome/gfpeps/reporting/spin_tables.jl"))
end

"""Write a comparison of explicitly selected projected-state runs.

Each entry is (label, result_directory, result_filename). Selection is explicit:
an unconverged low energy is not automatically ranked as a better wavefunction.
"""
function write_projected_energy_report(entries; destination=normpath(joinpath(@__DIR__,"../../results/gfpeps_validation/reports/PROJECTED_HEISENBERG_D4_D8.md")))
    audit=joinpath(@__DIR__,"reports/SPIN_SECTOR_AUDIT.md")
    audit_link=replace(relpath(normpath(audit),dirname(abspath(destination))),'\\'=>'/')
    mkpath(dirname(abspath(destination)))
    open(destination,"w") do io
        println(io,"# D=4、8 的 Gutzwiller 投影后海森堡能量\n")
        println(io,"H=Σ⟨ij⟩ Sᵢ·Sⱼ，J=1，无外场，不含 -1/4 常数项。能量按物理 kagome 格点平均。")
        println(io,"所有数值均为已保存 Gaussian 波函数作局域单占据投影后的期望值；没有对自旋 Hamiltonian 重新优化。")
        println(io,"本表使用原版 orig parton 参数的存档，上、下自旋的三角形通量均为 -π/8；此通量描述试探态的构造，不是给海森堡 Hamiltonian 加入额外项。")
        println(io,"这里列出源态实际投影后的结果，即使磁化不是目标 1/9 也保留并明确标注。\n")
        write_projected_energy_tables(io,entries)
        println(io,"磁化约定：Msat=1/2，目标 Mz/Msat=1/9 对应每格点 Mz=1/18。")
        println(io,"源 Gaussian 存档的总 Sz 守恒检查见 [SPIN_SECTOR_AUDIT.md]($audit_link)。")
        println(io,"不同种子及不同 D 的投影后能量需要实测，投影前自由费米子能量较低不保证投影后交换能较低。")
    end
    destination
end
