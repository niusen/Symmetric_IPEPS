using JLD2
if !isdefined(@__MODULE__,:write_spin_environment_tables)
    include(joinpath(@__DIR__,"../../../../../src/fermionic/kagome/gfpeps/reporting/spin_tables.jl"))
end

function write_heisenberg_report(selected)
    root=normpath(joinpath(@__DIR__,"../../results/gfpeps_validation"))
    destination=joinpath(root,"reports","HEISENBERG_D4.md")
    mkpath(dirname(destination))
    entries=[(family,joinpath(root,"$(family)_D4"),selected[family]) for family in ("ipess","ipeps")]
    open(destination,"w") do io
        println(io,"# D=4 Gutzwiller 投影态：海森堡能量和磁化\n")
        println(io,"**更正：直接 iPEPS 存档混合总 Sz 扇区，其能量不能当作目标 1/9 磁化态的结果。原生 D=4 iPESS 通过数值扇区检查。**")
        println(io,"独立于 CTMRG 的输入与精确投影检查见 [SPIN_SECTOR_AUDIT.md](../../../benchmarks/gfpeps_validation/reports/SPIN_SECTOR_AUDIT.md)。\n")
        println(io,"模型为 H=Σ⟨ij⟩ Sᵢ·Sⱼ，J=1，S=1/2，无外场，不含 -1/4 常数项。")
        println(io,"测量此前保存的 Gaussian 态经 Gutzwiller 投影后的固定波函数，没有针对海森堡模型重新优化。")
        println(io,"能量和磁化按物理 kagome 格点平均；每个基本胞有 3 个格点、6 条键。M_sat=1/2。\n")
        println(io,"下表保留原始输出小数便于复现；实际精度须结合收敛列及后面的环境对照，不能把打印位数当成有效位数。\n")
        write_spin_environment_tables(io,entries,dirname(destination))
        println(io,"\n## 九个格点的局域磁化\n")
        println(io,"当前 x=1,2,3 对应源晶胞 x=0,2,1；当前 (a,b,c) 对应源 (B,A,C)。以下是〈Sᵢ〉，不是自旋长度 √(S(S+1))。\n")
        write_local_spin_tables(io,entries,path->JLD2.load(path,"result"))
        println(io,"## 实现和复现\n")
        println(io,"`src/fermionic/kagome/gfpeps/observables/heisenberg.jl` 复用共享 fermionic 双层与 ob_2x2 收缩；自旋为偶宇称，跨胞相关不插入 hopping 的奇宇称符号串。")
        println(io,"非均匀 3×2 自旋乘积态逐键测试 40 项、胞内 singlet 测试 5 项通过，覆盖三个方向的自旋、上/下三角形键与每格点归一化。\n")
        println(io,"包含 `run_heisenberg.jl` 后使用 `run_spin_energy(family,4,chi;cutoff,maxiter,tol,warmstart,gauge_sweeps)` 可复现；具体参数与初始环境文件记录在各 TOML 中。")
        println(io,"局域磁化、逐键相关和 CTM 环境保存在对应 JLD2 的 result 中。此处没有计算场强依赖，不能仅凭一个磁化值证明存在磁化平台。")
    end
    println("Wrote ",destination)
end
