# 只修改本文件的配置区，然后在 Julia / 编辑器中运行整个文件。
# 无需传命令行参数。每次运行请使用新的输出目录。
using Dates

# :charged_trial  小规模带磁化的 SU 试算；默认从 SU2 投影初始化 U1。
# :simple_update 一般 SU 演化，支持 sqrt3xsqrt3 和 3x3。
# :convert       转换已有态，默认输出 U1，并核对九自旋有限环面的观测量。
task = :charged_trial
output_root = joinpath(@__DIR__, "results")
run_name = Dates.format(now(), "yyyymmdd_HHMMSS_sss")

# ---------- 小规模带磁化试算 ----------
charged_options = (
    symmetry = :u1,         # :u1 或 :su2
    Dmax = 3,
    seed = 1234,
    h = 0.38,              # H = H_exchange + h*sum(Sz)
    threads = 10,
    initial_state = "",    # 空字符串：新建 SU2 小 D 态；U1 模式随后做投影。
                           # 也可填已有 SU2 / U1 checkpoint 的完整路径。
    schedule = [(dt=0.01, tau=0.02), (dt=0.003, tau=0.006), (dt=0.001, tau=0.002)],
    out = joinpath(output_root, "charged_" * run_name),
)
# 此试算采用 sqrt3xsqrt3、辅助 j=1/2 的构造，默认总共六个完整步；
# 最后计算一次精确九自旋有限环面观测量，不启动 CTMRG。

# ---------- 一般 simple update ----------
su_options = (
    symmetry = :u1,             # :u1 或 :su2
    cell = "sqrt3xsqrt3",       # "sqrt3xsqrt3"、"3x3" 或 "1x1"
    Dmax = 3,
    seed = 1234,
    threads = 10,
    J1 = 1.0,
    virtual_space = "0:1,1/2:1", # 初始 SU2 表示：spin:multiplicity
    initial_state = "",        # U1 可读取已有 SU2 / U1 checkpoint。
    component = 1,             # SU2 辅助腿投影分量，1 为最高权。
    resume = "",               # 续算 checkpoint；与 initial_state 二选一。
    schedule = [(dt=0.03, tau=40.0), (dt=0.01, tau=20.0),
                (dt=0.003, tau=5.0), (dt=0.001, tau=2.0), (dt=0.0005, tau=1.0)],
    out = joinpath(output_root, "su_" * run_name),
)
# 此分区默认是正式的 9000 步演化，只有 task=:simple_update 才会执行。
# U1 的 initial_state 若留空，得到的是普通 SU2 singlet 的 U1 限制，磁化为零。
# 要研究 1/9 态，请先用 charged_trial，随后将其 final.jld2 路径填在这里。
# 同一 sqrt3 初态也可用于 cell="3x3"，此时复制成九个独立 PESS。
# 磁化扇区内 Zeeman 项是常数，SU 使用交换门；h 在观测量中计入一次。

# ---------- 已有态转换及独立验证 ----------
conversion_options = (
    input = "",               # 填 SU2 charged 或 U1 checkpoint 的完整路径。
    representation = :u1,     # 默认 :u1；需要普通数组时改为 :dense。
    component = 1,
    h = 0.38,
    threads = 10,
    out = joinpath(output_root, "converted_" * run_name),
)
# 此验证入口限于单个带辅助电荷的 sqrt3 cell、实际 D<=6。
# 一般 SU 保存的 U1 checkpoint / portable TOML 没有这个有限环面的 D 限制。

# ---------- 配置区结束，下面不需要修改 ----------
if !isdefined(@__MODULE__, :run_kagome_settings)
    include(joinpath(@__DIR__, "kagome_settings_driver.jl"))
end
run_kagome_settings(task; su=su_options, charged=charged_options, conversion=conversion_options)
