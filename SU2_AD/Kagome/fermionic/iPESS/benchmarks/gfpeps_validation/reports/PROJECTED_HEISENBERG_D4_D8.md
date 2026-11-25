# D=4、8 的 Gutzwiller 投影后海森堡能量

H=Σ⟨ij⟩ Sᵢ·Sⱼ，J=1，无外场，不含 -1/4 常数项。能量按物理 kagome 格点平均。
所有数值均为已保存 Gaussian 波函数作局域单占据投影后的期望值；没有对自旋 Hamiltonian 重新优化。
本表使用原版 orig parton 参数的存档，上、下自旋的三角形通量均为 -π/8；此通量描述试探态的构造，不是给海森堡 Hamiltonian 加入额外项。
这里列出源态实际投影后的结果，即使磁化不是目标 1/9 也保留并明确标注。

| 构造/存档 | D | χ | E/N | Mz/Msat | 角谱误差 | 角谱收敛 |
|---|---:|---:|---:|---:|---:|---|
| 原生 iPESS D4，seed12102 | 4 | 32 | -0.3600329115 | 0.1111111678 | 4.030936298895243e-8 | true |
| 直接 iPEPS D4 | 4 | 64 | -0.3859661451 | 0.1249219105 | 0.0013681350760264715 | false |
| 原生 iPESS D8，seed12003 | 8 | 32 | -0.4045555086 | 0.1114785727 | 8.968473987020893e-7 | false |
| 原生 iPESS D8，seed12103 | 8 | 32 | -0.4065065046 | 0.1111093213 | 3.2054073891892636e-7 | false |
| 直接 iPEPS D8 | 8 | 32 | -0.4080225198 | 0.1113850632 | 0.0002857695720335839 | false |

表中的打印小数不代表有效位数。角谱收敛与有限 χ 误差是两项不同检查；下面保留各次运行供比较。

## 原生 iPESS D4，seed12102

选用结果：`heisenberg_chi32_cut1.0e-10.toml`；source 元数据与 SHA256：`D:\My Documents\Code\Julia_codes\Tensor network\Symmetric_IPEPS\SU2_AD\Kagome\fermionic\iPESS\results\gfpeps_validation\ipess_D4/source.json`。

| 结果文件 | χ | E/N | Mz/Msat | 迭代数 | 角谱误差 | 收敛 |
|---|---:|---:|---:|---:|---:|---|
| heisenberg_chi16_cut1.0e-10.toml | 16 | -0.3600329534 | 0.1111129451 | 21 | 5.767735415826202e-8 | true |
| heisenberg_chi32_cut1.0e-10.toml | 32 | -0.3600329115 | 0.1111111678 | 12 | 4.030936298895243e-8 | true |
| heisenberg_chi32_cut1.0e-8.toml | 32 | -0.3600329384 | 0.1111111910 | 8 | 9.471093010519386e-8 | true |

## 直接 iPEPS D4

选用结果：`heisenberg_chi64_cut1.0e-8_gauge8_continue_together_update.toml`；source 元数据与 SHA256：`D:\My Documents\Code\Julia_codes\Tensor network\Symmetric_IPEPS\SU2_AD\Kagome\fermionic\iPESS\results\gfpeps_validation\ipeps_D4/source.json`。

| 结果文件 | χ | E/N | Mz/Msat | 迭代数 | 角谱误差 | 收敛 |
|---|---:|---:|---:|---:|---:|---|
| heisenberg_chi16_cut1.0e-10.toml | 16 | -0.3847405295 | 0.1257806953 | 100 | 0.007418932352987567 | false |
| heisenberg_chi32_cut1.0e-10.toml | 32 | -0.3880635246 | 0.1240283247 | 100 | 0.13967865070972052 | false |
| heisenberg_chi32_cut1.0e-6_regularized_together_update.toml | 32 | -0.3859106940 | 0.1249362682 | 100 | 2.5342979352032585e-5 | false |
| heisenberg_chi32_cut1.0e-8.toml | 32 | -0.3859185268 | 0.1249388126 | 80 | 0.0014970344144432425 | false |
| heisenberg_chi32_cut1.0e-8_continue_together_update.toml | 32 | -0.3859106319 | 0.1249364389 | 200 | 5.288737654385772e-5 | false |
| heisenberg_chi64_cut1.0e-8_gauge8.toml | 64 | -0.3859535717 | 0.1249318794 | 100 | 0.0026063492208760913 | false |
| heisenberg_chi64_cut1.0e-8_gauge8_continue_together_update.toml | 64 | -0.3859661451 | 0.1249219105 | 100 | 0.0013681350760264715 | false |

## 原生 iPESS D8，seed12003

选用结果：`heisenberg_chi32_cut1.0e-8_gauge8.toml`；source 元数据与 SHA256：`D:\My Documents\Code\Julia_codes\Tensor network\Symmetric_IPEPS\SU2_AD\Kagome\fermionic\iPESS\results\gfpeps_validation\ipess_D8/source.json`。

| 结果文件 | χ | E/N | Mz/Msat | 迭代数 | 角谱误差 | 收敛 |
|---|---:|---:|---:|---:|---:|---|
| heisenberg_chi16_cut1.0e-8_gauge8.toml | 16 | -0.4045200486 | 0.1116166342 | 58 | 9.894010975755915e-8 | true |
| heisenberg_chi32_cut1.0e-8_gauge8.toml | 32 | -0.4045555086 | 0.1114785727 | 40 | 8.968473987020893e-7 | false |

## 原生 iPESS D8，seed12103

选用结果：`heisenberg_chi32_cut1.0e-8_gauge8_continue_together_update.toml`；source 元数据与 SHA256：`D:\My Documents\Code\Julia_codes\Tensor network\Symmetric_IPEPS\SU2_AD\Kagome\fermionic\iPESS\results\gfpeps_validation\ipess_D8_seed12103/source.json`。

| 结果文件 | χ | E/N | Mz/Msat | 迭代数 | 角谱误差 | 收敛 |
|---|---:|---:|---:|---:|---:|---|
| heisenberg_chi16_cut1.0e-8_gauge8.toml | 16 | -0.4064397531 | 0.1111404681 | 60 | 5.043801914185953e-7 | false |
| heisenberg_chi32_cut1.0e-8_gauge8.toml | 32 | -0.4065066011 | 0.1111092791 | 40 | 1.017975987248696e-5 | false |
| heisenberg_chi32_cut1.0e-8_gauge8_continue_together_update.toml | 32 | -0.4065065046 | 0.1111093213 | 20 | 3.2054073891892636e-7 | false |

## 直接 iPEPS D8

选用结果：`heisenberg_chi32_cut1.0e-8_gauge8_continue_together_update.toml`；source 元数据与 SHA256：`D:\My Documents\Code\Julia_codes\Tensor network\Symmetric_IPEPS\SU2_AD\Kagome\fermionic\iPESS\results\gfpeps_validation\ipeps_D8/source.json`。

| 结果文件 | χ | E/N | Mz/Msat | 迭代数 | 角谱误差 | 收敛 |
|---|---:|---:|---:|---:|---:|---|
| heisenberg_chi16_cut1.0e-8_gauge8.toml | 16 | -0.4072294728 | 0.1119817472 | 60 | 5.198786634760299e-6 | false |
| heisenberg_chi32_cut1.0e-8_gauge8.toml | 32 | -0.4080263318 | 0.1113812842 | 40 | 0.0009169106068298914 | false |
| heisenberg_chi32_cut1.0e-8_gauge8_continue_together_update.toml | 32 | -0.4080225198 | 0.1113850632 | 20 | 0.0002857695720335839 | false |

磁化约定：Msat=1/2，目标 Mz/Msat=1/9 对应每格点 Mz=1/18。
源 Gaussian 存档的总 Sz 守恒检查见 [SPIN_SECTOR_AUDIT.md](SPIN_SECTOR_AUDIT.md)。
不同种子及不同 D 的投影后能量需要实测，投影前自由费米子能量较低不保证投影后交换能较低。

## D=8 的精度与磁化检查

| 存档 | χ16→32 的能量变化 | χ32 后续迭代的能量变化 | 最后结果的最大观测量虚部 |
|---|---:|---:|---:|
| 原生 iPESS D8，seed12003 | 3.546001003101873e-5 | 尚无延续结果 | 1.6076103232447283e-13 |
| 原生 iPESS D8，seed12103 | 6.684794152161944e-5 | 9.645169629646233e-8 | 5.0452621280446645e-6 |
| 直接 iPEPS D8 | 0.0007968590241504425 | 3.8119823539961573e-6 | 1.662123877447759e-16 |

以上变化仅为实际环境对照，不是严格误差上界。未满足角谱阈值的数值仍标为估计，不能按输出小数位数声称高精度。
原生 D8 seed12103 在源态及九格点精确投影检查中保持目标磁化；其 CTMRG 磁化偏差用于检验有限 χ 的影响。
测得磁化未通过数值容差本身不能证明输入态混合扇区。seed12103 的残余磁化偏差及观测量虚部在增加 χ 时减小，但在固定 χ 延长迭代后仍存在，须保留有限环境误差的限制。
另两份 D8 存档本身混合总 Sz 扇区，测得磁化接近 1/9 不能据此称为严格固定扇区。
D4 的环境检查和全部局域磁化另见 [HEISENBERG_D4.md](HEISENBERG_D4.md)。
