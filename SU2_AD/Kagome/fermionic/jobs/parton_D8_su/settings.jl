# Common production parameters for all six unprojected seeds. Each tuple is (tau,dt).
const UNIT_CELL = (3,3)       # kagome primitive cells; three physical sites per cell
const SU_SCHEDULE = [(1.0,0.05), (0.4,0.02), (0.1,0.005)]
const MEASUREMENT = (chi=32, maxiter=200, tol=1e-7)
const SU_TRUNCATION_TOL = 1e-10
const BLAS_THREADS = parse(Int,get(ENV,"KAGOME_BLAS_THREADS","4"))

# Frozen seed identity: family, Hamiltonian, directory, tensor name, source NPZ SHA256, MAT SHA256.
const CASES = [
 (:ipeps,:orig,"ipeps_D8","orig_m3_r0","563ce0115b30fcb7d4873591fdeae935787ec23f642cb091846238b7ffd9c9ef","614ed16402a64b9ce37e0b662c5613de0e08fdda62e1122e94342c3adddd23d0"),
 (:ipeps,:mild,"ipeps_D8_mild","mild_m3_r0","16d08f1c81e2de42553fdd5e3cb89c6b8c1a6e5821e29ce4c3c4373c0592680b","0eb7b39e8adc854e6f638d7b36c86362bd245416eac37eb02291b28388ef0a00"),
 (:ipeps,:zero,"ipeps_D8_zero","zero_m3_r0","f17f20abf4e2c701645de418768f5c55fa884f9fa9886dcf55dd32ce75090f3f","b7274298dd4ea670613183a9f092135a6507d2bad5f42ae48672cfc060f85eea"),
 (:ipess,:orig,"ipess_D8","orig_m3_s12003","f792f3f71f370921b3faeab2a8c1a667b79e7112b13413bb8b099d91fac4e064","b2124e5e7011001ebba9387043d34e016a53bc235564abfda14e80af75f0c22d"),
 (:ipess,:mild,"ipess_D8_mild","mild_m3_s12103","a0b23acbcacf0f11eb4221ba76c7b9d19a2ecbd98a1e82f2109c6bfb39d44a88","616fed7756def6bbcaa50c2c402da2edd7e1abadaf786550d505ed1ea9491731"),
 (:ipess,:zero,"ipess_D8_zero","zero_m3_s12003","d41dcdd190d7cbf9cc9450f192a13318f5e94c55c823e6df71c7cf050c14feae","8e9aad06134dbecd25e8fe2d615de1f4b882ed2345eb54eb9a43b597fcc7b6df"),
]
