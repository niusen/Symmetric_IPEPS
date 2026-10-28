# Load only user settings before selecting/installing the Julia environment.
const KAGOME_SETTINGS_ONLY=true
include(joinpath(@__DIR__,"run_su2_D4_D12_server.jl"))
import Pkg
project=su2_server_settings.device=="cpu" ? @__DIR__ : joinpath(@__DIR__,"gpu")
Pkg.activate(project)
Pkg.instantiate()
include(joinpath(@__DIR__,"su2_server_driver.jl"))
run_su2_server(su2_server_settings)
