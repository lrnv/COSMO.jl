using COSMO, Random, Test, Pkg
rng = Random.MersenneTwister(12345)




include("./UnitTests/COSMOTestUtils.jl")
# Define the types to run the unit tests with
UnitTestFloats = [Float32; Float64; BigFloat]

@testset "COSMO Native Testset" begin

  include("./UnitTests/sets.jl")
  include("./UnitTests/nuclear_norm_minimization.jl")
  include("./UnitTests/psd_completion.jl")
  include("./UnitTests/psd_completion_and_merging.jl")
  include("./UnitTests/clique_merging_example.jl")
  include("./UnitTests/reduced_clique_graph.jl")
  include("./UnitTests/chordal_decomposition_triangle.jl")
  include("./UnitTests/InfeasibilityTests/runTests.jl")

  # floating-point precision type agnostic unit tests
  # N.B.: Some infeasibility tests rely on sufficient precision (at least Float64 in some cases)
  # SDPs are currently only supported with Float32
  include("./UnitTests/simple.jl")
  include("./UnitTests/constraints.jl")
  include("./UnitTests/model.jl")
  include("./UnitTests/socp-lasso.jl")
  include("./UnitTests/closestcorr.jl")
  include("./UnitTests/exp_cone.jl")
  include("./UnitTests/pow_cone.jl")
  include("./UnitTests/qp-box.jl")
  include("./UnitTests/algebra.jl")
  include("./UnitTests/splitvector.jl")
  include("./UnitTests/interface.jl")
  include("./UnitTests/kktsolver.jl")
  include("./UnitTests/print.jl")

    # optional unittests
  if pkg_installed("Pardiso", "46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2")
    include("./UnitTests/options_factory.jl")
  end

end
nothing
