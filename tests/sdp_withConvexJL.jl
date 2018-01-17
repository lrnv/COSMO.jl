# Test script to have an easy example solved by SCS that can be used to compare OSSDP solutions against
workspace()
include("../Projections.jl")
include("../Solver.jl")

using Convex, SCS
using Base.Test
using OSSDP
# using PyPlot


# # Problem DATA
A1 = [1.0 0 1; 0 3 7; 1 7 5]
A2 = [0.0 2 8; 2 6 0; 8 0 4]
C = [1.0 2 3; 2 9 0; 3 0 7]
b1 = 11.0
b2 = 19.0

############## Solution with ConvexJL = SCS
Y = Variable(3,3)
problem = minimize(trace(C*Y))

constraint1 = trace(A1*Y) ==b1
constraint2 = trace(A2*Y) ==b2
constraint3 = isposdef(Y)

problem.constraints += [constraint1,constraint2,constraint3]

solve!(problem,SCSSolver(verbose=false))

# print result
# problem.status
println("\nSCS Result:")
println("X: $(Y.value)")
println("Cost: $(problem.optval)\n")
# @show(constraints[1].dual)

# A = [vec(A1)';vec(A2)']
# b = [b1;b2]
# @test A*vec(x) == b


############## Solution with OSSDP
q = vec(C)
A = [vec(A1)';vec(A2)']
b = [b1;b2]
P = zeros(9,9)

# define example problem
settings = sdpSettings(rho=1.0,sigma=1.0,alpha=1.6,max_iter=500,verbose=true)

# solve SDP problem
res,dbg = solveSDP(P,q,A,b,settings)
X = reshape(res.x,3,3)

# compute smallest eigenvalue of result X
F = eigfact(X)
XλMin = minimum(F[:values])
F = eigfact(reshape(res.s,3,3))
SλMin = minimum(F[:values])
epsEV = -1e-9
@testset "Compare results" begin
  @test norm(X-Y.value,Inf) < 1e-3
  @test norm(res.cost-problem.optval) < 1e-3
  @test norm(res.x-res.s,Inf)<1e-3
  @test XλMin > epsEV
  @test SλMin > epsEV

end

