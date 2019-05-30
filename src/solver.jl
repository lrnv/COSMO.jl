const LinsolveSubarray = SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int64}},true}

function admm_z!(x::Vector{Float64},
	s::SplitVector{Float64},
	μ::Vector{Float64},
	v::Vector{Float64},
	ρ::Vector{Float64},
	set::CompositeConvexSet{Float64},
	v1::SubArray,
	v2::SubArray)

	@. x = v1
	@. s = v2
	p_time = @elapsed project!(s, set)

	# recover original dual variable for conic constraints
	@. μ = ρ * (v2 - s)

	return p_time
end

function admm_x!(x::Vector{Float64},
	s::SplitVector{Float64},
	v::Vector{Float64},
	ν::LinsolveSubarray,
	x_tl::LinsolveSubarray,
	s_tl::Vector{Float64},
	ls::Vector{Float64},
	sol::Vector{Float64},
	F,
	q::Vector{Float64},
	b::Vector{Float64},
	ρ::Vector{Float64},
	σ::Float64,
	m::Int64,
	n::Int64,
	v1::SubArray,
	v2::SubArray)

	# linear solve
	# Create right hand side for linear system
	# deconstructed solution vector is ls = [x_tl(n+1); ν(n+1)]
	# x_tl and ν are automatically updated, since they are views on sol
	# ρ  = 0.1

	@. ls[1:n] = σ * (2 * x - v1) - q # or can it be ρ?????
	@. ls[(n + 1):end] = b + v2 - 2*s
	sol .= F \ ls

	@. s_tl = - ν / ρ + 2 * s - v2

	end

function admm_v!(x::Vector{Float64}, s::SplitVector{Float64}, x_tl::LinsolveSubarray, s_tl::Vector{Float64}, v::Vector{Float64}, α::Float64, m::Int64, n::Int64)
	# update dual variable v
	@. v[1:n] = v[1:n] + 2 * α .* (x_tl - x)
	@. v[n+1:n+m] = v[n+1:n+m] + 2 * α .* (s_tl - s)
end

# SOLVER ROUTINE
# -------------------------------------


"""
optimize!(model)

Attempts to solve the optimization problem defined in `COSMO.Model` object with the user settings defined in `COSMO.Settings`. Returns a `COSMO.Result` object.
"""
function optimize!(ws::COSMO.Workspace)
	solver_time_start = time()
	settings = ws.settings

	# create scaling variables
	# with scaling    -> uses mutable diagonal scaling matrices
	# without scaling -> uses identity matrices
	ws.sm = (settings.scaling > 0) ? ScaleMatrices(ws.p.m, ws.p.n) : ScaleMatrices()

	# perform preprocessing steps (scaling, initial KKT factorization)
	ws.times.setup_time = @elapsed setup!(ws);
	ws.times.proj_time  = 0. #reset projection time

	# instantiate variables
	num_iter = 0
	status = :Unsolved
	cost = Inf


	# print information about settings to the screen
	settings.verbose && print_header(ws)
	time_limit_start = time()

 	iter_history = IterateHistory(ws.p.m ,ws.p.n,settings.acc_mem)

 	update_iterate_history!(iter_history, ws.vars.x, ws.vars.s, -ws.vars.μ, ws.vars.v, ws.r_prim, ws.r_dual, zeros(ws.settings.acc_mem), NaN)

	#preallocate arrays
	m = ws.p.m
	n = ws.p.n
	δx = zeros(n)
	δy =  zeros(m)
	s_tl = zeros(m) # i.e. sTilde

	ls = zeros(n + m)
	sol = zeros(n + m)
	x_tl = view(sol, 1:n) # i.e. xTilde
	ν = view(sol, (n + 1):(n + m))
	v1 = view(ws.vars.v, 1:n)
	v2 = view(ws.vars.v, n+1:n+m)
	settings.verbose_timing && (iter_start = time())

	for iter = 1:settings.max_iter

		num_iter += 1

		if num_iter > 1
			COSMO.update_history!(ws.accelerator, ws.vars.v, ws.vars.v_prev)
			COSMO.accelerate!(ws.vars.v, ws.vars.v_prev, ws.accelerator, num_iter)
		end

		@. ws.vars.v_prev = ws.vars.v
		@. δx = ws.vars.x
		@. δy = ws.vars.μ


		# perform ADMM update steps
		ws.times.proj_time += admm_z!(ws.vars.x, ws.vars.s, ws.vars.μ, ws.vars.v, ws.ρvec, ws.p.C, v1, v2)


		# compute deltas for infeasibility detection
		@. δx = ws.vars.x - δx
		@. δy = -ws.vars.μ + δy

		if mod(iter, ws.settings.check_termination )  == 0
			calculate_residuals!(ws)
		end


		# check convergence with residuals every {settings.checkIteration} steps
		if mod(iter, settings.check_termination) == 0
			# update cost
			cost = ws.sm.cinv[] * (1/2 * ws.vars.x' * ws.p.P * ws.vars.x + ws.p.q' * ws.vars.x)[1]

			if abs(cost) > 1e20
				status = :Unsolved
				break
			end

			# print iteration steps
			settings.verbose && print_iteration(settings, iter, cost, ws.r_prim, ws.r_dual, ws.ρ)

			if has_converged(ws)
				status = :Solved
				break
			end
		end

		# check infeasibility conditions every {settings.checkInfeasibility} steps
		if mod(iter, settings.check_infeasibility) == 0
			if is_primal_infeasible(δy, ws)
				status = :Primal_infeasible
				cost = Inf
				break
			end

			if is_dual_infeasible(δx, ws)
				status = :Dual_infeasible
				cost = -Inf
				break
			end
		end


		# adapt rhoVec if enabled
		if ws.settings.adaptive_rho && (mod(iter, ws.settings.adaptive_rho_interval + 1) == 0) && (ws.settings.adaptive_rho_interval + 1 > 0)
			adapt_rho_vec!(ws, iter)
		end


		eta = zeros(ws.settings.acc_mem)
		cond = 0.
		if ws.settings.accelerator == :empty
			eta = zeros(ws.settings.acc_mem)
		else
			eta = ws.accelerator.eta
			ne = length(eta)
			if ne < ws.settings.acc_mem
				eta = [eta; zeros(ws.settings.acc_mem-ne)]
			end
			cond = ws.accelerator.cond
		end
		 update_iterate_history!(iter_history, ws.vars.x, ws.vars.s, -ws.vars.μ, ws.vars.v, ws.r_prim, ws.r_dual, eta, cond)



		admm_x!(ws.vars.x, ws.vars.s, ws.vars.v, ν, x_tl, s_tl, ls, sol, ws.F, ws.p.q, ws.p.b, ws.ρvec, settings.sigma, m, n, v1, v2)
		admm_v!(ws.vars.x, ws.vars.s, x_tl, s_tl, ws.vars.v, settings.alpha, m, n)

	if settings.time_limit !=0 &&  (time() - time_limit_start) > settings.time_limit
			status = :Time_limit_reached
			break
		end






	end #END-ADMM-MAIN-LOOP

	settings.verbose_timing && (ws.times.iter_time = (time() - iter_start))
	settings.verbose_timing && (ws.times.post_time = time())

	# calculate primal and dual residuals
	if num_iter == settings.max_iter
		calculate_residuals!(ws)
		status = :Max_iter_reached
	end

	# reverse scaling for scaled feasible cases
	if settings.scaling != 0
		reverse_scaling!(ws)
		# FIXME: Another cost calculation is not necessary since cost value is not affected by scaling
		cost =  (1/2 * ws.vars.x' * ws.p.P * ws.vars.x + ws.p.q' * ws.vars.x)[1] #sm.cinv * not necessary anymore since reverseScaling
	end

	ws.times.solver_time = time() - solver_time_start
	settings.verbose_timing && (ws.times.post_time = time() - ws.times.post_time)
	# print solution to screen
	settings.verbose && print_result(status, num_iter, cost, ws.times.solver_time)

	# create result object
	res_info = ResultInfo(ws.r_prim, ws.r_dual)
	y = -ws.vars.μ

	if typeof(ws.accelerator) <: AndersonAccelerator{Float64}
		iter_history.aa_fail_data = ws.accelerator.fail_counter
	end

	return Result{Float64}(ws.vars.x, y, ws.vars.s.data, cost, num_iter, status, res_info, ws.times), ws, iter_history

end

