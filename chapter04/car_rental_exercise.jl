using Distributions
using PyPlot
using Seaborn

# maximum # of cars in each location
const MAX_CARS = 20

# maximum # of cars to move during night
const MAX_MOVE_OF_CARS = 5

# expectation for rental requests in first location
const RENTAL_REQUEST_FIRST_LOC = 3

# expectation for rental requests in second location
const RENTAL_REQUEST_SECOND_LOC = 4

# expectation for # of cars returned in first location
const RETURNS_FIRST_LOC = 3

# expectation for # of cars returned in second location
const RETURNS_SECOND_LOC = 2

const DISCOUNT = 0.9

# credit earned by a car
const RENTAL_CREDIT = 10

# cost of moving a car
const MOVE_CAR_COST = 2

const actions = -MAX_MOVE_OF_CARS:MAX_MOVE_OF_CARS

const TOLERANCE = 0.5

rand_cache = IdDict()

function poisson_probablility(n, lam)::Float64
    global rand_cache
    key = 10 * n + lam
    if !haskey(rand_cache, key)
        poisson::Poisson = Poisson(lam)
        rand_cache[key] = pdf(poisson, n)
    end
    rand_cache[key]
end

function expected_return(state, action, state_value, use_constant_return)::Float64
    expected_return::Float64 = 0.0
    expected_return -= MOVE_CAR_COST * abs(action)

    NUM_FIRST::Int = min(state[1] - action, MAX_CARS)
    NUM_SECOND::Int = min(state[2] + action, MAX_CARS)

    for request_first_loc in 0:10
        for request_second_loc in 0:10
            prob_request::Float64 = poisson_probablility(request_first_loc, RENTAL_REQUEST_FIRST_LOC) *
                            poisson_probablility(request_second_loc, RENTAL_REQUEST_SECOND_LOC)
            rent_first::Int = min(NUM_FIRST, request_first_loc)
            rent_second::Int = min(NUM_SECOND, request_second_loc)
            reward::Int = (rent_first + rent_second) * RENTAL_CREDIT
            if use_constant_return
                num_first::Int = min(NUM_FIRST - rent_first + RETURNS_FIRST_LOC, MAX_CARS)
                num_second::Int = min(NUM_SECOND - rent_second + RETURNS_SECOND_LOC, MAX_CARS)
                expected_return += prob_request * (reward + DISCOUNT * state_value[num_first + 1, num_second + 1])
            else
                for return_first_loc in 0:10
                    for return_second_loc in 0:10
                        prob_return::Float64 = prob_request * poisson_probablility(return_first_loc, RETURNS_FIRST_LOC) *
                            poisson_probablility(return_second_loc, RETURNS_SECOND_LOC)
                        num_first = min(NUM_FIRST - rent_first + return_first_loc, MAX_CARS)
                        num_second = min(NUM_SECOND - rent_second + return_second_loc, MAX_CARS)
                        expected_return += prob_return * (reward + DISCOUNT * state_value[num_first + 1, num_second + 1])
                    end
                end
            end 
        end
    end

    expected_return
end

function expected_return_more_dynamics(state, action, state_value, use_constant_return)::Float64
    expected_return::Float64 = 0.0
    if action > 0
        expected_return -= MOVE_CAR_COST * (action - 1)
    else
        expected_return -= MOVE_CAR_COST * (-action)
    end

    NUM_FIRST::Int = min(state[1] - action, MAX_CARS)
    NUM_SECOND::Int = min(state[2] + action, MAX_CARS)

    for request_first_loc in 0:10
        for request_second_loc in 0:10
            prob_request::Float64 = poisson_probablility(request_first_loc, RENTAL_REQUEST_FIRST_LOC) *
                            poisson_probablility(request_second_loc, RENTAL_REQUEST_SECOND_LOC)
            rent_first::Int = min(NUM_FIRST, request_first_loc)
            rent_second::Int = min(NUM_SECOND, request_second_loc)
            reward::Int = (rent_first + rent_second) * RENTAL_CREDIT
            if use_constant_return
                num_first::Int = min(NUM_FIRST - rent_first + RETURNS_FIRST_LOC, MAX_CARS)
                num_second::Int = min(NUM_SECOND - rent_second + RETURNS_SECOND_LOC, MAX_CARS)
                if num_first > 10
                    reward -= 4
                end
                if num_second > 10
                    reward -= 4
                end
                expected_return += prob_request * (reward + DISCOUNT * state_value[num_first + 1, num_second + 1])
            else
                for return_first_loc in 0:10
                    for return_second_loc in 0:10
                        prob_return::Float64 = prob_request * poisson_probablility(return_first_loc, RETURNS_FIRST_LOC) *
                            poisson_probablility(return_second_loc, RETURNS_SECOND_LOC)
                        num_first = min(NUM_FIRST - rent_first + return_first_loc, MAX_CARS)
                        num_second = min(NUM_SECOND - rent_second + return_second_loc, MAX_CARS)
                        if num_first > 10
                            reward -= 4
                        end
                        if num_second > 10
                            reward -= 4
                        end
                        expected_return += prob_return * (reward + DISCOUNT * state_value[num_first + 1, num_second + 1])
                    end
                end
            end 
        end
    end

    expected_return
end

function rental_car(constant_return=true, use_more_dynamics=false)
    values::Array{Float64} = zeros((MAX_CARS + 1, MAX_CARS + 1))
    policy::Array{Int} = zeros(size(values))

    iteration::Int = 0
    master_fig, axes = PyPlot.subplots(3, 3, figsize=(40, 30))
    subplots_adjust(wspace=0.1, hspace=0.2)

    while true
        row::Int = div(iteration, 3) + 1
        col::Int = rem(iteration, 3) + 1
        fig = heatmap(policy, cmap="YlGnBu", ax=axes[row, col])
        axes[row, col].invert_yaxis()
        fig.set_ylabel("# cars at first location", fontsize=30)
        fig.set_xlabel("# cars at second location", fontsize=30)
        fig.set_title("policy $iteration", fontsize=30)

        # policy evaluation
        while true
            old_values::Array{Float64} = copy(values)
            for i in 0:MAX_CARS
                for j in 0:MAX_CARS
                    if use_more_dynamics
                        new_state_value::Float64 = expected_return_more_dynamics([i, j], policy[i + 1, j + 1], values, constant_return)
                    else
                        new_state_value = expected_return([i, j], policy[i + 1, j + 1], values, constant_return)
                    end
                    values[i + 1, j + 1] = new_state_value
                end
            end

            max_value_change::Float64 = maximum(map((x) -> abs(x), (old_values - values)))
            println("Max value change: $max_value_change")
            if max_value_change < 1e-4
                break
            end
        end

        # policy improvement
        policy_stable::Bool = true
        for i in 0:MAX_CARS
            for j in 0:MAX_CARS
                old_action::Int = policy[i + 1, j + 1]
                action_returns::Array{Float64} = []
                for action in actions
                    if (0 <= action <= i) || (-j <= action <= 0)
                        if use_more_dynamics
                            action_return::Float64 = expected_return_more_dynamics([i + 1, j + 1], action, values, constant_return)
                        else
                            action_return = expected_return([i + 1, j + 1], action, values, constant_return)
                        end
                        append!(action_returns, action_return)
                    else
                        append!(action_returns, -Inf)
                    end
                end
                new_actions::Array{Int} = findall(-TOLERANCE .< action_returns .- maximum(action_returns) .< TOLERANCE)
                map!((x) -> actions[x], new_actions, new_actions)
                policy[i + 1, j + 1] = new_actions[1]
                if policy_stable && !in(old_action, new_actions)
                    policy_stable = false
                end
            end
        end
        println("Policy stable $policy_stable")

        iteration += 1

        if iteration == 9
            break
        end
        if policy_stable
            row = div(iteration, 3) + 1
            col = rem(iteration, 3) + 1
            fig = heatmap(values, cmap="YlGnBu", ax=axes[row, col])
            axes[row, col].invert_yaxis()
            fig.set_ylabel("# cars at first location", fontsize=30)
            fig.set_xlabel("# cars at second location", fontsize=30)
            fig.set_title("optimal value", fontsize=30)
            break
        end
    end

    savefig("test.png")
    PyPlot.close(master_fig)
end

function main()
    rental_car(false, true)
end

main()
