using Distributions
using PyPlot
using Seaborn

# maximum # of cars in each location
MAX_CARS = 20

# maximum # of cars to move during night
MAX_MOVE_OF_CARS = 5

# expectation for rental requests in first location
RENTAL_REQUEST_FIRST_LOC = 3

# expectation for rental requests in second location
RENTAL_REQUEST_SECOND_LOC = 4

# expectation for # of cars returned in first location
RETURNS_FIRST_LOC = 3

# expectation for # of cars returned in second location
RETURNS_SECOND_LOC = 2

DISCOUNT = 0.9

# credit earned by a car
RENTAL_CREDIT = 10

# cost of moving a car
MOVE_CAR_COST = 2

actions = -MAX_MOVE_OF_CARS:MAX_MOVE_OF_CARS

TOLERANCE = 0.05

rand_cache = IdDict()

function poisson_probablility(n, lam)
    global rand_cache
    key = 10 * n + lam
    if !haskey(rand_cache, key)
        poisson = Poisson(lam)
        rand_cache[key] = pdf(poisson, n)
    end
    rand_cache[key]
end

function expected_return(state, action, state_value, use_constant_return)::Float64
    expected_return = 0.0
    expected_return -= MOVE_CAR_COST * abs(action)

    NUM_FIRST = min(state[1] - action, MAX_CARS)
    NUM_SECOND = min(state[2] + action, MAX_CARS)

    for request_first_loc in 0:10
        for request_second_loc in 0:10
            prob = poisson_probablility(request_first_loc, RENTAL_REQUEST_FIRST_LOC) *
                            poisson_probablility(request_second_loc, RENTAL_REQUEST_SECOND_LOC)
            rent_first = min(NUM_FIRST, request_first_loc)
            rent_second = min(NUM_SECOND, request_second_loc)
            if use_constant_return
                reward = (rent_first + rent_second) * RENTAL_CREDIT
                num_first::Int = min(NUM_FIRST - rent_first + RETURNS_FIRST_LOC, MAX_CARS)
                num_second::Int = min(NUM_SECOND - rent_second + RETURNS_SECOND_LOC, MAX_CARS)
                expected_return += prob * (reward + DISCOUNT * state_value[num_first + 1, num_second + 1])
            else
                for return_first_loc in 0:10
                    for return_second_loc in 0:10
                        prob *= poisson_probablility(return_first_loc, RETURNS_FIRST_LOC) *
                            poisson_probablility(return_second_loc, RETURNS_SECOND_LOC)
                        reward = (rent_first + rent_second) * RENTAL_CREDIT
                        num_first = min(NUM_FIRST - rent_first + return_first_loc, MAX_CARS)
                        num_second = min(NUM_SECOND - rent_second + return_second_loc, MAX_CARS)
                        expected_return += prob * (reward + DISCOUNT * state_value[num_first + 1, num_second + 1])
                    end
                end
            end
        end
    end

    expected_return
end

function figure_4_2(constant_return=true)
    values = zeros(Float64, (MAX_CARS + 1, MAX_CARS + 1))
    policy = zeros(Int, size(values))

    iteration = 0
    master_fig, axes = PyPlot.subplots(2, 3, figsize=(40, 20))
    subplots_adjust(wspace=0.1, hspace=0.2)

    while true
        row = div(iteration, 3) + 1
        col = rem(iteration, 3) + 1
        fig = heatmap(policy, cmap="YlGnBu", ax=axes[row, col])
        axes[row, col].invert_yaxis()
        fig.set_ylabel("# cars at first location", fontsize=30)
        fig.set_xlabel("# cars at second location", fontsize=30)
        fig.set_title("policy $iteration", fontsize=30)

        # policy evaluation
        while true
            old_values = copy(values)
            for i in 0:MAX_CARS
                for j in 0:MAX_CARS
                    new_state_value = expected_return([i, j], policy[i + 1, j + 1], values, constant_return)
                    values[i + 1, j + 1] = new_state_value
                end
            end

            max_value_change = maximum(map((x) -> abs(x), (old_values - values)))
            println("Max value change: $max_value_change")
            if max_value_change < 1e-4
                break
            end
        end

        # policy improvement
        policy_stable = true
        for i in 0:MAX_CARS
            for j in 0:MAX_CARS
                old_action = policy[i + 1, j + 1]
                action_returns = []
                for action in actions
                    if (0 <= action <= i) || (-j <= action <= 0)
                        action_return = expected_return([i + 1, j + 1], action, values, constant_return)
                        append!(action_returns, action_return)
                    else
                        append!(action_returns, -Inf)
                    end
                end
                new_actions = findall(-TOLERANCE .< action_returns .- maximum(action_returns) .< TOLERANCE)
                map!((x) -> actions[x], new_actions, new_actions)
                policy[i + 1, j + 1] = new_actions[1]
                if policy_stable && !in(old_action, new_actions)
                    policy_stable = false
                end
            end
        end
        println("Policy stable $policy_stable")

        iteration += 1

        if iteration == 6
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
    figure_4_2(false)
end

main()
