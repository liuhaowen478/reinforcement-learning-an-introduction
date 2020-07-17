using PyPlot

const P_H = 0.4

const THETA = 1e-9

const STATES = 1:99

const DISCOUNT = 1

function expected_return(state::Int, action::Int, state_values::Array{Float64})::Float64
    expected_return::Float64 = 0
    expected_return += P_H * (state_values[state + action + 1])
    expected_return += (1 - P_H) * (state_values[state - action + 1])
end

function figure_4_3()
    sweeps_history::Array{Array{Float64}} = []
    values::Array{Float64} = zeros(size(STATES)[1] + 2)
    values[101] = 1
    policy::Array{Int} = zeros(size(STATES))
    while true
        old_values::Array{Float64} = copy(values)
        push!(sweeps_history, old_values)
        for state in STATES
            actions::Array{Int} = 0:min(state, 100 - state)
            expected_returns::Array{Float64} = []
            for action in actions
                append!(expected_returns, expected_return(state, action, values))
            end
            values[state + 1] = maximum(expected_returns)
        end
        delta::Float64 = maximum(abs, old_values - values)
        println("Maximum value change: $delta")
        if delta < THETA
            push!(sweeps_history, values)
            break
        end
    end


    for state in STATES
        actions::Array{Int} = 0:min(state, 100 - state)
        expected_returns::Array{Float64} = []
        for action in actions
            append!(expected_returns, expected_return(state, action, values))
        end
        policy[state] = findmax(map((x) -> round(x, digits=5), expected_returns)[2:end])[2]
    end

    figure(figsize=(10, 20))
    subplot(2, 1, 1)

    for (index, state_values) in enumerate(sweeps_history)
        plot(state_values, label="sweep $index")
    end
    xlabel("Capital")
    ylabel("Value estimates")
    legend(loc="best")

    subplot(2, 1, 2)
    bar(STATES, policy)
    xlabel("Capital")
    ylabel("Final policy (stake)")

    savefig("test.png")
    PyPlot.close()
end

function main()
    figure_4_3()
end

main()
