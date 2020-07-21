using PyPlot
using Seaborn

const CARDS = 1:13
const VALUES = vcat(1:10, [10, 10, 10])
const EPS = [10000, 500000]

function deal_card(hand::Int, usable_ace::Int)
    card::Int = rand(CARDS)
    if card == 1 && hand < 11
        hand += 11
        usable_ace = 2
    else
        hand += card
    end
    if hand > 21 && usable_ace == 2
        hand -= 10
        usable_ace = 1
    end
    return hand, usable_ace
end

function figure_5_1()
    PyPlot.rc("font", size=6)
    fig = figure()
    subplots_adjust(wspace=0.4, hspace=0.4)
    fig_count::Int = 1

    for eps in EPS
        values::Array{Float64} = zeros(10, 10, 2) # Dealer, player, usable ace
        visits::Array{Int} = zeros(10, 10, 2)
        for _ in 1:eps
            state_sequence::Array{Array{Int}} = []
            player_hand::Int = 0
            dealer_showing::Int = VALUES[rand(CARDS)]
            if dealer_showing == 1
                dealer_hand::Int = 11
                dealer_ace::Int = 2
            else
                dealer_hand = dealer_showing
                dealer_ace = 1
            end
            dealer_hand, dealer_ace = deal_card(dealer_hand, dealer_ace)
            player_ace::Int = 1
            # Initial hand
            player_hand, player_ace = deal_card(player_hand, player_ace)
            player_hand, player_ace = deal_card(player_hand, player_ace)
            # Get to at least 12
            while player_hand < 12
                player_hand, player_ace = deal_card(player_hand, player_ace)
            end
            # Policy that sticks only on 20 and 21
            while player_hand < 20
                push!(state_sequence, [player_hand, player_ace])
                player_hand, player_ace = deal_card(player_hand, player_ace)
            end
            if player_hand > 21
                reward::Int = -1
            else
                push!(state_sequence, [player_hand, player_ace])
                while dealer_hand < 17
                    dealer_hand, dealer_ace = deal_card(dealer_hand, dealer_ace)
                end
                if dealer_hand > 21
                    reward = 1
                else
                    if dealer_hand > player_hand
                        reward = -1
                    elseif dealer_hand == player_hand
                        reward = 0
                    else
                        reward = 1
                    end
                end
            end
            for i in length(state_sequence):-1:1
                if in(state_sequence[i], state_sequence[1:i - 1])
                    continue
                end
                state::Int = state_sequence[i][1] - 11
                usable_ace::Int = state_sequence[i][2]
                visits[state, dealer_showing, usable_ace] += 1
                old_value = values[state, dealer_showing, usable_ace]
                values[state, dealer_showing, usable_ace] =
                old_value + (reward - old_value) / visits[state, dealer_showing, usable_ace]
            end
        end
        
        ax = fig.add_subplot(2, 2, fig_count)
        fig_count += 1
        heatmap(values[:,:,2], cmap="YlGnBu", ax=ax, xticklabels=1:10, yticklabels=12:21)
        ax.set_xlabel("Dealer showing")
        ax.set_ylabel("Player sum")
        ax.set_title("usable ace, $eps episodes")
        ax.invert_yaxis()

        ax = fig.add_subplot(2, 2, fig_count)
        fig_count += 1
        heatmap(values[:,:,1], cmap="YlGnBu", ax=ax, xticklabels=1:10, yticklabels=12:21)
        ax.set_xlabel("Dealer showing")
        ax.set_ylabel("Player sum")
        ax.set_title("no usable ace, $eps episodes")
        ax.invert_yaxis()
    end
    savefig("test.png")
end

function main()
    figure_5_1()
end

main()
