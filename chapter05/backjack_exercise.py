import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm

EPS = [10000, 500000]
VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def set_axes_equal(ax):
    ''' https://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().

    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def deal_card(hand, ace):
    card = VALUES[np.random.randint(13)]

    if card == 1 and hand < 11:
        new_hand = hand + 11
        new_ace = 1
    else:
        new_hand = hand + card
        new_ace = ace

    if new_hand > 21 and new_ace == 1:
        new_hand -= 10
        new_ace = 0

    return new_hand, new_ace


def compute_reward(player_hand, dealer_hand, dealer_ace):
    if player_hand > 21:
        reward = -1
    else:
        # Dealer's round to stick on only 17 or higher
        while dealer_hand < 17:
            dealer_hand, dealer_ace = deal_card(
                dealer_hand, dealer_ace)

        if dealer_hand > 21:
            reward = 1
        else:
            if dealer_hand > player_hand:
                reward = -1
            elif dealer_hand == player_hand:
                reward = 0
            else:
                reward = 1

    return reward


def figure_5_1():
    plt.rc("font", size=6)
    fig = plt.figure(dpi=300)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    fig_count = 1

    for eps in EPS:
        values = np.zeros(shape=(10, 10, 2))
        visits = np.zeros(shape=(10, 10, 2))
        for _ in tqdm(range(eps)):
            state_sequence = []
            # Deal dealer
            dealer_showing = VALUES[np.random.randint(13)]
            if dealer_showing == 1:
                dealer_hand = 11
                dealer_ace = 1
            else:
                dealer_hand = dealer_showing
                dealer_ace = 0
            dealer_hand, dealer_ace = deal_card(dealer_hand, dealer_ace)
            # Deal player
            player_hand = 0
            player_ace = 0
            player_hand, player_ace = deal_card(player_hand, player_ace)
            player_hand, player_ace = deal_card(player_hand, player_ace)
            # Go to at least 12
            while player_hand < 12:
                player_hand, player_ace = deal_card(player_hand, player_ace)

            # Execute policy that sticks on only 20 and 21
            while player_hand < 20:
                state_sequence.append((player_hand - 12, player_ace))
                player_hand, player_ace = deal_card(player_hand, player_ace)

            if player_hand < 22:
                state_sequence.append((player_hand - 12, player_ace))

            reward = compute_reward(player_hand, dealer_hand, dealer_ace)

            for index in range(len(state_sequence) - 1, -1, -1):
                if state_sequence[index] not in state_sequence[:index]:
                    index_tuple = (
                        state_sequence[index][0], dealer_showing - 1, state_sequence[index][1])
                    visits[index_tuple] += 1
                    old_value = values[index_tuple]
                    new_value = old_value + \
                        (reward - old_value) / \
                        visits[index_tuple]
                    values[index_tuple] = new_value

        ax = fig.add_subplot(2, 2, fig_count, projection='3d')
        fig_count += 1
        x = np.linspace(1, 10, 10)
        y = np.linspace(12, 21, 10)
        X, Y = np.meshgrid(x, y)
        ax.plot_wireframe(X, Y, values[:, :, 1], linewidth=1)
        ax.set_xlabel("Dealer showing")
        ax.set_ylabel("Player sum")
        ax.set_title(f"usable ace, {eps} episodes")
        set_axes_equal(ax)

        ax = fig.add_subplot(2, 2, fig_count, projection='3d')
        fig_count += 1
        x = np.linspace(1, 10, 10)
        y = np.linspace(12, 21, 10)
        X, Y = np.meshgrid(x, y)
        ax.plot_wireframe(X, Y, values[:, :, 0], linewidth=1)
        ax.set_xlabel("Dealer showing")
        ax.set_ylabel("Player sum")
        ax.set_title(f"no usable ace, {eps} episodes")
        set_axes_equal(ax)

    plt.savefig("./my_images/figure_5_1.png")


def figure_5_2():
    eps = 50000
    policies = np.zeros(10, 10, 2)  # 0 for stick and 1 for hit
    # player_hand, dealer_showing, usable ace, action
    action_values = np.zeros(10, 10, 2, 2)
    visits = np.zeros(10, 10, 2, 2)

    for _ in range(eps):
        # Exploring start
        player_hand = np.random.randint(12, 22)
        if np.random.randint(0, 13) == 0:
            player_ace = 1
        else:
            player_ace = 0
        policy = np.random.randint(0, 2)

        # Dealer's hand
        dealer_showing = VALUES[np.random.randint(13)]
        if dealer_showing == 1:
            dealer_hand = 11
            dealer_ace = 1
        else:
            dealer_hand = dealer_showing
            dealer_ace = 0
        dealer_hand, dealer_ace = deal_card(dealer_hand, dealer_ace)

        state_sequence = []

        # Execute policy
        while policy == 1 and player_hand < 22:
            state_sequence.append((player_hand, dealer_showing, player_ace, 1))
            player_hand, player_ace = deal_card(player_hand, player_ace)
            policy = policies[player_hand - 12, dealer_showing, player_ace]

        if player_hand < 22:
            state_sequence.append((player_hand, dealer_showing, player_ace, 0))

        reward = compute_reward(player_hand, dealer_hand, dealer_ace)


def main():
    figure_5_1()


if __name__ == "__main__":
    main()
