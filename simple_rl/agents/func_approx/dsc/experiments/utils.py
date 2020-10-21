import scipy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy

def get_grid_states(mdp):
    ss = []
    x_low_lim, y_low_lim = mdp.get_x_y_low_lims()
    x_high_lim, y_high_lim = mdp.get_x_y_high_lims()
    for x in np.arange(x_low_lim, x_high_lim+1, 1):
        for y in np.arange(y_low_lim, y_high_lim+1, 1):
            ss.append(np.array((x, y)))
    return ss


def get_initiation_set_values(option):
    values = []
    x_low_lim, y_low_lim = option.overall_mdp.get_x_y_low_lims()
    x_high_lim, y_high_lim = option.overall_mdp.get_x_y_high_lims()
    for x in np.arange(x_low_lim, x_high_lim+1, 1):
        for y in np.arange(y_low_lim, y_high_lim+1, 1):
            pos = np.array((x, y))
            init = option.is_init_true(pos) and not option.overall_mdp.env.env._is_in_collision(pos)
            values.append(init)

    return values

def plot_one_class_initiation_classifier(option):

    colors = ["blue", "yellow", "green", "red", "cyan", "brown"]

    X = option.construct_feature_matrix(option.positive_examples)
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)
    Z1 = option.pessimistic_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z1 = Z1.reshape(xx.shape)

    color = colors[option.option_idx % len(colors)]
    plt.contour(xx, yy, Z1, levels=[0], linewidths=2, colors=[color])

def plot_two_class_classifier(option, episode, experiment_name, plot_examples=True):
    states = get_grid_states(option.overall_mdp)
    values = get_initiation_set_values(option)

    x = np.array([state[0] for state in states])
    y = np.array([state[1] for state in states])
    xi, yi = np.linspace(x.min(), x.max(), 1000), np.linspace(y.min(), y.max(), 1000)
    xx, yy = np.meshgrid(xi, yi)
    rbf = scipy.interpolate.Rbf(x, y, values, function="linear")
    zz = rbf(xx, yy)
    plt.imshow(zz, vmin=min(values), vmax=max(values), extent=[x.min(), x.max(), y.min(), y.max()], origin="lower", alpha=0.6, cmap=plt.cm.coolwarm)
    plt.colorbar()

    # Plot trajectories
    positive_examples = option.construct_feature_matrix(option.positive_examples)
    negative_examples = option.construct_feature_matrix(option.negative_examples)

    if positive_examples.shape[0] > 0 and plot_examples:
        plt.scatter(positive_examples[:, 0], positive_examples[:, 1], label="positive", c="black", alpha=0.3, s=10)

    if negative_examples.shape[0] > 0 and plot_examples:
        plt.scatter(negative_examples[:, 0], negative_examples[:, 1], label="negative", c="lime", alpha=1.0, s=10)

    if option.pessimistic_classifier is not None:
        plot_one_class_initiation_classifier(option)

    # background_image = imageio.imread("four_room_domain.png")
    # plt.imshow(background_image, zorder=0, alpha=0.5, extent=[-2.5, 10., -2.5, 10.])

    name = option.name if episode is None else option.name + "_{}_{}".format(experiment_name, episode)
    plt.title("{} Initiation Set".format(option.name))
    plt.savefig("initiation_set_plots/{}/{}_initiation_classifier_{}.png".format(experiment_name, name, option.seed))
    plt.close()


def plot_initiation_distribution(option, mdp, episode, experiment_name, chunk_size=10000):
    assert option.initiation_distribution is not None
    data = mdp.dataset[:, :2]

    num_chunks = int(np.ceil(data.shape[0] / chunk_size))
    if num_chunks == 0:
        return 0.

    state_chunks = np.array_split(data, num_chunks, axis=0)
    pvalues = np.zeros((data.shape[0],))
    current_idx = 0

    for chunk_number, state_chunk in tqdm(enumerate(state_chunks)):
        probabilities = np.exp(option.initiation_distribution.score_samples(state_chunk))
        pvalues[current_idx:current_idx + len(state_chunk)] = probabilities
        current_idx += len(state_chunk)

    plt.scatter(data[:, 0], data[:, 1], c=pvalues)
    plt.colorbar()
    plt.title("Density Estimator Fitted on Pessimistic Classifier")
    plt.savefig(f"initiation_set_plots/{experiment_name}/{option.name}_initiation_distribution_{episode}.png")
    plt.close()