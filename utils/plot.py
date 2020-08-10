from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import numpy as np
import mpl_toolkits.mplot3d as plt3d
from mpl_toolkits.mplot3d import Axes3D

def rect(ax, poke, color, min_z=-1, max_z=1, scale_factor=1, pred=False):
    x, y, z = poke
    # z is positive pointing towards me.
    z = -z
    sign = 1

    dx = 70 * (x / scale_factor)
    dz = 70 * (z / scale_factor)

    cmap = cm.seismic
    norm = Normalize(vmin=min_z, vmax=max_z)
    c = cmap(norm(y))
    if norm(y) < 0.5:
        sign = -1

    ax.arrow(100, 100, dx, dz, width=1, head_width=7, head_length=6, color=color)

    # up down arrow

    if pred:
        ax.arrow(25, 190, 0, sign * 20, width=4, head_width=10, head_length=8, color=c)
    else:
        ax.arrow(10, 190, 0, sign * 20, width=4, head_width=10, head_length=8, color=c)

def plot_single(
        img_before,
        real_action_pos, real_action_ang,
        gripper_label=-1, gripper_logits=[],
        predicted_action_pos=[], predicted_action_ang=[],
        img_name=None, min_z=-1, max_z=1,scale_factor=1, verbose=True):

    fontsize=16
    if len(predicted_action_ang) > 0:
        predicted_action_ang = predicted_action_ang.cpu().detach().numpy()

    fig = plt.figure(figsize=(16,9))
    fig.suptitle('Aqua is real, yellow is predicted')
    axes = fig.add_subplot(2, 3, 1)
    ax1 = axes
    ax1.text(-130, -20, "Img:" + img_name, fontsize=fontsize)
    ax1.imshow(img_before.copy())

    # Angle subplot
    ax3 = fig.add_subplot(2, 3, 2, projection='3d')

    ax3.set_xlim3d(-1, 1)
    ax3.set_ylim3d(-1,1)
    ax3.set_zlim3d(-1,1)

    # origin
    s = 1
    for x, y, z, c in [(s, 0, 0, 'black'), (0, s, 0, 'black'), (0, 0, s, 'black')]:
        line = plt3d.art3d.Line3D((0, x), (0, y), (0, z), color=c, linewidth=2)
        ax3.add_line(line)

    if len(predicted_action_pos) > 0:
        prx = ("{0:.5f}".format(predicted_action_pos[0]))
        pry = ("{0:.5f}".format(predicted_action_pos[1]))
        prz = ("{0:.5f}".format(predicted_action_pos[2]))
        rect(ax1, predicted_action_pos, "yellow", pred=True)
        if verbose:
            ax1.text(0, 300, "Predicted Action is " + str(prx) + ", " + str(pry) + ", " + str(prz), ha='left', fontsize=fontsize)

            ax1.text(0, 380, "Predicted Angle is " + str(predicted_action_ang), ha='left', fontsize=fontsize)

        # Angle
        dotted = predicted_action_ang.dot(np.array([1, 0, 0]))
        ax3.quiver(
            0, 0, 0,  # <-- starting point of vector
            dotted[0], dotted[1], dotted[2],  # <-- directions of vector
            color='lime', alpha=.8, lw=1,
        )

    if len(real_action_pos) > 0:
        rect(ax1, real_action_pos, "aqua")

        px = ("{0:.5f}".format(real_action_pos[0]))
        py = ("{0:.5f}".format(real_action_pos[1]))
        pz = ("{0:.5f}".format(real_action_pos[2]))
        if verbose:
            ax1.text(00, 400, "Label (Scaled) Action is " + str(px) + ", " + str(py) + ", " + str(pz), ha='left', fontsize=fontsize)
            ax1.text(00, 470, "Label Angle is " + str(real_action_ang), fontsize=fontsize)

        # Angle plot
        dotted = np.array(real_action_ang).dot(np.array([1, 0, 0]))
        ax3.quiver(
            0, 0, 0,  # <-- starting point of vector
            dotted[0], dotted[1], dotted[2],  # <-- directions of vector
            color='blue', alpha=.8, lw=1,
        )


        if len(predicted_action_pos) > 0:
            def cos_loss(real, pred):
                top = np.dot(real, pred)
                bottom = np.linalg.norm(real) * np.linalg.norm(pred)
                eps = 1e-7
                divded = np.clip(top/bottom, -1+eps, 1-eps)
                z = np.arccos(divded)
                return z

            ax1.text(00, 510, "L2 dist is " + str(np.linalg.norm(real_action_pos - predicted_action_pos)) , fontsize=fontsize)

    return plt
