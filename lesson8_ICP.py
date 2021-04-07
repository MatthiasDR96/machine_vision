import matplotlib.pyplot as plt
import numpy as np
import scipy.spatial as spatial


def rigid_transform(P, Q):
    N = len(P)

    # Compute centroids
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)

    # Normalize clouds to their center
    PP = P - np.tile(centroid_P, (N, 1))
    QQ = Q - np.tile(centroid_Q, (N, 1))

    # Single value deconposition
    H = np.dot(PP.T, QQ)
    U, S, Vt = np.linalg.svd(H)

    # Get rotation matrix
    R = np.dot(Vt.T, U.T)

    # Get translation matrix
    t = np.array([centroid_Q.T - np.dot(R, centroid_P.T)])

    return R, t


def icp(ref_pcl, new_pcl, init_rot, init_trans, max_tolerance=0.001, max_iters=20, outlier_ratio=0.0, verbose=True):
    # Get data
    ref = ref_pcl
    new = new_pcl
    R_tot = np.identity(2)
    t_tot = np.array([np.zeros(2)])
    new = np.dot(init_rot, new) + init_trans.T

    prev_error = 0

    # Make KD tree with reference cloud
    tree = spatial.KDTree(ref.T)

    # Iterate
    for i in range(max_iters):

        n = new.shape[1]
        number = 200
        indexes = np.random.choice(n, number)
        samples = new[:, indexes]

        # Closest points
        distances, indices = tree.query(samples.T)

        # Rigid transformation
        R, t = rigid_transform(samples.T, ref[:, indices].T)
        R_tot = R_tot.dot(R)
        t_tot = t_tot + t

        # New pointcloud
        new = np.dot(R, new) + t.T
        mean_error = np.mean(distances)

        # End criterion
        if np.abs(prev_error - mean_error) < max_tolerance:
            break

        prev_error = mean_error

    return R_tot, t_tot


if __name__ == "__main__":

    # Load data
    data = np.load('data_files/data_2d.npy')
    odom = data['odom']
    twod = data['2D']

    # Show data
    fig = plt.figure(frameon=True)
    ax = fig.gca()
    ax.axis('equal')
    ax.invert_xaxis()

    # Plot initial position
    orientation = np.array([1, 0])
    ax.plot(0, 0)
    ax.quiver(0, 0, -orientation[1], orientation[0])

    # Get reference pointcloud
    ref_pcl = twod[0]
    ref_pcl = ref_pcl[:, ~np.isnan(ref_pcl[0, :])]
    ax.scatter(ref_pcl[1, :], ref_pcl[0, :], s=1, edgecolors="none", c='b')

    # Initial transformation matrix
    R_tot = np.identity(2)
    t_tot = np.array([np.zeros(2)])
    
    plt.show()

    # Show data
    fig = plt.figure(frameon=True)
    ax = fig.gca()
    ax.axis('equal')
    ax.invert_xaxis()

    # Plot initial position
    orientation = np.array([1, 0])
    ax.plot(0, 0)
    ax.quiver(0, 0, -orientation[1], orientation[0])

    # Get reference pointcloud
    ref_pcl = twod[0]
    ref_pcl = ref_pcl[:, ~np.isnan(ref_pcl[0, :])]
    ax.scatter(ref_pcl[1, :], ref_pcl[0, :], s=1, edgecolors="none", c='b')

    # Initial transformation matrix
    R_tot = np.identity(2)
    t_tot = np.array([np.zeros(2)])

    # Iterate over pointclouds
    plt.ion()
    for i in range(0, 25, 2):

        # Get new pointcloud
        new_pcl = twod[i]
        new_pcl = new_pcl[:, ~np.isnan(new_pcl[0, :])]
        ax.scatter(new_pcl[1, :], new_pcl[0, :], s=1, edgecolors="none", c='r')

        # Compute transformation
        R, t = icp(ref_pcl, new_pcl, R_tot, t_tot)
        R_tot = R_tot.dot(R)
        t_tot = t_tot + t
        new_pcl = np.dot(R_tot, new_pcl) + t_tot.T
        ax.scatter(new_pcl[1, :], new_pcl[0, :], s=1, edgecolors="none", c='y')

        # New position
        orientation = np.dot(R_tot, np.array([1, 0]))
        ax.plot(t_tot[:, 1], t_tot[:, 0])
        ax.quiver(t_tot[:, 1], t_tot[:, 0], -orientation[1], orientation[0])

        # Draw
        plt.draw()
        plt.pause(1)
    plt.show()