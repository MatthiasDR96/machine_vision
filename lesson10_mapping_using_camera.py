import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def depth_to_pcl(depth, depth_K):
    # Invert K matrix
    depth_K_inv = np.linalg.inv(depth_K)
    depth_K1 = depth_K_inv[0]
    depth_K2 = depth_K_inv[1]
    depth_K3 = depth_K_inv[2]

    # Init pointcloud arrays
    xarray = np.array([])
    yarray = np.array([])
    deptharray = np.array([])

    # Iterate over all pixels
    for u in range(50, 600):
        for v in range(250, 450):
            uv = np.array([u, v, 1])
            # Remove points which are too close or too far
            if (not np.isnan(depth[v][u])) and (depth[v][u] > 0.2) and (depth[v][u] < 3.0):
                x = np.array((np.dot(depth_K1, uv) / np.dot(depth_K3, uv)) * depth[v][u])
                y = np.array((np.dot(depth_K2, uv) / np.dot(depth_K3, uv)) * depth[v][u])
                # Remove points which are too low (ground)
                if y < 0.3:
                    # Add 3D point to pointcloud array
                    xarray = np.append(xarray, [x])
                    yarray = np.append(yarray, [y])
                    deptharray = np.append(deptharray, depth[v][u])
    # Concatenate X, Y, and Z into pointcloud
    pointcloud = np.array([xarray, yarray, deptharray])
    return pointcloud


def pcl_to_2d(pcl):
    pclx = pcl[2]
    pcly = -pcl[0]
    twoD = np.array([pclx, pcly])
    return twoD


if __name__ == "__main__":
    
    # Load camera data
    data = np.load('data_files/rgbd_data.npy')
    depth_K = data['depth_K']
    depth = data['depth']

    # Show RGB image
    img = data['rgb'] / 255.0
    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(depth)
    plt.subplot(1, 2, 2)
    plt.imshow(img)

    # RGB data to 3D pointcloud
    pointcloud = depth_to_pcl(depth, depth_K)

    # Show 3D pointcloud
    fig1 = plt.figure(2)
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(pointcloud[0, :], -pointcloud[2, :], pointcloud[1, :])
    ax1.set_xlabel('X Label')
    ax1.set_ylabel('Y Label')
    ax1.set_zlabel('Z Label')
    ax1.scatter([0], [0], [0], c='r')
    ax1.invert_xaxis()

    # 3D pointcloud to 2D pointcloud
    ogrid = pcl_to_2d(pointcloud)

    # Show 2D pointcloud
    fig2 = plt.figure(3)
    ax2 = fig2.gca()
    ax2.axis('equal')
    ax2.scatter(ogrid[1, :], ogrid[0, :], s=1)
    ax2.scatter([0], [0], c='r')
    ax2.invert_xaxis()
    plt.show()
