import torch, wandb, uuid, os
import numpy as np
from numpy.linalg import norm
from numpy.random import seed, randn

from plotly import graph_objs as go
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

def get_rand_orthogonal_vectors(dim):
    """
    Generate two orthonormal vectors in R^dim.

    The first vector is the normalized all-ones vector.
    The second vector is a random vector orthogonal to the first, normalized to unit length.
    The random seed is set for reproducibility.

    Args:
        dim (int): The dimension of the vectors.

    Returns:
        tuple: A tuple (d1, d2) where both are numpy arrays of shape (dim,), orthonormal to each other.
    """
    d1 = np.ones(dim)
    d1 = d1 / norm(d1)
    d2 = randn(dim)
    
    while np.abs(np.abs(d1 @ d2 / (norm(d1) * norm(d2))) - 1) <= 1e-5:  # ensure d1 and d2 are not collinear
        d2 = torch.randn(dim)
    d2 = d2 - (d2 @ d1) * d1
    d2 = d2 / norm(d2)
    
    return d1, d2

def get_orthogonal_vectors(dim):
    """
    Generate two orthogonal vectors in a deterministic way for a given dimension.
    Returns:
        v: A vector of ones, normalized.
        u: A vector orthogonal to v
    """
    v = np.ones(dim)
    v = v / norm(v)
    # provided by chatgpt to maximize the effect of parameters by argmax(min(abs(u_i))))
    if dim % 2 == 0:
        # Even n: half +1/√n, half -1/√n
        half = dim // 2
        u = np.array([1]*half + [-1]*half, dtype=float)
        u /= np.linalg.norm(u)
    else:
        # Odd n: m of +α, m+1 of -β, with α > β
        m = dim // 2
        alpha = np.sqrt((dim+1) / (dim * (dim-1)))
        beta  = np.sqrt((dim-1) / (dim * (dim+1)))
        u = np.array([alpha]*m + [-beta]*(m+1), dtype=float)  
    return v, u

@torch.no_grad
def illuminate_loss_surface_2d(center, parameter_to_loss, range_lim=0.1, resolution=100):
    """
    Computes the 2D loss surface around the current model parameters by evaluating the loss
    on a grid defined by two orthogonal directions in parameter space.

    Args:
        center: The center point in parameter space (numpy array).
        parameter_to_loss: Function that maps parameters to loss values.
        resolution (int): Number of points along each axis of the grid.
        range_lim (float): Range limit for the grid in parameter space.

    Returns:
        parameter_grid (np.ndarray): Array of shape [resolution, resolution, num_parameters]
            containing the parameter vectors for each grid point.
        loss_grid (np.ndarray): Array of shape [resolution, resolution] containing the loss
            values evaluated at each grid point.
    """
    center = np.asarray(center, dtype=np.float32)  # ensure center is a numpy array
    if center.ndim != 1:
        raise ValueError("Center must be a 1D array representing the model parameters.")
    
    # create two orthogonal directions in the parameter space
    d1, d2 = get_orthogonal_vectors(center.size)

    # build grid
    axis = np.linspace(-range_lim, range_lim, resolution)
    alphas, betas = np.meshgrid(axis, axis, indexing="ij")  # shape: [resolution, resolution]
    alphas = np.expand_dims(alphas, -1)  # shape: [resolution, resolution, 1]
    betas = np.expand_dims(betas, -1)
    offset = alphas * d1 + betas * d2
    parameter_grid = center + offset  # shape: [resolution, resolution, dim_parameter]

    # sweep
    flat = parameter_grid.reshape(-1, parameter_grid.shape[-1])  # shape: [resolution*resolution, dim_parameter]
    loss_grid = np.apply_along_axis(parameter_to_loss, axis=1, arr=flat)  # shape: [resolution*resolution]
    loss_grid = loss_grid.reshape(resolution, resolution)  # shape: [resolution, resolution]

    return parameter_grid, loss_grid

def plot_loss_surface_2d(loss_grid, range_lim=0.1):
    resolution = loss_grid.shape[0]
    alphas = np.linspace(-range_lim, range_lim, resolution)
    betas = np.linspace(-range_lim, range_lim, resolution)

    # build Plotly figure
    fig = go.Figure(
        data=go.Surface(
            z=loss_grid,
            x=alphas,  # horizontal axis (α)
            y=betas,  # vertical axis (β)
            colorscale="Viridis",
            colorbar=dict(title="Loss"),
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="α",
            yaxis_title="β",
            zaxis_title="Loss",
        ),
        margin=dict(l=0, r=0, b=0, t=0),
    )
    return fig

def log_loss_plot(fig, run):
    run.log({f"{run.id}-loss-surface": fig})

def log_loss_grids(parameter_grid, loss_grid, run):
    """
    Logs the parameter grid and loss grid as a W&B artifact.

    Args:
        parameter_grid (np.ndarray): Array containing parameter vectors for each grid point.
        loss_grid (np.ndarray): Array containing loss values for each grid point.
        run (wandb.run): The active Weights & Biases run instance for logging the artifact.

    This function saves the provided grids as temporary .npy files, creates a W&B artifact,
    adds the files to the artifact, logs the artifact, and then removes the temporary files.
    """

    # paths for saving
    parameter_grid_path = f"wandb/tmp/{run.id}-{uuid.uuid4().hex[:8]}.npy"
    loss_grid_path = f"wandb/tmp/{run.id}-{uuid.uuid4().hex[:8]}.npy"

    # save
    os.makedirs("wandb/tmp/", exist_ok=True)
    np.save(parameter_grid_path, parameter_grid)
    np.save(loss_grid_path, loss_grid)

    # log
    artifact = wandb.Artifact(f"{run.id}-loss-surface", type="dataset")
    artifact.add_file(parameter_grid_path, name="parameter_grid.npy")
    artifact.add_file(loss_grid_path, name="loss_grid.npy")
    artifact.save()

    # remove file
    os.remove(parameter_grid_path)
    os.remove(loss_grid_path)

class LossSurfacePlotter:
    def __init__(self, filename=None):
        """
        Initializes the LossSurfacePlotter.
        """
        self.parameters = None
        self.loss = None
        self.dim = None
        self.path = filename
    
    def save_data(self):
        if self.path is None:
            raise ValueError("No path specified for saving data.")    
        np.savez(self.path, parameters=self.parameters, loss=self.loss)
    
    def read_data(self, filename):
        data = np.load(filename)
        self.add_loss(data['parameters'], data['loss'], save_data=False)

    def add_loss(self, parameter_grid, loss_grid, save_data=True):
        """
        Adds new parameter and loss data to the plotter.

        Args:
            paramter_grid (np.ndarray): Parameter grid, shape (resolution, resolution, nb_parameters) or (nb_samples, nb_parameters).
            loss_grid (np.ndarray): Loss grid, shape (resolution, resolution) or (nb_samples,).

        Notes:
            - parameter_grid shape: (resolution, resolution, nb_parameters) or just (nb_samples, nb_parameters)
            - loss_grid shape: (resolution, resolution) or just (nb_samples,)
        """
        # initialize for the first time
        if self.dim is None:
            self.dim = parameter_grid.shape[-1]
            self.parameters = np.empty([0, self.dim])
            self.loss = np.empty([0,])
            
        # remove the resolution dimension if it exists
        parameter_grid = parameter_grid.reshape(-1, parameter_grid.shape[-1])
        loss_grid = loss_grid.reshape(-1)
        assert parameter_grid.shape[0] == loss_grid.shape[0]
        
        # add to the existing data
        self.parameters = np.vstack((self.parameters, parameter_grid))
        self.loss = np.hstack((self.loss, loss_grid))
        
        # save data
        if save_data and self.path is not None:
            self.save_data()

    def illuminate_2d(self, center, parameter_to_loss, range_lim=0.1, resolution=100):
        """
        Evaluates the loss surface in a 2D region around a center point and adds the results.

        Args:
            center (np.ndarray): Center point in parameter space.
            parameter_to_loss (callable): Function mapping parameters to loss.
            range_lim (float, optional): Range to explore around the center. Default is 0.1.
            resolution (int, optional): Number of points per axis. Default is 100.

        Returns:
            tuple: (parameter_grid, loss_grid) evaluated over the 2D region.
        """
        parameter_grid, loss_grid = illuminate_loss_surface_2d(center, parameter_to_loss, range_lim, resolution)
        self.add_loss(parameter_grid, loss_grid)
        return parameter_grid, loss_grid

    def get_plot(self, resolution=512):
        """
        Generates a 3D surface plot of the loss landscape using the collected parameter and loss data.

        Args:
            resolution (int, optional): Number of points per axis for interpolation grid. 

        Returns:
            plotly.graph_objs.Figure: 3D surface plot of the loss surface.
        """
        # project the parameters onto two orthogonal vectors
        u, v = get_orthogonal_vectors(self.dim)
        x, y = self.parameters @ u, self.parameters @ v

        # create a grid for the loss surface
        axis_min = np.min([x.min(), y.min()])
        axis_max = np.max([x.max(), y.max()])
        axis = np.linspace(axis_min, axis_max, resolution)
        X, Y = np.meshgrid(axis, axis, indexing="ij")
        
        # interpolate the loss values onto the grid
        z = griddata((x, y), self.loss, (X, Y), method="linear")
               
        # Create a mask to hide grid points that are too far from any illuminated data points using k-d tree nearest neighbor search
        points = np.column_stack([x, y])
        flat_grid = np.column_stack([X.ravel(), Y.ravel()])
        tree = cKDTree(points)
        distances, _ = tree.query(flat_grid, k=1) # distance to the nearest illuminated point
        distances = distances.reshape(X.shape)
        nearest_neighbor_distances = tree.query(points, k=2)[0][:, 1]  
        threshold = 5 * np.median(nearest_neighbor_distances)# threshold for illumination
        z[distances > threshold] = np.nan 

        # plot
        fig = go.Figure(data=[go.Surface(z=z, x=X, y=Y, colorscale="Viridis", colorbar=dict(title="Loss"))])
        fig.update_layout(
            title="Loss Surface",
            scene=dict(
                xaxis_title="direction 1",
                yaxis_title="direction 2",
                zaxis_title="Loss",
            ),
            margin=dict(l=0, r=0, b=0, t=0),
        )
        return fig