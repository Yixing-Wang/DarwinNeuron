################################ Generate Randman Problem ################################
import os
import pandas as pd

from dataclasses import dataclass
from src.Models import calc_randmansnn_parameters
from src.RandmanFunctions import RandmanConfig, generate_and_save_randman, match_config
from src.Utilities import next_id

@dataclass
class RandmanProblemConfig:
    randman_id: int
    nb_hidden: int
    loss_fn: str

    @classmethod
    def lookup_by_id(cls, table_path, id: int):
        """
        Lookup a row by id in a CSV file.
        """
        df = pd.read_csv(table_path, index_col="id")
        if id not in df.index:
            raise ValueError(f"ID {id} not found in {table_path}.")
        return cls(df.loc[id, "randman_id"], df.loc[id, "nb_hidden"], df.loc[id, "loss_fn"])

def generate_randman_problem(randman_config: RandmanConfig, nb_hidden=40, loss_fn = 'cross_entropy', problem_dir="data/First-experiment"):
    randman_df = pd.read_csv("data/randman/meta-data.csv")

    # Find matching randman configuration
    randman_row = match_config(randman_df, randman_config)

    # generate a new randman if no matching configuration is found
    if randman_row.empty:
        generate_and_save_randman(randman_config, "data/randman")
        randman_df = pd.read_csv("data/randman/meta-data.csv")
        randman_row = match_config(randman_df, randman_config)

    # select the first (and the only) matching row
    randman_row = randman_row.iloc[0]
    # create entry for problems.csv
    new_problem = {"randman_id": randman_row["id"], 
                "nb_hidden": nb_hidden, 
                "loss_fn": loss_fn,
                "dim": calc_randmansnn_parameters(randman_config.nb_units, nb_hidden, randman_config.nb_classes)}

    problem_path = os.path.join(problem_dir, "problems.csv")
    
    # check if problems.csv exists
    if os.path.exists(problem_path):
        problem_df = pd.read_csv(problem_path, index_col="id")
        
        # check if randman config already exists in problems.csv
        match = problem_df.query(f"randman_id == {randman_row['id']} & nb_hidden == {nb_hidden}")
        if not match.empty:
            raise ValueError(f"Problem with randman_id {randman_row['id']} and nb_hidden {nb_hidden} already exists.")


        problem_df = pd.concat([problem_df, pd.DataFrame(new_problem, index=[next_id(problem_df)])])
    else:
        problem_df = pd.DataFrame(new_problem, index=[0])

    problem_df.to_csv(problem_path, index_label="id")
    
##################################### Generate Parameter Samples #####################################
import uuid, os
import numpy as np
import pandas as pd
from dataclasses import dataclass
from pflacco.sampling import create_initial_sample
from dataclasses import asdict
from src.Utilities import next_id, match_config

@dataclass
class ParameterSampleConfig:
    dim: int = 3
    nb_sample: int = 1024
    method: str = "sobol"
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    
    @classmethod
    def lookup_by_id(cls, table_path, id: int):
        """
        Lookup a row by id in a CSV file.
        """
        df = pd.read_csv(table_path, index_col="id")
        if id not in df.index:
            raise ValueError(f"ID {id} not found in {table_path}.")
        return cls(df.loc[id, "dim"], df.loc[id, "nb_sample"], 
                   df.loc[id, "method"], df.loc[id, "lower_bound"], df.loc[id, "upper_bound"])
        
    def read_dataset(self, save_dir="data/samples"):
        meta_path = os.path.join(save_dir, "samples.csv")
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(f"Meta-data file not found at {meta_path}")

        df = pd.read_csv(meta_path)
        match = match_config(df, self)
        if match.empty:
            raise ValueError("No dataset found with the specified parameters.")

        filename = match.iloc[0]["filename"]
        filepath = os.path.join(save_dir, filename)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Dataset file not found at {filepath}")

        data = np.load(filepath)
        return data

def generate_and_save_samples(sample_config: ParameterSampleConfig, nb_versions=30, sample_dir="data/samples"):
    os.makedirs(sample_dir, exist_ok=True)
    csv_path = os.path.join(sample_dir, "samples.csv")
    
    samples_df = None
    if os.path.exists(csv_path):
        samples_df = pd.read_csv(csv_path, index_col="id")
        
        # check whether config already exists
        if not match_config(samples_df, sample_config).empty:
            raise ValueError(f"Sample with config {sample_config} already exists. Use add_sample()") # use case: add more samples
    
    new_rows = []
    for version in range(nb_versions):                 
        # create sample
        sample = create_initial_sample(sample_config.dim, 
                                       n=sample_config.nb_sample, 
                                       lower_bound=sample_config.lower_bound, 
                                       upper_bound=sample_config.upper_bound, 
                                       sample_type=sample_config.method,
                                       seed=version)
        
        # Save the sample
        filename = f"{uuid.uuid4().hex}.npy"
        sample_path = os.path.join(sample_dir, filename)
        np.save(sample_path, sample.to_numpy())
        
        # new record
        config_dict = asdict(sample_config)
        config_dict.update({
            "version": version,
            "filename": filename
        })
        new_rows.append(config_dict)
        
    # Append new rows to df
    new_ids = np.array(range(nb_versions))+next_id(samples_df)
    new_rows = pd.DataFrame(new_rows, index=new_ids)
    samples_df = pd.concat([samples_df, new_rows]) # pd.concat can handle samples_df = None
    samples_df.to_csv(csv_path, index_label="id")
    
#################################### Match problem and sample ####################################
from dataclasses import dataclass
from src.Utilities import match_config
import pandas as pd

@dataclass
class LossSurfaceConfig:
    problem_id: int
    sample_id: int
    
    @classmethod
    def lookup_by_id(cls, table_path, id: int):
        """
        Lookup a row by id in a CSV file.
        """
        df = pd.read_csv(table_path, index_col="id")
        if id not in df.index:
            raise ValueError(f"ID {id} not found in {table_path}.")
        return cls(df.loc[id, "problem_id"], df.loc[id, "sample_id"])
    
    def read_sample(self, sample_dir="data/samples"):
        """
        Read the sample associated with this loss surface configuration.
        """
        sample_config = ParameterSampleConfig.lookup_by_id(os.path.join(sample_dir, "samples.csv"), self.sample_id)
        return sample_config.read_dataset(sample_dir)
    
    def read_loss(self, loss_surface_dir="data/First-experiment", loss_dir="data/First-experiment/losses"):
        """
        Read the loss associated with this loss surface configuration.
        """
        loss_surface_df = pd.read_csv(os.path.join(loss_surface_dir, "loss-surfaces.csv"), index_col="id")
        match = match_config(loss_surface_df, self)
        if match.empty:
            raise ValueError(f"No loss surface found for configuration {self}.")
        loss_filename = match.iloc[0]["loss_filename"]
        return np.load(os.path.join(loss_dir, loss_filename))

def assign_samples_to_problem(problem_id: int, sample_config: LossSurfaceConfig, nb_versions=None, loss_surface_dir="data/First-experiment", sample_dir="data/samples"):    
    # find all the samples which match the sample_config
    samples_df = pd.read_csv(os.path.join(sample_dir, "samples.csv"), index_col="id")
    
    matched_samples = match_config(samples_df, sample_config)    
    
    # make sure the samples exist
    if matched_samples.empty:
        raise ValueError(f"No samples found matching the config {sample_config}. Please generate samples first.")
        
    ## add the matched samples, along with problem_id, to loss-surfaces.csv
    # create new rows for loss-surfaces.csv
    sample_ids = matched_samples.index
    new_metric_rows = pd.DataFrame({
        'problem_id': problem_id,
        'sample_id': sample_ids,
    })
    
    metric_df = None
    metric_path = os.path.join(loss_surface_dir, "loss-surfaces.csv")
    os.makedirs(loss_surface_dir, exist_ok=True)
    if os.path.exists(metric_path):
        metric_df = pd.read_csv(metric_path, index_col="id")
        
        # check if any of the problem-sample combinations already exist in loss-surfaces.csv
        existing_mask = new_metric_rows.set_index(['problem_id', 'sample_id']).index.isin(
            metric_df.set_index(['problem_id', 'sample_id']).index
        )    
        if existing_mask.any():
            existing_combinations = new_metric_rows[existing_mask]
            raise ValueError(f"Some problem-sample combinations already exist in loss-surfaces.csv: {existing_combinations.to_dict('records')}")

    # add new rows to loss-surfaces.csv
    new_ids = np.array(range(len(new_metric_rows))) + next_id(metric_df)
    new_metric_rows.index = new_ids
    metric_df = pd.concat([metric_df, new_metric_rows])
    metric_df.to_csv(metric_path, index_label="id")
    
################################### Calculate Loss ###################################
import os, uuid, portalocker
import numpy as np
import pandas as pd
from torch.nn.functional import cross_entropy
from src.RandmanFunctions import split_and_load
from src.LandscapeAnalysis import get_parameter_to_loss_fn
from src.Models import RandmanSNN

def calculate_and_save_loss(loss_surface_id: int, 
                            loss_surface_dir='data/First-experiment', 
                            problem_dir='data/First-experiment',
                            sample_dir='data/samples',
                            randman_dir='data/randman',
                            loss_dir='data/First-experiment/losses',
                            device='cuda'):
    # read configurations
    loss_surface_config = LossSurfaceConfig.lookup_by_id(os.path.join(loss_surface_dir, "loss-surfaces.csv"), loss_surface_id)
    problem_config = RandmanProblemConfig.lookup_by_id(os.path.join(problem_dir, "problems.csv"), loss_surface_config.problem_id)
    sample_config = ParameterSampleConfig.lookup_by_id(os.path.join(sample_dir, "samples.csv"), loss_surface_config.sample_id)
    randman_config = RandmanConfig.lookup_by_id(os.path.join(randman_dir, "meta-data.csv"), problem_config.randman_id)
    train_loader,_ = split_and_load(randman_config.read_dataset(randman_dir), batch_size=516)
    model = RandmanSNN(randman_config.nb_units, problem_config.nb_hidden, randman_config.nb_classes, learn_beta=False, beta=0.95)
    if problem_config.loss_fn == 'cross_entropy':
        loss_fn = cross_entropy     
    f = get_parameter_to_loss_fn(train_loader, model, loss_fn, device)
    samples = sample_config.read_dataset(sample_dir)
    
    # The computation part
    loss = np.apply_along_axis(f, 1, samples) 

    # record in loss-surfaces.csv
    loss_filename = f"{uuid.uuid4().hex}.npy"

    with open(os.path.join(loss_surface_dir, "loss-surfaces.csv"), 'r+', newline="") as f:
        portalocker.lock(f, portalocker.LOCK_EX)
        loss_surface_df = pd.read_csv(f, index_col="id")
        
        # check if the loss_filname already exists for this loss_surface_id
        if "loss_filename" in loss_surface_df.columns:
            val = loss_surface_df.loc[loss_surface_id, "loss_filename"]
            if (not pd.isna(val)) and (val != "pending"):
                raise ValueError(f"Loss filename already exists for loss_surface_id {loss_surface_id}: {val}")        
        
        loss_surface_df.loc[loss_surface_id, "loss_filename"] = loss_filename
        
        # write to file
        f.seek(0)
        f.truncate()
        loss_surface_df.to_csv(f, index_label="id")

        # make sure everything is on disk BEFORE we unlock
        f.flush()
        os.fsync(f.fileno())

        # release lock
        portalocker.unlock(f)

    # save loss
    os.makedirs(loss_dir, exist_ok=True)
    np.save(os.path.join(loss_dir, loss_filename), loss)
