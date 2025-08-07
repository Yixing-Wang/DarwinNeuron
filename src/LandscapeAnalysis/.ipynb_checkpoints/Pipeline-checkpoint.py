################################ Randman Problem ################################
import sqlite3
import pandas as pd

from dataclasses import dataclass
from typing import Callable
from abc import ABC, abstractmethod

from src.RandmanFunctions import split_and_load
from torch.nn.functional import cross_entropy
from src.Models import RandmanSNNConfig, RandmanSNN, calc_randmansnn_parameters
from src.RandmanFunctions import RandmanConfig, generate_and_save_randman, match_config

from src.LandscapeAnalysis.Utils import get_parameter_to_loss_fn

@dataclass
class ELAProblemConfig(ABC):
    """
    Abstract base class for all problem configurations in the ELA pipeline.
    """
    
    @classmethod
    @abstractmethod
    def lookup_by_id(cls, id: int, db_path='data/landscape-analysis.db') -> 'ELAProblemConfig':
        pass
    
    @abstractmethod
    def write_to_db(self, dim: int, db_path='data/landscape-analysis.db') -> None:
        pass
    
    @abstractmethod
    def get_id(self, db_path='data/landscape-analysis.db') -> int:
        pass
    
    @abstractmethod
    def get_type(self) -> str:
        pass
    
    @abstractmethod
    def get_loss_surface_fn(self, *args) -> Callable:
        pass

@dataclass
class NNProblemConfig(ELAProblemConfig):
    data_type: str
    data_id: int
    model_type: str
    model_id: int
    loss_fn: str

    @classmethod
    def lookup_by_id(cls, id: int, db_path='data/landscape-analysis.db'):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute(f"SELECT data_type, data_id, model_type, model_id, loss_fn FROM nn_problems WHERE id = {id}")
        row = cur.fetchone()
        con.close()
        if row is None:
            raise ValueError(f"No NNProblemConfig found with id {id}.")

        return cls(*row) 
    
    def write_to_db(self, dim: int, db_path='data/landscape-analysis.db'):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        # Insert the new problem into the database. Note: uniqueness is enforced by db constraints
        try:
            cur.execute(f"""
                INSERT INTO nn_problems (data_type, data_id, model_type, model_id, loss_fn, dim) 
                VALUES ('{self.data_type}', {self.data_id}, '{self.model_type}', {self.model_id}, '{self.loss_fn}', {dim})
            """)
        except sqlite3.IntegrityError:
            con.close()
            print(f"Problem {self} already exists in the database.")
            raise
        
        con.commit()
        con.close()
        print(f"Added {self} to the nn_problems.")

    def get_id(self, db_path='data/landscape-analysis.db'):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        # Check if the problem configuration exists in the database
        cur.execute(f"""
            SELECT id FROM nn_problems 
            WHERE data_type = ? AND data_id = ? AND model_type = ? AND model_id = ? AND loss_fn = ?
        """, (self.data_type, self.data_id, self.model_type, self.model_id, self.loss_fn))
        
        row = cur.fetchone()
        con.close()
        
        if row is None:
            raise ValueError(f"No problem found with {self}.")
        
        return row[0]

    def get_type(self):
        return 'nn'

    def get_model(self, db_path='data/landscape-analysis.db', randman_table_path='data/randman/meta-data.csv'):
        if self.data_type == 'randman' and self.model_type == 'snn':
            data_config = RandmanConfig.lookup_by_id(self.data_id, randman_table_path)
            model_config = RandmanSNNConfig.lookup_by_id(self.model_id, db_path)
            model = RandmanSNN(data_config.nb_units, data_config.nb_classes, model_config)
        else:
            raise ValueError(f"Unsupported data type {self.data_type} or model type {self.model_type}.")
        return model

    def get_loader(self, db_path='data/landscape-analysis.db', randman_table_path='data/randman/meta-data.csv'):
        if self.data_type == 'randman':
            data_config = RandmanConfig.lookup_by_id(self.data_id, randman_table_path)
            loader, _ = split_and_load(data_config.read_dataset(), batch_size=516)
        else:
            raise ValueError(f"Unsupported data type {self.data_type}.")
        return loader

    def get_loss_surface_fn(self, db_path='data/landscape-analysis.db', randman_table_path='data/randman/meta-data.csv', device='cuda') -> Callable:
        if self.data_type == 'randman':
            loader = self.get_loader(db_path)
        if self.model_type == 'snn':
            model = self.get_model(db_path, randman_table_path)
        if self.loss_fn == 'cross_entropy':
            loss_fn = cross_entropy
        f = get_parameter_to_loss_fn(loader, model, loss_fn, device)
        return f

def generate_randman_problem(randman_config: RandmanConfig, snn_config: RandmanSNNConfig, loss_fn = 'cross_entropy', db_path='data/landscape-analysis.db'):
    ## 1) Randman
    randman_df = pd.read_csv("data/randman/meta-data.csv")
    randman_row = match_config(randman_df, randman_config)

    # Generate a new randman if no matching configuration is found
    if randman_row.empty:
        generate_and_save_randman(randman_config, "data/randman")
        randman_df = pd.read_csv("data/randman/meta-data.csv")
        randman_row = match_config(randman_df, randman_config)

    # Select the first (and the only) matching row
    randman_row = randman_row.iloc[0]

    # Calculate the dimension for the problem
    dim = calc_randmansnn_parameters(randman_config.nb_units, randman_config.nb_classes, snn_config)
    
    ## 2) RandmanSNN
    try:
        snn_id = snn_config.get_id(db_path)
    # create a new SNN configuration if it does not exist
    except ValueError:
        snn_config.write_to_db(db_path)
        snn_id = snn_config.get_id(db_path)

    ## 3) NNProblemConfig
    problem_config = NNProblemConfig(
        data_type='randman',
        data_id=randman_row['id'],
        model_type='snn',
        model_id=snn_id,
        loss_fn=loss_fn
    )
    problem_config.write_to_db(dim, db_path)
    print(f"Generated Randman problem with ID {problem_config.data_id} and SNN model ID {problem_config.model_id}.")
    
#################################### BBOB Problems ###################################
import cocoex

@dataclass
class BBOBProblemConfig(ELAProblemConfig):
    """
    Note: COCO's interface is bizarre. Up to 44 dims, we can use BareProblem to index arbitrary instances.
    Above 44 dims, we need to use Suite with only options in [80, 160, 320, 640] and up to 15 instances.
    """
    function_idx: int
    instance_idx: int
    dim: int
    
    @classmethod
    def lookup_by_id(cls, id: int, db_path='data/landscape-analysis.db'):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute(f"SELECT function_idx, instance_idx, dim FROM bbob_problems WHERE id = {id}")
        row = cur.fetchone()
        con.close()
        if row is None:
            raise ValueError(f"No BBOBProblemConfig found with id {id}.")
        
        return cls(*row)
    
    def write_to_db(self, db_path='data/landscape-analysis.db'):
        # check dims and indices
        available_large_dims = [80, 160, 320, 640]
        if self.dim > 44 and (self.dim not in available_large_dims or self.instance_idx > 15):
            raise ValueError(f"Invalid configuration: dim > 44 must be in {available_large_dims} and instance_idx must be in [1, 15].")
        
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        # Insert the new problem into the database. Note: uniqueness is enforced by db constraints
        try:
            cur.execute(f"""
                INSERT INTO bbob_problems (function_idx, instance_idx, dim) 
                VALUES ({self.function_idx}, {self.instance_idx}, {self.dim})
            """)
        except sqlite3.IntegrityError:
            con.close()
            print(f"Problem {self} already exists in the database.")
            raise
        
        con.commit()
        con.close()
        print(f"Added {self} to the bbob_problems.")

    def get_loss_surface_fn(self, *args) -> Callable:        
        if self.dim <= 44:
            problem = cocoex.BareProblem(
                suite_name="bbob",
                function=self.function_idx,
                dimension=self.dim,
                instance=self.instance_idx
            )
        else:
            suite = cocoex.Suite("bbob-largescale","",f"dimensions:{self.dim} function_indices:{self.function_idx} instance_indices:{self.instance_idx}")
            problem = suite[0]
        return problem
    
    def get_id(self, db_path='data/landscape-analysis.db'):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        # Check if the problem configuration exists in the database
        cur.execute(f"""
            SELECT id FROM bbob_problems 
            WHERE function_idx = ? AND instance_idx = ? AND dim = ?
        """, (self.function_idx, self.instance_idx, self.dim))
        
        row = cur.fetchone()
        con.close()
        
        if row is None:
            raise ValueError(f"No problem found with {self}.")
        
        return row[0]

    def get_type(self):
        return 'bbob'

##################################### Parameter Samples #####################################
import uuid, os, sqlite3
import numpy as np
from dataclasses import dataclass
from pflacco.sampling import create_initial_sample

@dataclass
class ParameterSampleConfig:
    dim: int
    nb_sample: int
    method: str
    lower_bound: float
    upper_bound: float
    # version is appended in lookup_by_id().
    
    @classmethod
    def lookup_by_id(cls, id: int, db_path='data/landscape-analysis.db'):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute(f"SELECT dim, nb_sample, method, lower_bound, upper_bound, version FROM samples WHERE id={id}")
        row = cur.fetchone()
        con.close()
        if row is None:
            raise ValueError(f"ID {id} not found in {db_path}.")

        result = cls(*row[:-1])
        result.version = row[-1]
        return result

    def add_samples(self, nb_versions: int, sample_dir="data/samples", db_path='data/landscape-analysis.db'):
        os.makedirs(sample_dir, exist_ok=True)
           
        min_version = self.get_nb_versions(db_path)
        
        filename_list = []
        sample_list = []
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        for version in range(min_version, min_version + nb_versions):                 
            # Create sample
            sample = create_initial_sample(self.dim, 
                                        n=self.nb_sample, 
                                        lower_bound=self.lower_bound, 
                                        upper_bound=self.upper_bound, 
                                        sample_type=self.method,
                                        seed=10000000 + (version * 99991) % 2**31)
            sample_list.append(sample)
            
            # name the file
            filename = f"{uuid.uuid4().hex}.npy"
            filename_list.append(filename)
            
            # Insert the new sample configuration into the database
            cur.execute("""
                INSERT INTO samples (dim, nb_sample, method, lower_bound, upper_bound, version, filename) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (self.dim, self.nb_sample, self.method, self.lower_bound, self.upper_bound, version, filename))
            print(f"Added sample {filename} with version {version} to {self}.")
            
        con.commit()
        con.close()
        
        # save after closing the database connection 
        for filename, sample in zip(filename_list, sample_list):
            filepath = os.path.join(sample_dir, filename)
            np.save(filepath, sample)

    def read_sample(self, sample_dir="data/samples", db_path='data/landscape-analysis.db'):
        if not hasattr(self, 'version'):
            raise AttributeError("version is not set. Use lookup_by_id() or manually set it.")
        
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        # Check if the sample configuration exists in the database
        cur.execute(f"""
            SELECT filename FROM samples 
            WHERE dim = ? AND nb_sample = ? AND method = ? 
            AND lower_bound = ? AND upper_bound = ? AND version = ?
        """, (self.dim, self.nb_sample, self.method, self.lower_bound, self.upper_bound, self.version))
        
        row = cur.fetchone()
        con.close()
        
        if row is None:
            raise ValueError(f"No sample found with {self}.")
        
        filename = row[0]
        filepath = os.path.join(sample_dir, filename)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Dataset file not found at {filepath}")
        
        data = np.load(filepath)
        return data

    def get_nb_versions(self, db_path='data/landscape-analysis.db'):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        cur.execute(f"""
            SELECT version FROM samples 
            WHERE dim = {self.dim} AND nb_sample = {self.nb_sample} 
            AND method = '{self.method}' AND lower_bound = {self.lower_bound} 
            AND upper_bound = {self.upper_bound}
        """)
        
        nb_versions = len(cur.fetchall())
        con.close()
        return nb_versions

    def get_id(self, version: int, db_path='data/landscape-analysis.db'):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        # Check if the sample configuration exists in the database
        cur.execute(f"""
            SELECT id FROM samples 
            WHERE dim = ? AND nb_sample = ? AND method = ? 
            AND lower_bound = ? AND upper_bound = ? AND version = ?
        """, (self.dim, self.nb_sample, self.method, self.lower_bound, self.upper_bound, version))
        
        row = cur.fetchone()
        con.close()
        
        if row is None:
            raise ValueError(f"No sample found with {self} and version {version}.")
        
        return row[0]

#################################### Match problem and sample ####################################
import os, sqlite3, numpy as np
from dataclasses import dataclass

@dataclass
class LossSurfaceConfig:
    problem_type: str
    problem_id: int
    sample_id: int

    @classmethod
    def lookup_by_id(cls, id: int, db_path="data/landscape-analysis.db"):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("SELECT problem_type, problem_id, sample_id FROM loss_surfaces WHERE id=?", (id,))
        row = cur.fetchone()
        con.close()
        if row is None:
            raise ValueError(f"ID {id} not found in {db_path}.")
        return cls(*row)

    def write_to_db(self, db_path="data/landscape-analysis.db"):
        """
        Write this loss surface configuration to the database.
        """
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        # Insert the new loss surface configuration into the database
        try:
            cur.execute("""
                INSERT INTO loss_surfaces (problem_type, problem_id, sample_id) 
                VALUES (?, ?, ?)
            """, (self.problem_type, self.problem_id, self.sample_id))
        except sqlite3.IntegrityError:
            con.close()
            print(f"Loss surface {self} already exists in the database.")
            raise
        
        con.commit()
        con.close()
        print(f"Added loss surface {self} to the loss_surfaces.")

    def read_sample(self, sample_dir="data/samples", db_path="data/landscape-analysis.db"):
        sample_config = ParameterSampleConfig.lookup_by_id(self.sample_id, db_path)
        return sample_config.read_sample(sample_dir, db_path)

    def read_loss(self, loss_dir="data/losses", db_path="data/landscape-analysis.db"):
        """
        Read the loss associated with this loss surface configuration using the database.
        """
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        cur.execute(f"""
            SELECT loss_filename FROM loss_surfaces 
            WHERE problem_type=? AND problem_id=? AND sample_id=?
        """, (self.problem_type, self.problem_id, self.sample_id))

        row = cur.fetchone()
        con.close()
        
        if row is None:
            raise ValueError(f"No loss found for problem_id {self.problem_id} and sample_id {self.sample_id}.")
        
        loss_filename = row[0]
        return np.load(os.path.join(loss_dir, loss_filename))

    def write_loss_filename(self, loss_filename: str, db_path="data/landscape-analysis.db"):
        """
        Write the loss filename to the database for this loss surface configuration.
        """
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        # Check if the loss surface configuration exists in the database
        cur.execute(f"""
            SELECT loss_filename FROM loss_surfaces 
            WHERE problem_type = ? AND problem_id = ? AND sample_id = ?
        """, (self.problem_type, self.problem_id, self.sample_id))
        existing_filename = cur.fetchone()[0]
        if existing_filename is not None and existing_filename != 'pending':
            con.close()
            raise ValueError(f"Loss surface {self} already has a loss filename in the database.")

        # Update the loss filename in the database
        cur.execute(f"""
            UPDATE loss_surfaces 
            SET loss_filename = ? 
            WHERE problem_type = ? AND problem_id = ? AND sample_id = ?
        """, (loss_filename, self.problem_type, self.problem_id, self.sample_id))


        con.commit()
        con.close()

    def get_problem_config(self, db_path="data/landscape-analysis.db"):
        if self.problem_type == 'nn':
            result = NNProblemConfig.lookup_by_id(self.problem_id, db_path)
        elif self.problem_type == 'bbob':
            result = BBOBProblemConfig.lookup_by_id(self.problem_id, db_path)
        else:
            raise ValueError(f"Unknown problem type: {self.problem_type}")

        return result

def assign_samples_to_problem(problem_config, sample_config: ParameterSampleConfig, up_to_nb_versions: int, sample_dir = 'data/samples', db_path='data/landscape-analysis.db'):
    problem_id = problem_config.get_id(db_path) # raise: ValueError if not found
    problem_type = problem_config.get_type()
    
    # make sure enough samples are in samples table
    nb_versions_in_samples = sample_config.get_nb_versions(db_path)    
    if nb_versions_in_samples < up_to_nb_versions:
        sample_config.add_samples(up_to_nb_versions - nb_versions_in_samples, sample_dir, db_path)
    
    for version in range(up_to_nb_versions):
        sample_id = sample_config.get_id(version, db_path)
        try:
            LossSurfaceConfig(problem_type, problem_id, sample_id).write_to_db(db_path)
        except sqlite3.IntegrityError:
            print(f"Loss surface for problem {problem_id} with sample {sample_id} already exists in the database.")
            continue  
################################### Calculate Loss ###################################
import os, uuid, sqlite3
import numpy as np
from torch.nn.functional import cross_entropy
from src.RandmanFunctions import split_and_load
from src.LandscapeAnalysis import get_parameter_to_loss_fn
from src.Models import RandmanSNN

def calculate_and_save_loss(loss_surface_id: int, 
                            sample_dir='data/samples',
                            randman_dir='data/randman',
                            loss_dir='data/losses',
                            db_path='data/landscape-analysis.db',
                            device='cuda'):
    loss_surface_config = LossSurfaceConfig.lookup_by_id(loss_surface_id, db_path)
    problem_config = loss_surface_config.get_problem_config(db_path)
    f = problem_config.get_loss_surface_fn(db_path, os.path.join(randman_dir, 'meta-data.csv'), device)
    samples = loss_surface_config.read_sample(sample_dir, db_path)

    # The computation part
    print(f"calculating loss for loss_surface_id {loss_surface_id} with {len(samples)} samples")
    loss = np.apply_along_axis(f, 1, samples)

    # Save loss to the database
    loss_filename = f"{uuid.uuid4().hex}.npy"
    loss_surface_config.write_loss_filename(loss_filename, db_path)

    # Save loss to file
    os.makedirs(loss_dir, exist_ok=True)
    np.save(os.path.join(loss_dir, loss_filename), loss)

def get_next_available_id(column_name, db_path='data/landscape-analysis.db'):
    con = sqlite3.connect(db_path, timeout=30)
    cur = con.cursor()
    
    # Aquire writer lock
    cur.execute("BEGIN EXCLUSIVE")
    
    # Check if the column exists in the loss_surfaces table
    cur.execute("PRAGMA table_info(loss_surfaces)")
    columns = [row[1] for row in cur.fetchall()]
    if column_name not in columns:
        cur.execute(f"ALTER TABLE loss_surfaces ADD COLUMN {column_name} REAL")
        print(f"Column '{column_name}' added to 'loss_surfaces' table by get_next_available_id()")
    
    # Find the first row where the column is NULL
    if column_name == "loss_filename":
        cur.execute(f"""
            SELECT MIN(id) FROM loss_surfaces
            WHERE {column_name} IS NULL
            """)
    # if working on features, then loss_filename should not be NULL or pending
    else:
        cur.execute(f"""
            SELECT MIN(id) FROM loss_surfaces
            WHERE {column_name} IS NULL 
            AND loss_filename IS NOT NULL 
            AND loss_filename != 'pending'
            """)
    
    id = cur.fetchone()[0]
    
    # set pending status
    if id is not None:
        cur.execute(f"""
            UPDATE loss_surfaces 
            SET {column_name} = 'pending' 
            WHERE id = {id}
        """)
    
    con.commit()
    con.close()
    
    return id

############################################ Calculate Features ############################################
from typing import Callable
def get_xy_by_id(loss_surface_id: int, loss_dir="data/losses", sample_dir="data/samples", db_path="data/landscape-analysis.db"):
    """
    Get the x and y values for a given loss surface ID.
    """
    loss_surface_config = LossSurfaceConfig.lookup_by_id(loss_surface_id, db_path)
    x = loss_surface_config.read_sample(sample_dir)
    y = loss_surface_config.read_loss(loss_dir, db_path)
    return x, y  

def calculate_and_save_features(loss_surface_id: int, sample_to_feature: Callable, loss_dir="data/losses", sample_dir="data/samples", db_path="data/landscape-analysis.db"):
    x, y = get_xy_by_id(loss_surface_id, loss_dir, sample_dir, db_path)
    
    # calculate features
    print(f"Calculating features for loss_surface_id {loss_surface_id} with {len(x)} samples")
    features = sample_to_feature(x, y)
    
    # Replace all "." in features' keys with "_"
    features = {key.replace('.', '_'): value for key, value in features.items()}
    
    # Save the features to loss-surfaces.csv
    # Connect to the SQLite database
    con = sqlite3.connect(db_path, timeout=30)
    cur = con.cursor()
    
    # aquire write lock
    cur.execute("BEGIN EXCLUSIVE")

    # Check and add columns if they don't exist
    cur.execute(f"PRAGMA table_info(loss_surfaces)")
    columns = [info[1] for info in cur.fetchall()]
    # Add missing columns 
    for key in features.keys():
        if key not in columns:
            cur.execute(f"ALTER TABLE loss_surfaces ADD COLUMN {key} REAL")
            print(f"Added column {key} to loss_surfaces table by calculate_and_save_features()")

    # Update the database with the feature values
    cur.execute(
        f"UPDATE loss_surfaces SET {', '.join(f'{key} = ?' for key in features.keys())} WHERE id = ?",
        (*features.values(), loss_surface_id)
    )

    # Commit changes and close the connection
    con.commit()
    con.close()