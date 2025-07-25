################################ Generate Randman Problem ################################
import sqlite3
import pandas as pd

from dataclasses import dataclass
from src.Models import calc_randmansnn_parameters
from src.RandmanFunctions import RandmanConfig, generate_and_save_randman, match_config

@dataclass
class RandmanProblemConfig:
    randman_id: int
    nb_hidden: int
    loss_fn: str

    @classmethod
    def lookup_by_id(cls, id: int, db_path='data/landscape-analysis.db'):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute(f"SELECT randman_id, nb_hidden, loss_fn FROM problems WHERE id = {id}")
        row = cur.fetchone()
        con.close()
        return cls(*row) 
    
def generate_randman_problem(randman_config: RandmanConfig, nb_hidden=40, loss_fn = 'cross_entropy', db_path='data/landscape-analysis.db'):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Find matching randman configuration
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
    dim = calc_randmansnn_parameters(randman_config.nb_units, nb_hidden, randman_config.nb_classes)

    # Check if the problem already exists in the database
    cur.execute(f"""
        SELECT id FROM problems 
        WHERE randman_id = {randman_row["id"]} AND nb_hidden = {nb_hidden} AND loss_fn = '{loss_fn}'
    """)
    existing_problem = cur.fetchone()

    if existing_problem:
        con.close()
        raise ValueError(f"Problem with randman_id {randman_row['id']} and nb_hidden {nb_hidden} already exists.")

    # Insert the new problem into the database
    cur.execute(f"""
        INSERT INTO problems (randman_id, nb_hidden, loss_fn, dim) 
        VALUES ({int(randman_row["id"])}, {nb_hidden}, '{loss_fn}', {dim})
    """)

    con.commit()
    con.close()
    
##################################### Generate Parameter Samples #####################################
import uuid, os, sqlite3
import numpy as np
from dataclasses import dataclass
from pflacco.sampling import create_initial_sample

@dataclass
class ParameterSampleConfig:
    dim: int = 3
    nb_sample: int = 1024
    method: str = "sobol"
    lower_bound: float = 0.0
    upper_bound: float = 1.0
    
    @classmethod
    def lookup_by_id(cls, id: int, db_path='data/landscape-analysis.db'):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute(f"SELECT dim, nb_sample, method, lower_bound, upper_bound FROM samples WHERE id={id}")
        row = cur.fetchone()
        con.close()
        if row is None:
            raise ValueError(f"ID {id} not found in {db_path}.")
        return cls(*row)
        
    def read_samples(self, sample_dir="data/samples", db_path='data/landscape-analysis.db'):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        # Check if the sample configuration exists in the database
        cur.execute(f"""
            SELECT filename FROM samples 
            WHERE dim = {self.dim} AND nb_sample = {self.nb_sample} AND method = '{self.method}' 
            AND lower_bound = {self.lower_bound} AND upper_bound = {self.upper_bound}
        """)
        
        row = cur.fetchone()
        con.close()
        
        if row is None:
            raise ValueError("No dataset found with the specified parameters.")
        
        filename = row[0]
        filepath = os.path.join(sample_dir, filename)
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Dataset file not found at {filepath}")
        
        data = np.load(filepath)
        return data

def add_samples(sample_config: ParameterSampleConfig, nb_versions=30, sample_dir="data/samples", db_path='data/landscape-analysis.db'):
    os.makedirs(sample_dir, exist_ok=True)
    
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    
    # Check if the sample configuration already exists in the database
    cur.execute(f"""
        SELECT version FROM samples 
        WHERE dim = {sample_config.dim} AND nb_sample = {sample_config.nb_sample} 
        AND method = '{sample_config.method}' AND lower_bound = {sample_config.lower_bound} 
        AND upper_bound = {sample_config.upper_bound}
    """)
    existing_versions = cur.fetchall()
    min_version = 0 if existing_versions is None else int(np.max(existing_versions)) + 1
    
    filename_list = []
    sample_list = []
    for version in range(min_version, min_version + nb_versions):                 
        # Create sample
        sample = create_initial_sample(sample_config.dim, 
                                       n=sample_config.nb_sample, 
                                       lower_bound=sample_config.lower_bound, 
                                       upper_bound=sample_config.upper_bound, 
                                       sample_type=sample_config.method,
                                       seed=version)
        sample_list.append(sample)
        
        # name the file
        filename = f"{uuid.uuid4().hex}.npy"
        filename_list.append(filename)
        
        # Insert the new sample configuration into the database
        cur.execute(f"""
            INSERT INTO samples (dim, nb_sample, method, lower_bound, upper_bound, version, filename) 
            VALUES ({sample_config.dim}, {sample_config.nb_sample}, '{sample_config.method}', 
            {sample_config.lower_bound}, {sample_config.upper_bound}, {version}, '{filename}')
        """)
    
    con.commit()
    con.close()
    
    # save after closing the database connection 
    for filename, sample in zip(filename_list, sample_list):
        filepath = os.path.join(sample_dir, filename)
        np.save(filepath, sample)
    
#################################### Match problem and sample ####################################
import os, sqlite3, numpy as np
from dataclasses import dataclass

@dataclass
class LossSurfaceConfig:
    problem_id: int
    sample_id: int

    @classmethod
    def lookup_by_id(cls, id: int, db_path="data/landscape-analysis.db"):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        cur.execute("SELECT problem_id, sample_id FROM loss_surfaces WHERE id=?", (id,))
        row = cur.fetchone()
        con.close()
        if row is None:
            raise ValueError(f"ID {id} not found in {db_path}.")
        return cls(*row)

    def read_sample(self, sample_dir="data/samples", db_path="data/landscape-analysis.db"):
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        cur.execute(f"""
            SELECT samples.filename FROM samples 
            INNER JOIN loss_surfaces ON samples.id=loss_surfaces.sample_id
            WHERE loss_surfaces.problem_id={self.problem_id} AND loss_surfaces.sample_id={self.sample_id}
            """)
        
        filename = cur.fetchone()
        if filename is None:
            raise ValueError(f"No sample found for problem_id {self.problem_id} and sample_id {self.sample_id}.")
        filename = filename[0]
        
        con.close()
        sample = np.load(os.path.join(sample_dir, filename))
        return sample
    
    def read_loss(self, loss_dir="data/losses", db_path="data/landscape-analysis.db"):
        """
        Read the loss associated with this loss surface configuration using the database.
        """
        con = sqlite3.connect(db_path)
        cur = con.cursor()
        
        cur.execute(f"""
            SELECT loss_filename FROM loss_surfaces 
            WHERE problem_id={self.problem_id} AND sample_id={self.sample_id}
        """)
        
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
        
        cur.execute(f"""
            UPDATE loss_surfaces 
            SET loss_filename = '{loss_filename}' 
            WHERE problem_id = {self.problem_id} AND sample_id = {self.sample_id}
        """)
        
        con.commit()
        con.close()

def assign_samples_to_problem(problem_id: int, sample_config: ParameterSampleConfig, nb_versions=None, sample_dir="data/samples", db_path="data/landscape-analysis.db"):    
    # Connect to the database
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Find all the samples which match the sample_config
    cur.execute(f"""
        SELECT samples.id FROM samples 
        WHERE dim = {sample_config.dim} AND nb_sample = {sample_config.nb_sample} 
        AND method = '{sample_config.method}' AND lower_bound = {sample_config.lower_bound} 
        AND upper_bound = {sample_config.upper_bound}
        AND samples.id NOT IN (SELECT sample_id FROM loss_surfaces WHERE problem_id = {problem_id})
    """)
    sample_ids = [row[0] for row in cur.fetchall()]
    
    # Make sure the samples exist
    if len(sample_ids) == 0:
        con.close()
        raise ValueError(f"No unassigned samples found matching the config {sample_config}. Please generate samples first.")

    # Add the matched samples, along with problem_id, to the loss_surfaces table

    # Insert new rows into the loss_surfaces table
    new_rows = [(problem_id, sample_id) for sample_id in sample_ids]
    cur.executemany("""
        INSERT INTO loss_surfaces (problem_id, sample_id) 
        VALUES (?, ?)
    """, new_rows)
    
    # print total number of rows inserted, and current number of rows for the problem
    cur.execute(f"SELECT COUNT(*) FROM loss_surfaces WHERE problem_id = {problem_id}")
    total_samples = cur.fetchone()[0]
    print(f"Problem {problem_id}: Assigned {len(new_rows)} samples. Total samples: {total_samples}")

    # Commit and close the connection
    con.commit()
    con.close()
    
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
    loss_surface = LossSurfaceConfig.lookup_by_id(loss_surface_id, db_path)
    problem = RandmanProblemConfig.lookup_by_id(loss_surface.problem_id, db_path)
    randman = RandmanConfig.lookup_by_id(problem.randman_id, os.path.join(randman_dir, "meta-data.csv"))

    # Prepare data and model
    train_loader, _ = split_and_load(randman.read_dataset(randman_dir), batch_size=516)
    model = RandmanSNN(randman.nb_units, problem.nb_hidden, randman.nb_classes, learn_beta=False, beta=0.95)
    if problem.loss_fn == 'cross_entropy':
        loss_fn = cross_entropy
    f = get_parameter_to_loss_fn(train_loader, model, loss_fn, device)
    samples = loss_surface.read_sample(sample_dir, db_path)
    
    # The computation part
    print(f"calculating loss for loss_surface_id {loss_surface_id} with {len(samples)} samples")
    loss = np.apply_along_axis(f, 1, samples)

    # Save loss to the database
    loss_filename = f"{uuid.uuid4().hex}.npy"
    loss_surface.write_loss_filename(loss_filename, db_path)

    # Save loss to file
    os.makedirs(loss_dir, exist_ok=True)
    np.save(os.path.join(loss_dir, loss_filename), loss)

def get_next_available_id(column_name, db_path='data/landscape-analysis.db'):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    
    # Aquire writer lock
    cur.execute("BEGIN EXCLUSIVE")
    
    # Check if the column exists in the loss_surfaces table
    cur.execute(f"PRAGMA table_info(loss_surfaces)")
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
    # if working on features, loss_filename should not be NULL or pending
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
    con = sqlite3.connect(db_path)
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