import anndata
import numpy as np
import os

def train_test1_test2_split(adata, batch_column, perturb_column, control_column, control_key, train_size=0.7, random_state=None):
    """
    Splits an AnnData object into training, test1, and test2 datasets based on distinct batch and perturbation criteria, 
    ensuring exclusivity across the splits. The training set is formed by selecting a defined proportion of batches and 
    perturbations. Test1 contains data from the remaining batches and uses exactly the same perturbations as the training set.
    Test2 uses the same batches as Test1 but includes perturbations not used in either the training set or Test1.
    Each sample is exclusively assigned to only one of the datasets.

    Parameters:
    adata (anndata.AnnData): The AnnData object to be split.
    batch_column (str): Column name in adata.obs for batch differentiation.
    perturb_column (str): Column name in adata.obs for perturbation differentiation.
    control_column
    control_key
    train_size (float): Proportion of batches and perturbations allocated to the training set.
    random_state (int, optional): Seed for the random number generator to ensure reproducibility.

    Returns:
    tuple of anndata.AnnData: (train_data, test1_data, test2_data), representing the split AnnData objects.
    """
    np.random.seed(random_state)
    adata.obs_names_make_unique()
    print("Separating control")
    # put in separate variable a copy of control data
    control_data = adata[adata.obs[control_column] == control_key].copy()

    # Remove control data from adata
    print("Remove control data from adata")
    adata = adata[adata.obs[control_column] != control_key].copy()

    
    
    # Shuffle batches and perturbations
    print("Shuffle batches and perturbations")
    batches = np.random.permutation(adata.obs[batch_column].unique())
    perturbations = np.random.permutation(adata.obs[perturb_column].unique())

    
    num_train_batches = int(len(batches) * train_size)
    train_batches = set(batches[:num_train_batches])
    test_batches = set(batches[num_train_batches:])
    print("Ensure test1 uses exactly the same perturbations as train")
    num_train_perturbations = int(len(perturbations) * train_size)
    train_perturbations = set(perturbations[:num_train_perturbations])
    test1_perturbations = set(perturbations[:num_train_perturbations])  # Ensure test1 uses exactly the same perturbations as train
    remaining_perts = set(perturbations[num_train_perturbations:])
    print("Ensure Exclusive perturbations for test2")
    test2_perturbations = remaining_perts  # Exclusive perturbations for test2

    # Create boolean masks for each dataset
    print("Create boolean masks for each dataset")
    is_train = adata.obs[batch_column].isin(train_batches) & adata.obs[perturb_column].isin(train_perturbations)
    is_test1 = adata.obs[batch_column].isin(test_batches) & adata.obs[perturb_column].isin(test1_perturbations)
    is_test2 = adata.obs[batch_column].isin(test_batches) & adata.obs[perturb_column].isin(test2_perturbations)

    # Ensure exclusive assignment by removing overlaps
    # Priority: train > test1 > test2 
    print("Ensure exclusive assignment by removing overlaps")
    is_test1 = is_test1 & ~is_train
    is_test2 = is_test2 & ~is_train & ~is_test1

    # Split the data based on masks
    print("Split the data based on masks")
    train_data = adata[is_train].copy()
    test1_data = adata[is_test1].copy()
    test2_data = adata[is_test2].copy()

    # Filter train_data to remove perturbations not in test1
    print("Filter test_data1 to remove perturbations not in train")
    perturbations_in_train = set(train_data.obs[perturb_column].unique())
    test1_data = test1_data[test1_data.obs[perturb_column].isin(perturbations_in_train)].copy()

    test_batches_1 = test1_data.obs[batch_column].unique()
    test_batches_2 = test2_data.obs[batch_column].unique()
    train_batches = train_data.obs[batch_column].unique()

    # Add the subset of control data that shares the same batches of the set to each set 
    print("Add the subset of control data that shares the same batches of the set to each set")
    control_train = control_data[control_data.obs[batch_column].isin(train_batches)].copy()
    control_test1 = control_data[control_data.obs[batch_column].isin(test_batches_1)].copy()
    control_test2 = control_data[control_data.obs[batch_column].isin(test_batches_2)].copy()

    print("Making obs names unique")
    train_data.obs_names_make_unique()
    test1_data.obs_names_make_unique()
    test2_data.obs_names_make_unique()
    control_train.obs_names_make_unique()
    control_test1.obs_names_make_unique()
    control_test2.obs_names_make_unique()

    train_data.var_names_make_unique()
    control_train.var_names_make_unique()
    test1_data.var_names_make_unique()
    test2_data.var_names_make_unique()
    control_test1.var_names_make_unique()
    control_test2.var_names_make_unique()

    print("Merging all data")
    train_data = train_data.concatenate(control_train)
    test1_data = test1_data.concatenate(control_test1)
    test2_data = test2_data.concatenate(control_test2)

    return train_data, test1_data, test2_data




if __name__ == "__main__":
    print("Reading file")
    # Load your AnnData from a file
    save_root = "./datasets"
    file_path = "./datasets/eval/crispr_l1000.h5ad"
    adata = anndata.read_h5ad(file_path)
    print("file read")
    batch = "dataset_batch_num" # "dataset_batch_num"
    perturbation = "gene_name" # "ensembl_gene_id"

    control_column = "is_control"
    control_key = True


    # Split the data
    train_data, test1_data, test2_data = train_test1_test2_split(
        adata,
        batch_column=batch,
        perturb_column=perturbation,
        control_column=control_column,
        control_key=control_key,
        train_size=0.7,
        random_state=42
    )
    print(file_path.split("/")[-1])
    # Print stats of each dataset
    print("Train data: ",train_data.shape)
    print("Nb of unique batches in train : ",len(train_data.obs[batch].unique()))
    print("Nb of unique perturbations in train : ",len(train_data.obs[perturbation].unique()))
    print("Test1 data: ",test1_data.shape)
    print("Nb of unique batches in Test1 : ",len(test1_data.obs[batch].unique()))
    print("Nb of unique perturbations in Test1 : ",len(test1_data.obs[perturbation].unique()))
    print("Test2 data: ",test2_data.shape)
    print("Nb of unique batches in Test2 : ",len(test2_data.obs[batch].unique()))
    print("Nb of unique perturbations in Test2 : ",len(test2_data.obs[perturbation].unique()))

    # save all to disk
    train_path = os.path.join(save_root,"train",file_path.split("/")[-1])
    test1_path = os.path.join(save_root,"test1",file_path.split("/")[-1])
    test2_path = os.path.join(save_root,"test2",file_path.split("/")[-1])
    print(train_path)
    print(test1_path)
    print(test2_path)
    train_data.write_h5ad(train_path)
    test1_data.write_h5ad(test1_path)
    test2_data.write_h5ad(test2_path)