import os
import gc
import torch
import wandb
import math
import anndata
import numpy as np
import pandas as pd
import random
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import logging


from biomodalities.data.custom_datasets import AnnDataset
from biomodalities.data.custom_dataloaders import AnnLoader
from biomodalities.args import parse_config_and_args
from biomodalities.eval import LinearModel, WeightedKNNClassifier, OfflineVIZ, DecoderModel, TorchILISIMetric
from biomodalities.data.data_utils import balance_batches_for_ilisi, downsample_data
from biomodalities.eval.bmdb import aggregate, known_relationship_benchmark, pert_signal_consistency_benchmark, pert_signal_magnitude_benchmark
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    # Parse  arguments
    args = parse_config_and_args()

    logging.info("Arguments parsed successfully.")

    # Set seed for reproducibility
    set_seed(args.seed)

    logging.info(f"Seed set to {args.seed}")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('medium')
        logging.info("CUDA is available. Set float32 matrix multiplication precision to 'medium'.")

    # Construct a run name using the dataset name and obsm_key
    run_name = f"{args.run_name}_{args.dataset_name}_{args.obsm_key}_seed_{args.seed}_eval_{args.eval_method}"
    
    # Initialize wandb with a custom run name
    if args.debug :
        project_name = "debug_" + args.wandb_project_name
    else :
        project_name = args.wandb_project_name

    wandb.init(project=project_name, entity=args.wandb_entity, name=run_name, config=vars(args))
    wandb_logger = WandbLogger()
    logging.info(f"WandB initialized with run name: {run_name}")

    # convert args.control_label to boolean from str if value is "true" or "false"
    if args.control_label.lower() == "true":
        control_label = True
    elif args.control_label.lower() == "false":
        control_label = False

    if args.eval_method == "bmdb" :
        print("\nRunning bmdb evaluation Task\n")
        recall_thr_pairs = [(args.recall_threshold, 1 - args.recall_threshold)]
        adata = anndata.read_h5ad(args.bmdb_path)
        simulate_perfect_recall = False
        
        
        adata = adata[adata.obs[args.bmdb_ctrl_col] != args.control_label]
        metadata = adata.obs
        
        if args.obsm_key == "random" :
            emb = np.random.rand(adata.shape[0], 512)
        elif args.obsm_key == "fixed_random" :
            fixed_vector = np.random.rand(512)
            emb = np.tile(fixed_vector, (adata.shape[0], 1))
        elif args.obsm_key == "higher_bound" :
            simulate_perfect_recall = True
            emb = np.random.rand(adata.shape[0], 512)
        elif args.obsm_key == "random_scramble_PCA" :
            print("Extracting data")
            emb = adata.obsm["PCA"].copy()
            print("Shuffling data")
            np.random.shuffle(emb)
        else :
            emb = adata.obsm[args.obsm_key]
        print("aggregating data")
        map_data = aggregate(emb, metadata, pert_col=args.bmdb_pert_col, keys_to_remove=[])
        print("computing metrics")
        metrics = known_relationship_benchmark(map_data, recall_thr_pairs=recall_thr_pairs, 
                            pert_col=args.bmdb_pert_col, simulate_perfect_recall=simulate_perfect_recall)
        print("extracting metrics")
        result_col = f"recall_{recall_thr_pairs[0][0]}_{recall_thr_pairs[0][1]}"
        # metrics is a dataframe, extract a list of all results from the dataframe
        datasets = list(metrics['source'])
        results = list(metrics[result_col])

        # log results into wandb
        for dataset, result in zip(datasets, results):
            wandb.log({dataset: result})
        exit()

    if args.eval_method == "bmdb_precision" :
        print("\nRunning bmdb precision Task\n")
        print("Reading data from:", args.bmdb_path)
        adata = anndata.read_h5ad(args.bmdb_path)
        metadata = adata.obs
        quota_unexpressed = 0.15

        if args.dataset_name == "crispr_l1000":
            import scanpy as sc
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
            quota_unexpressed = 0.43

        # Convert adata.X to a DataFrame with gene names as columns
        expr_df = pd.DataFrame(adata.X, columns=adata.var.gene_name, index=adata.obs_names)
        print("Expression DataFrame created")

        # Calculate the sum of expression values for each gene
        gene_sums = expr_df.sum(axis=0)

        gene_sums = gene_sums / adata.shape[0]
        print("Sum of expression values for each gene calculated.")

        # Extract genes where the sum of expression values is less than 3
        unexpr_genes = list(gene_sums[gene_sums < quota_unexpressed].index)
        print("Unexpressed genes extracted. Count:", len(unexpr_genes))


        # For completeness, let's get the expressed genes where the sum is 3 or more
        expr_genes = list(gene_sums[gene_sums >= quota_unexpressed].index)
        print("Expressed genes extracted. Count:", len(expr_genes))

        expr_ind = metadata[args.bmdb_pert_col].isin(expr_genes + ['non-targeting'])


        if args.obsm_key == "random" :
            print("Generating random embeddings...")
            emb = np.random.rand(adata.shape[0], 512)
        elif args.obsm_key == "fixed_random" :
            print("Generating fixed random embeddings...")
            fixed_vector = np.random.rand(512)
            emb = np.tile(fixed_vector, (adata.shape[0], 1))
        elif args.obsm_key == "random_scramble_PCA" :
            emb = adata.obsm["PCA"]
            np.random.shuffle(emb)
        else :
            print("Using embeddings from adata.obsm with key:", args.obsm_key)
            emb = adata.obsm[args.obsm_key]

        print("Running pert_signal_consistency_benchmark...")
        cons_res = pert_signal_consistency_benchmark(emb, metadata, pert_col=args.bmdb_pert_col, neg_ctrl_perts=unexpr_genes, keys_to_drop=['non-targeting'])
        consistency_metric = round(sum(cons_res.pval <= 0.05) / sum(~pd.isna(cons_res.pval)) * 100, 1)
        consistency_gene_not_nan  = sum(~pd.isna(cons_res.pval))


        print("Running pert_signal_magnitude_benchmark...")
        magn_res = pert_signal_magnitude_benchmark(emb, metadata, pert_col=args.bmdb_pert_col, neg_ctrl_perts=unexpr_genes, control_key='non-targeting', keys_to_drop=[])

        magnitude_metric = round(sum(magn_res.pval <= 0.05) / sum(~pd.isna(magn_res.pval)) * 100, 1)
        magnitude_gene_not_nan  = sum(~pd.isna(magn_res.pval))





        # log results into wandb
        print("Logging results to wandb...")
        wandb.log({"consistency": consistency_metric,
                    "consistency_gene_not_nan":consistency_gene_not_nan, 
                    "magnitude": magnitude_metric, 
                    "magnitude_gene_not_nan":magnitude_gene_not_nan})

        exit()



    

    if args.eval_method in ['viz', 'all']:

        # Initialize dataset and data loader
        train_data_path = os.path.join("datasets", "train", f"{args.dataset_name}.h5ad")
        train_dataset = AnnDataset(train_data_path, chunk_size=args.chunk_size, control_key=args.control_key, control_label=control_label)
    
        if args.use_test2_as_test:
            test_data_path = os.path.join("datasets", "test2", f"{args.dataset_name}.h5ad")
        else:
            test_data_path = os.path.join("datasets", "test1", f"{args.dataset_name}.h5ad")
    
        test_dataset = AnnDataset(test_data_path, chunk_size=args.chunk_size, control_key=args.control_key, control_label=control_label)
    
        train_loader = AnnLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=args.shuffle,
            data_source=args.data_source,
            obsm_key=args.obsm_key
        )
    
        test_loader = AnnLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=args.shuffle,
            data_source=args.data_source,
            obsm_key=args.obsm_key
        )
    
        # Get shape of the first value of dataloader
        input_dim = next(iter(train_loader))[0].shape[1]
        print("input dimension :", input_dim)
    
        labels = train_dataset.get_labels(args.label_key)  
        print("number of train labels : ", len(labels))
    
        test_labels = test_dataset.get_labels(args.label_key)  
        print("number of test labels : ", len(test_labels))

        
        test_batches = test_dataset.get_labels(args.batch_key)  
        print("number of test batches : ", len(test_batches))
        print("\nRunning Embedding Viz Task\n")
        
        visualization = OfflineVIZ(color_palette="tab20")
        
        # Initialize lists to store embeddings and labels
        all_embeddings = []
        all_labels = []
        
        # Collect all embeddings and labels from the test loader
        for X_tensor, batch in test_loader:
            # Assuming X_tensor is a numpy array and batch[args.batch_key] is directly accessible
            all_embeddings.append(X_tensor.numpy())  # Convert tensor to numpy if needed
            all_labels.extend(batch[args.batch_key])  # Extend the list by labels
        
        # Concatenate all collected embeddings and convert labels to a numpy array
        embeddings = np.concatenate(all_embeddings, axis=0)
        labels = np.array(all_labels)
        
        # Generate visualizations
        fig_umap, fig_pca = visualization.plot(embeddings, labels)
        # log figures to wandb
        # Log figures to wandb
        wandb.log({"UMAP Visualization": wandb.Image(fig_umap, caption="UMAP Visualization")})
        wandb.log({"PCA Visualization": wandb.Image(fig_pca, caption="PCA Visualization")})

    



        


    

    
    if args.eval_method == "all" or args.eval_method == "knn" : 

        # Initialize dataset and data loader
        train_data_path = os.path.join("datasets", "train", f"{args.dataset_name}.h5ad")
        train_dataset = AnnDataset(train_data_path, chunk_size=args.chunk_size, control_key=None, control_label=control_label)
    
        if args.use_test2_as_test:
            test_data_path = os.path.join("datasets", "test2", f"{args.dataset_name}.h5ad")
        else:
            test_data_path = os.path.join("datasets", "test1", f"{args.dataset_name}.h5ad")
    
        test_dataset = AnnDataset(test_data_path, chunk_size=args.chunk_size, control_key=None, control_label=control_label)
    

    
        test_loader = AnnLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=args.shuffle,
            data_source=args.data_source,
            obsm_key=args.obsm_key
        )

    
        train_labels = train_dataset.get_labels(args.label_key)  
        print("number of train labels : ", len(train_labels))
    
        test_labels = test_dataset.get_labels(args.label_key)  
        print("number of test labels : ", len(test_labels))

        labels = list(set(train_labels) | set(test_labels))


        print("\nRunning KNN evaluation Task\n")
        train_nb_samples = len(train_dataset)

    
        if train_nb_samples > 100000:
            print(f"Original dataset size: {train_nb_samples}, performing downsampling...")
            train_dataset = downsample_data(train_dataset, args.label_key, args.seed)
            print("Downsampling done.")

        # Proceed to initialize the DataLoader with the now possibly modified dataset
        train_loader = AnnLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            data_source=args.data_source,
            obsm_key=args.obsm_key
        )



        if args.k_initial is None :
            args.k = int(math.sqrt(len(train_dataset)))
        else :
            args.k = args.k_initial
        if args.k % 2 == 0 :
            args.k += 1
        print("Number of k neighbors is ",args.k)

        # k-NN Evaluation
        knn = WeightedKNNClassifier(
            unique_labels=train_labels,
            k=args.k,
            T=args.T,
            max_distance_matrix_size=args.max_distance_matrix_size,
            distance_fx=args.distance_fx,
            epsilon=args.epsilon,
            use_pca=args.use_pca,
        )
        
        # Update the k-NN classifier with training data
        print("Getting train data")
        for X_tensor, batch in train_loader:
            knn.update(train_features=X_tensor, train_targets=batch[args.label_key])  

        # Update the k-NN classifier with test data
        print("Getting test data")
        for X_tensor, batch in test_loader:
            knn.update(test_features=X_tensor, test_targets=batch[args.label_key])  

        print("Computing knn")
        top1_acc, top5_acc = knn.compute()

        torch.cuda.empty_cache()
        del knn
        gc.collect()

        wandb.log({"Test k-NN Accuracy @1": top1_acc,"Test k-NN Accuracy @5": top5_acc})


    

    if args.eval_method == "all" or args.eval_method == "linear" :
        # Initialize dataset and data loader
        train_data_path = os.path.join("datasets", "train", f"{args.dataset_name}.h5ad")
        train_dataset = AnnDataset(train_data_path, chunk_size=args.chunk_size, control_key=args.control_key, control_label=control_label)
    
        if args.use_test2_as_test:
            test_data_path = os.path.join("datasets", "test2", f"{args.dataset_name}.h5ad")
        else:
            test_data_path = os.path.join("datasets", "test1", f"{args.dataset_name}.h5ad")
    
        test_dataset = AnnDataset(test_data_path, chunk_size=args.chunk_size, control_key=args.control_key, control_label=control_label)
    
        train_loader = AnnLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=args.shuffle,
            data_source=args.data_source,
            obsm_key=args.obsm_key
        )
    
        test_loader = AnnLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=args.shuffle,
            data_source=args.data_source,
            obsm_key=args.obsm_key
        )
    
        # Get shape of the first value of dataloader
        input_dim = next(iter(train_loader))[0].shape[1]
        print("input dimension :", input_dim)
    
        labels = train_dataset.get_labels(args.label_key)  
        print("number of train labels : ", len(labels))
    
        test_labels = test_dataset.get_labels(args.label_key)  
        print("number of test labels : ", len(test_labels))
        print("\nRunning Linear Probing evaluation Task\n")

        # Linear Evaluation
        linear_model = LinearModel(
            input_dim=input_dim,
            num_classes=len(labels),
            optimizer_name=args.optimizer_name,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            scheduler_name=args.scheduler_name,
            min_lr=args.min_lr,
            warmup_start_lr=args.warmup_start_lr,
            warmup_epochs=args.warmup_epochs,
            lr_decay_steps=args.lr_decay_steps,
            scheduler_interval=args.scheduler_interval,
            seed=args.seed,
            label_key=args.label_key,  
            unique_labels=labels,  
        )

        # Configure the Trainer with parsed utility arguments
        trainer = pl.Trainer(
            logger=wandb_logger,
            max_epochs=args.max_epochs,
            devices=args.gpus if args.accelerator == 'gpu' else 1,
            accelerator=args.accelerator,
            precision=args.precision,
            strategy=args.distributed_backend,
        )

        trainer.fit(linear_model, train_dataloaders=train_loader, val_dataloaders=test_loader)
        linear_results = trainer.test(linear_model, test_loader)
        wandb.log({"Linear Evaluation Results": linear_results})

    if args.eval_method in ['ilisi', 'all']:
        # Initialize dataset and data loader
        train_data_path = os.path.join("datasets", "train", f"{args.dataset_name}.h5ad")
        train_dataset = AnnDataset(train_data_path, chunk_size=args.chunk_size, control_key=None, control_label=control_label)
    
        if args.use_test2_as_test:
            test_data_path = os.path.join("datasets", "test2", f"{args.dataset_name}.h5ad")
        else:
            test_data_path = os.path.join("datasets", "test1", f"{args.dataset_name}.h5ad")
    
        test_dataset = AnnDataset(test_data_path, chunk_size=args.chunk_size, control_key=None, control_label=control_label)
    
        train_loader = AnnLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=args.shuffle,
            data_source=args.data_source,
            obsm_key=args.obsm_key
        )
    

    
        # Get shape of the first value of dataloader
        input_dim = next(iter(train_loader))[0].shape[1]
        print("input dimension :", input_dim)
    
        labels = train_dataset.get_labels(args.label_key)  
        print("number of train labels : ", len(labels))
    
        test_labels = test_dataset.get_labels(args.label_key)  
        print("number of test labels : ", len(test_labels))
        print("length of dataset : ", len(test_dataset))
        #test_dataset = balance_batches_for_ilisi(test_dataset,args.batch_key)
        #print("length of dataset after batch balancing : ", len(test_dataset))
        if len(test_dataset) > 300000:
            print(f"Original dataset size: {len(test_dataset)}, performing downsampling...")
            test_dataset = downsample_data(test_dataset, args.batch_key, args.seed,300000)
            print("Downsampling done.")
        test_loader = AnnLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=args.shuffle,
            data_source=args.data_source,
            obsm_key=args.obsm_key
        )



        test_batches = test_dataset.get_labels(args.batch_key)  
        print("number of test batches : ", len(test_batches))
        if args.k_initial is None :
            args.k = int(math.sqrt(len(test_dataset)))
        else :
            args.k = args.k_initial
        if args.k % 2 == 0 :
            args.k += 1
        print("Number of k neighbors is ",args.k)
        wandb.log({"nb neighbors ilisi": args.k})
        print("\nRunning ILISI evaluation Task\n")
        ilisi_metric = TorchILISIMetric(perplexity=args.k//3, unique_labels=test_batches, use_pca=args.use_pca)

        for X_tensor, batch in test_loader:
            ilisi_metric.update(X_tensor, batch[args.batch_key])

        normalized_ilisi_score = ilisi_metric.compute()
        print("Normalized ILISI Score:", normalized_ilisi_score.item())
        wandb.log({"Normalized batch ILISI Score": normalized_ilisi_score.item()})

        # Free up memory if needed
        torch.cuda.empty_cache()
        del ilisi_metric
        gc.collect()

    if args.eval_method == "reconstruct":
        logging.info("Starting reconstruction task...")

        train_data_path = os.path.join("datasets", "train", f"{args.dataset_name}.h5ad")
        train_dataset = AnnDataset(train_data_path, chunk_size=args.chunk_size, hvg=True)

        train_loader = AnnLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=args.shuffle,
            data_source=args.data_source,
            obsm_key=args.obsm_key,
            task=args.eval_method
        )
        logging.info("Train data loaderss for reconstruction task initialized.")

        test2_data_path = os.path.join("datasets", "test2", f"{args.dataset_name}.h5ad")

        test2_dataset = AnnDataset(test2_data_path, chunk_size=args.chunk_size, hvg=True)


        

        test2_loader = AnnLoader(
            dataset=test2_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=args.shuffle,
            data_source=args.data_source,
            obsm_key=args.obsm_key,
            task=args.eval_method
        )

        
        logging.info("Test data loaders for reconstruction task initialized.")

        # Initialize the model for reconstruction task
        embedding_dim = next(iter(train_loader))[0].shape[1]
        output_dim = next(iter(train_loader))[1][0].shape[1]  # Assuming the data is loaded as (data, obs)

        hidden_dims = [int(embedding_dim + (output_dim - embedding_dim) * i / (args.model_depth + 1)) for i in range(1, args.model_depth + 1)]
        model = DecoderModel(
            embedding_dim=embedding_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            optimizer_name=args.optimizer_name,
            scheduler_name=args.scheduler_name,
            lr=args.lr,
            weight_decay=args.weight_decay,
            loss_type='mse',  # TODO : Could be parameterized if different types are needed
            norm_type='log',  # TODO : This could also be parameterized
            batch_key=args.batch_key,
            control_key=args.control_key,
            control_label=control_label,
            scheduler_interval=args.scheduler_interval,
            warmup_start_lr=args.warmup_start_lr,
            min_lr=args.min_lr,
            warmup_epochs=args.warmup_epochs,
            lr_decay_steps=args.lr_decay_steps
        )
        logging.info("Decoder model initialized for reconstruction task.")

        trainer = pl.Trainer(
            logger=wandb_logger,
            max_epochs=args.max_epochs,
            devices=args.gpus if args.accelerator == 'gpu' else 1,
            accelerator=args.accelerator,
            precision=args.precision,
        )

        # Train the model
        trainer.fit(model, train_loader)
        logging.info("Model training completed.")



        logging.info("Testing on different perturbations and different batches dataset...")
        test2_results = trainer.test(model, test2_loader)
        logging.info("Test completed.")

        
        


        logging.info("Logging to Wandb...")
        #for metric_name, metric_value in test1_results[0].items():
        #    wandb.log({f'test1_{metric_name}': metric_value})

        for metric_name, metric_value in test2_results[0].items():
            wandb.log({f'test2_{metric_name}': metric_value})


    if args.eval_method == "all" : 
        # Summarize results
        summary = {
            "Linear Evaluation Results": linear_results,
            "k-NN Accuracy @1": top1_acc,
            "k-NN Accuracy @5": top5_acc,
        }
        for key, value in summary.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()