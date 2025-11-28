"""
OpenTargets Disease-Disease Network Preparation and Harmonization Script
=========================================================================
This script processes OpenTargets disease association data to create a disease-disease
similarity network based on shared genetic associations, then harmonizes it with ICD-10 codes.

Key steps:
1. Load OpenTargets disease-gene association data
2. Filter out literature-based associations
3. Apply custom weighting scheme to different data types
4. Calculate harmonic sum of associations
5. Compute disease-disease similarity based on shared genes
6. Map disease names to ICD-10 codes using semantic similarity
7. Harmonize the network with ICD-10 coding system
8. Save both raw and harmonized matrices

Author: Vladimir K.
Date: 2025-01-27
"""

import logging
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# Configure logging
def setup_logger(name: str, log_file: str) -> logging.Logger:
    """Setup logger to avoid duplicate handlers."""
    logger = logging.getLogger(name)

    # Only add handlers if logger doesn't have them
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

logger = setup_logger(__name__, 'prepare_open_targets.log')


class OpenTargetsProcessor:
    """Process OpenTargets disease-gene associations into disease-disease networks."""

    def __init__(self, data_dir: str, output_dir: str = "data/open_targets_25_09",
                 use_cache: bool = True):
        """
        Initialize the processor.

        Args:
            data_dir: Directory containing OpenTargets association data
            output_dir: Directory for output files
            use_cache: Whether to use cached intermediate results
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.use_cache = use_cache

        # Cache directory for intermediate results
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Weighting scheme for different datasources
        self.weights = {
            "europe_pm": 0.2,
            "expression_atlas": 0.2,
            "impc": 0.2,
            "progeny": 0.5,
            "slapenrich": 0.5,
            "cancer_biomarkers": 0.5,
            "sysbio": 0.5,
            "otar_projects": 0.5,
            "default": 1.0
        }

        logger.info(f"Initialized OpenTargetsProcessor with data_dir={data_dir}")
        logger.info(f"Cache enabled: {use_cache}, cache_dir: {self.cache_dir}")

    def get_weight(self, datasource: str) -> float:
        """Get weight for a datasource."""
        return self.weights.get(datasource, self.weights['default'])

    def load_associations(self) -> pd.DataFrame:
        """
        Load OpenTargets association data from parquet files.

        Returns:
            DataFrame with disease-gene associations

        Raises:
            FileNotFoundError: If data directory doesn't exist
            ValueError: If required columns are missing
        """
        start_time = time.time()
        logger.info("Loading OpenTargets association data...")

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        df = pd.read_parquet(self.data_dir, engine="pyarrow")

        # Validate required columns
        required_cols = ['datatypeId', 'datasourceId', 'diseaseId', 'targetId', 'score']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        elapsed = time.time() - start_time
        logger.info(f"Loaded {len(df):,} associations in {elapsed:.2f}s")
        logger.info(f"Data shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")

        return df

    def filter_literature(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove literature-based associations to focus on experimental evidence.

        Args:
            df: Input DataFrame

        Returns:
            Filtered DataFrame
        """
        start_time = time.time()
        initial_count = len(df)

        logger.info("Filtering out literature-based associations...")
        df_filtered = df[df['datatypeId'] != 'literature'].copy()

        removed = initial_count - len(df_filtered)
        elapsed = time.time() - start_time

        logger.info(f"Removed {removed:,} literature associations ({removed/initial_count*100:.1f}%)")
        logger.info(f"Remaining associations: {len(df_filtered):,}")
        logger.info(f"Data type distribution:\n{df_filtered['datatypeId'].value_counts()}")
        logger.info(f"Filtering completed in {elapsed:.2f}s")

        return df_filtered

    def calculate_weighted_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate weighted association scores using harmonic sum.

        Args:
            df: DataFrame with associations

        Returns:
            DataFrame with disease-gene custom association scores
        """
        start_time = time.time()
        logger.info("Calculating weighted association scores...")

        # Apply weights based on datasource
        df['weight'] = df['datasourceId'].apply(self.get_weight)
        df['weighted_term'] = 1 - (df['weight'] * df['score'])

        # Calculate harmonic sum per disease-gene pair
        def harmonic_sum(group):
            product_term = np.prod(group['weighted_term'])
            return 1 - product_term

        logger.info("Computing harmonic sum for disease-gene pairs...")
        final_scores = (df.groupby(['diseaseId', 'targetId'], group_keys=False)
                       .apply(harmonic_sum, include_groups=False)
                       .reset_index(name='customAssociationScore'))

        elapsed = time.time() - start_time
        logger.info(f"Calculated scores for {len(final_scores):,} disease-gene pairs")
        logger.info(f"Score statistics: min={final_scores['customAssociationScore'].min():.4f}, "
                   f"max={final_scores['customAssociationScore'].max():.4f}, "
                   f"mean={final_scores['customAssociationScore'].mean():.4f}")
        logger.info(f"Calculation completed in {elapsed:.2f}s")

        return final_scores

    def load_metadata(self) -> tuple:
        """
        Load disease and gene metadata.

        Returns:
            Tuple of (disease_df, gene_df)
        """
        start_time = time.time()
        logger.info("Loading metadata...")

        disease_df = pd.read_csv(self.output_dir / "disease_map.csv")
        disease_df = disease_df[["diseaseId", "name"]]
        logger.info(f"Loaded {len(disease_df):,} disease mappings")

        gene_df = pd.read_csv(self.output_dir / "gene_map.csv")
        gene_df = gene_df[["targetId", "approvedSymbol"]]
        logger.info(f"Loaded {len(gene_df):,} gene mappings")

        elapsed = time.time() - start_time
        logger.info(f"Metadata loaded in {elapsed:.2f}s")

        return disease_df, gene_df

    def create_disease_gene_matrix(self, scores_df: pd.DataFrame,
                                   disease_df: pd.DataFrame,
                                   gene_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create disease-gene association matrix.

        Args:
            scores_df: DataFrame with association scores
            disease_df: Disease metadata
            gene_df: Gene metadata

        Returns:
            Pivot table with diseases as rows and genes as columns
        """
        start_time = time.time()
        logger.info("Creating disease-gene matrix...")

        # Merge with metadata
        disease_edge = pd.merge(scores_df, disease_df, on="diseaseId")
        disease_edge = pd.merge(disease_edge, gene_df, on='targetId')
        disease_edge = disease_edge[["name", "approvedSymbol", "customAssociationScore"]]

        logger.info(f"Disease-gene edges: {len(disease_edge):,}")

        # Filter out measurement-related phenotypes
        measurement_keywords = [
            'measurement', 'level', 'concentration', 'trait',
            'response', 'ratio', 'presence', 'status',
            'microdeletion', 'microduplication', 'deletion', 'duplication'
        ]

        pattern = '|'.join(measurement_keywords)
        disease_edge_filtered = disease_edge[
            ~disease_edge['name'].str.contains(pattern, case=False, na=False)
        ]

        removed = len(disease_edge) - len(disease_edge_filtered)
        logger.info(f"Filtered out {removed:,} measurement-related associations")
        logger.info(f"Remaining associations: {len(disease_edge_filtered):,}")

        # Create pivot table
        logger.info("Creating pivot table (diseases × genes)...")
        disease_gene_matrix = disease_edge_filtered.pivot_table(
            index="approvedSymbol",
            columns="name",
            values="customAssociationScore",
            aggfunc="mean"
        ).fillna(0)

        elapsed = time.time() - start_time
        logger.info(f"Matrix shape: {disease_gene_matrix.shape} "
                   f"({disease_gene_matrix.shape[1]:,} diseases × {disease_gene_matrix.shape[0]:,} genes)")
        logger.info(f"Matrix creation completed in {elapsed:.2f}s")

        return disease_gene_matrix

    def compute_disease_disease_matrix(self, disease_gene_matrix: pd.DataFrame,
                                      force_recompute: bool = False) -> pd.DataFrame:
        """
        Compute disease-disease similarity based on shared genes.
        Uses mean of minimum association scores for shared genes.

        Args:
            disease_gene_matrix: Matrix with genes as rows, diseases as columns
            force_recompute: If True, recompute even if cache exists

        Returns:
            Disease-disease similarity matrix
        """
        cache_file = self.cache_dir / "disease_disease_matrix.pkl"
        checkpoint_file = self.cache_dir / "checkpoint.npz"

        # Try to load from cache
        if self.use_cache and cache_file.exists() and not force_recompute:
            logger.info(f"Loading cached disease-disease matrix from {cache_file}...")
            try:
                dis_dis_df = pd.read_pickle(cache_file)
                logger.info(f"✅ Loaded cached matrix: shape={dis_dis_df.shape}")
                return dis_dis_df
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Recomputing...")

        start_time = time.time()
        logger.info("Computing disease-disease similarity matrix...")

        # Transpose to get diseases as rows and genes as columns
        disease_matrix = disease_gene_matrix.T
        X = disease_matrix.values
        disease_names = disease_matrix.index.to_list()
        n_diseases = len(disease_names)

        logger.info(f"Computing similarities for {n_diseases:,} diseases...")
        logger.info(f"⏱️  Estimated time: ~{(n_diseases * n_diseases / 2) / 5000:.1f} minutes")

        # Initialize similarity matrix
        dis_dis_matrix = np.zeros((n_diseases, n_diseases), dtype=float)

        # Check for checkpoint
        start_i = 0
        if self.use_cache and checkpoint_file.exists() and not force_recompute:
            logger.info(f"Found checkpoint file. Resuming computation...")
            try:
                checkpoint = np.load(checkpoint_file)
                dis_dis_matrix = checkpoint['matrix']
                start_i = int(checkpoint['last_i']) + 1
                logger.info(f"✅ Resuming from row {start_i}/{n_diseases}")
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}. Starting from scratch...")

        # Compute upper triangle (symmetric matrix)
        checkpoint_interval = 100  # Save every 100 rows
        for i in tqdm(range(start_i, n_diseases), desc="Computing shared gene similarities"):
            for j in range(i, n_diseases):
                if i == j:
                    value = 0  # No self-loops
                else:
                    # Compute mean of minimum associations (shared strength)
                    shared_arr = np.minimum(X[i], X[j])
                    non_zero = shared_arr[shared_arr != 0]
                    # Suppress RuntimeWarning for mean of empty slice
                    with np.errstate(invalid='ignore'):
                        value = non_zero.mean() if len(non_zero) > 0 else 0
                    value = 0 if np.isnan(value) else value

                dis_dis_matrix[i, j] = value
                dis_dis_matrix[j, i] = value  # Symmetry

            # Save checkpoint periodically
            if self.use_cache and (i + 1) % checkpoint_interval == 0:
                np.savez_compressed(checkpoint_file, matrix=dis_dis_matrix, last_i=i)
                logger.info(f"💾 Checkpoint saved at row {i+1}/{n_diseases}")

        # Remove checkpoint file after completion
        if checkpoint_file.exists():
            checkpoint_file.unlink()
            logger.info("✅ Computation complete. Checkpoint removed.")

        # Convert to DataFrame
        dis_dis_df = pd.DataFrame(dis_dis_matrix,
                                  index=disease_names,
                                  columns=disease_names)
        np.fill_diagonal(dis_dis_df.values, 0)
        dis_dis_df = dis_dis_df.fillna(0)

        # Remove all-zero rows/columns
        dis_dis_df = dis_dis_df.loc[~(dis_dis_df == 0).all(axis=1)]
        dis_dis_df = dis_dis_df.loc[:, ~(dis_dis_df == 0).all(axis=0)]

        # Save to cache
        if self.use_cache:
            logger.info(f"💾 Saving to cache: {cache_file}")
            dis_dis_df.to_pickle(cache_file)
            logger.info(f"✅ Cache saved ({cache_file.stat().st_size / 1024 / 1024:.2f} MB)")

        # Calculate sparsity
        all_values = dis_dis_df.values.flatten()
        zero_percentage = (np.count_nonzero(all_values == 0) / len(all_values)) * 100

        elapsed = time.time() - start_time
        logger.info(f"Final matrix shape: {dis_dis_df.shape}")
        logger.info(f"Sparsity: {zero_percentage:.2f}% zeros")
        logger.info(f"Non-zero value statistics: "
                   f"min={dis_dis_df[dis_dis_df > 0].min().min():.4f}, "
                   f"max={dis_dis_df.max().max():.4f}, "
                   f"mean={dis_dis_df[dis_dis_df > 0].mean().mean():.4f}")
        logger.info(f"Disease-disease matrix computed in {elapsed/60:.2f} minutes")

        return dis_dis_df

    def save_results(self, dis_dis_df: pd.DataFrame, filename: str = "dis_dis_nolit_25_09.csv"):
        """
        Save disease-disease matrix to CSV.

        Args:
            dis_dis_df: Disease-disease similarity matrix
            filename: Output filename
        """
        start_time = time.time()
        output_path = self.output_dir / filename

        logger.info(f"Saving disease-disease matrix to {output_path}...")
        # Save adjacency matrix (row names = column names, so only header needed)
        dis_dis_df.to_csv(output_path, index=True)

        elapsed = time.time() - start_time
        logger.info(f"Saved to {output_path} in {elapsed:.2f}s")

        if output_path.exists():
            logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        else:
            logger.warning(f"File was not created: {output_path}")

    def run_pipeline(self, skip_to_harmonization: bool = False) -> pd.DataFrame:
        """
        Execute the complete processing pipeline.

        Args:
            skip_to_harmonization: If True, skip OpenTargets processing and load cached matrix

        Returns:
            Disease-disease similarity matrix
        """
        total_start = time.time()
        logger.info("="*60)
        logger.info("Starting OpenTargets processing pipeline")
        logger.info("="*60)

        # Check if we can skip to harmonization
        cache_file = self.cache_dir / "disease_disease_matrix.pkl"
        if skip_to_harmonization and cache_file.exists():
            logger.info("⚡ Fast mode: Loading cached disease-disease matrix...")
            try:
                dis_dis_df = pd.read_pickle(cache_file)
                logger.info(f"✅ Loaded from cache in seconds!")
                logger.info(f"Matrix shape: {dis_dis_df.shape}")
                return dis_dis_df
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Running full pipeline...")

        # Step 1: Load associations
        df = self.load_associations()

        # Step 2: Filter literature
        df_filtered = self.filter_literature(df)

        # Step 3: Calculate weighted scores
        final_scores = self.calculate_weighted_scores(df_filtered)

        # Step 4: Load metadata
        disease_df, gene_df = self.load_metadata()

        # Step 5: Create disease-gene matrix
        disease_gene_matrix = self.create_disease_gene_matrix(
            final_scores, disease_df, gene_df
        )

        # Step 6: Compute disease-disease similarities (with caching/checkpointing)
        dis_dis_df = self.compute_disease_disease_matrix(disease_gene_matrix)

        # Step 7: Save results
        self.save_results(dis_dis_df)

        total_elapsed = time.time() - total_start
        logger.info("="*60)
        logger.info(f"Pipeline completed successfully in {total_elapsed/60:.2f} minutes")
        logger.info("="*60)

        return dis_dis_df


class GraphHarmonizer:
    """Harmonize disease networks to ICD-10 coding system."""

    def __init__(self,
                 icd_file: str = "data/DiagAll_Eng.csv",
                 similarity_threshold: float = 0.5,
                 model_name: str = "NeuML/pubmedbert-base-embeddings"):
        """
        Initialize the harmonizer.

        Args:
            icd_file: Path to ICD-10 mapping file
            similarity_threshold: Minimum cosine similarity for mapping
            model_name: Sentence transformer model for semantic similarity
        """
        self.icd_file = Path(icd_file)
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name

        logger.info(f"Initializing GraphHarmonizer with model: {model_name}")
        logger.info(f"Similarity threshold: {similarity_threshold}")

        # Load ICD-10 data
        self.icd_dict, self.icd_descriptions, self.icd_codes = self._load_icd_data()

        # Load semantic similarity model
        self.model = self._load_model()

    def _load_icd_data(self) -> Tuple[Dict, list, list]:
        """
        Load ICD-10 disease descriptions and codes.

        Returns:
            Tuple of (icd_dict, descriptions, codes)
        """
        start_time = time.time()
        logger.info(f"Loading ICD-10 data from {self.icd_file}...")

        df_icd = pd.read_csv(self.icd_file)

        # Special handling for dementia
        df_icd['ShortDescription'] = df_icd['ShortDescription'].replace(
            "Dementia in Alzheimer\'s disease", "Dementia"
        )

        df_icd = df_icd.drop_duplicates(subset=["ShortDescription"])
        icd_dict = dict(zip(df_icd["ShortDescription"], df_icd["Code"]))
        icd_descriptions = list(icd_dict.keys())
        icd_codes = list(icd_dict.values())

        elapsed = time.time() - start_time
        logger.info(f"Loaded {len(icd_descriptions):,} ICD-10 descriptions in {elapsed:.2f}s")

        return icd_dict, icd_descriptions, icd_codes

    def _load_model(self) -> SentenceTransformer:
        """Load sentence transformer model for semantic similarity."""
        start_time = time.time()
        logger.info(f"Loading sentence transformer model: {self.model_name}...")

        model = SentenceTransformer(self.model_name)

        elapsed = time.time() - start_time
        logger.info(f"Model loaded in {elapsed:.2f}s")

        return model

    def create_embeddings(self, diseases: list) -> Tuple[np.ndarray, Dict]:
        """
        Create embeddings for ICD-10 descriptions and diseases.

        Args:
            diseases: List of disease names to map

        Returns:
            Tuple of (icd_embeddings, disease_embeddings_dict)
        """
        start_time = time.time()

        logger.info("Creating ICD-10 embeddings...")
        icd_embeddings = self.model.encode(self.icd_descriptions, convert_to_tensor=False)
        logger.info(f"Created {len(icd_embeddings):,} ICD-10 embeddings")

        logger.info(f"Creating embeddings for {len(diseases):,} diseases...")
        disease_embeddings = {
            disease: self.model.encode(disease, convert_to_tensor=False)
            for disease in tqdm(diseases, desc="Encoding diseases")
        }

        elapsed = time.time() - start_time
        logger.info(f"Embeddings created in {elapsed:.2f}s")

        return icd_embeddings, disease_embeddings

    def map_disease(self,
                   disease_name: str,
                   disease_embedding: np.ndarray,
                   icd_embeddings: np.ndarray) -> Tuple[str, float]:
        """
        Map a disease name to ICD-10 description using semantic similarity.

        Args:
            disease_name: Disease name to map
            disease_embedding: Pre-computed embedding
            icd_embeddings: Pre-computed ICD embeddings

        Returns:
            Tuple of (matched_description, similarity_score)
        """
        # Check for exact match first
        if disease_name in self.icd_dict:
            return disease_name, 1.0

        # Use semantic similarity
        disease_tensor = torch.tensor(disease_embedding)
        icd_tensor = torch.tensor(icd_embeddings)

        similarities = util.pytorch_cos_sim(disease_tensor, icd_tensor)[0]
        best_match_idx = similarities.argmax().item()
        best_similarity = similarities[best_match_idx].item()

        if best_similarity > self.similarity_threshold:
            return self.icd_descriptions[best_match_idx], best_similarity
        else:
            return "UNKNOWN", best_similarity

    def map_all_diseases(self,
                        diseases: list,
                        icd_embeddings: np.ndarray,
                        disease_embeddings: Dict) -> Dict:
        """
        Map all diseases to ICD-10.

        Args:
            diseases: List of disease names
            icd_embeddings: Pre-computed ICD embeddings
            disease_embeddings: Pre-computed disease embeddings

        Returns:
            Dictionary mapping disease names to ICD descriptions
        """
        start_time = time.time()
        logger.info(f"Mapping {len(diseases):,} diseases to ICD-10...")

        diseases_mapping = {}
        similarity_scores = {}

        for disease in tqdm(diseases, desc="Mapping disease names to ICD-10"):
            matched_disease, similarity = self.map_disease(
                disease,
                disease_embeddings[disease],
                icd_embeddings
            )
            diseases_mapping[disease] = matched_disease
            similarity_scores[disease] = similarity

        # Count results
        unknown_count = sum(1 for v in diseases_mapping.values() if v == "UNKNOWN")
        valid_count = len(diseases_mapping) - unknown_count

        elapsed = time.time() - start_time
        logger.info(f"Mapping completed in {elapsed:.2f}s")
        logger.info(f"Valid mappings: {valid_count:,} ({valid_count/len(diseases)*100:.1f}%)")
        logger.info(f"Unknown mappings: {unknown_count:,} ({unknown_count/len(diseases)*100:.1f}%)")

        # Log low-confidence mappings
        low_confidence = {k: (v, similarity_scores[k])
                         for k, v in diseases_mapping.items()
                         if v != "UNKNOWN" and similarity_scores[k] < 0.7}
        if low_confidence:
            logger.warning(f"Found {len(low_confidence)} low-confidence mappings (similarity < 0.7)")
            for orig, (mapped, sim) in list(low_confidence.items())[:10]:
                logger.warning(f"  {orig} -> {mapped} (similarity: {sim:.3f})")

        return diseases_mapping

    def harmonize_matrix(self,
                        matrix: pd.DataFrame,
                        diseases_mapping: Dict) -> pd.DataFrame:
        """
        Harmonize disease-disease matrix by applying ICD-10 mapping.

        Args:
            matrix: Original disease-disease matrix
            diseases_mapping: Mapping from disease names to ICD descriptions

        Returns:
            Harmonized matrix with ICD-10 disease names
        """
        start_time = time.time()
        logger.info("Harmonizing matrix with ICD-10 mappings...")

        # Filter out UNKNOWN mappings
        valid_diseases = {k: v for k, v in diseases_mapping.items() if v != 'UNKNOWN'}
        valid_keys = set(valid_diseases.keys()) & set(matrix.columns)

        logger.info(f"Selecting {len(valid_keys):,} valid diseases from matrix...")
        filtered_matrix = matrix.loc[list(valid_keys), list(valid_keys)]

        # Rename using mapping
        rename_mapping = {k: v for k, v in valid_diseases.items()
                         if k in filtered_matrix.columns}
        filtered_matrix = filtered_matrix.rename(index=rename_mapping, columns=rename_mapping)

        logger.info(f"Matrix shape after renaming: {filtered_matrix.shape}")

        # Handle duplicate column/row names (diseases mapped to same ICD code)
        duplicated_columns = filtered_matrix.columns[filtered_matrix.columns.duplicated()]

        if len(duplicated_columns) > 0:
            logger.info(f"Found {len(duplicated_columns.unique())} duplicate disease mappings")
            logger.info("Merging duplicates by taking maximum similarity...")

            # Process all duplicates: aggregate by taking max for each unique name
            # Group by column names and take max
            filtered_matrix = filtered_matrix.groupby(level=0, axis=1).max()
            # Group by row names and take max
            filtered_matrix = filtered_matrix.groupby(level=0, axis=0).max()

            logger.info(f"Matrix shape after merging duplicates: {filtered_matrix.shape}")

        elapsed = time.time() - start_time
        logger.info(f"Harmonization completed in {elapsed:.2f}s")

        return filtered_matrix

    def merge_dementia_diseases(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Merge dementia-related diseases into a unified representation.

        Args:
            matrix: Harmonized matrix

        Returns:
            Matrix with merged dementia diseases
        """
        start_time = time.time()
        logger.info("Merging dementia-related diseases...")

        # Rename Dementia to match ICD-10 (only if it exists)
        if 'Dementia' in matrix.index:
            matrix.rename(
                index={'Dementia': "Dementia in Alzheimer's disease"},
                columns={'Dementia': "Dementia in Alzheimer's disease"},
                inplace=True
            )
        if 'Dementia' in matrix.columns:
            matrix.rename(
                columns={'Dementia': "Dementia in Alzheimer's disease"},
                inplace=True
            )

        # Define dementia umbrella terms
        target_labels = [
            "Dementia in Alzheimer's disease",
            'Vascular dementia',
            "Alzheimer's disease"
        ]

        # Check which labels exist in matrix
        existing_labels = [label for label in target_labels if label in matrix.index]

        if len(existing_labels) > 1:
            logger.info(f"Merging {len(existing_labels)} dementia-related diseases:")
            for label in existing_labels:
                logger.info(f"  - {label}")

            # Take maximum across dementia diseases
            row_max = matrix.loc[existing_labels].max(axis=0).astype(float)

            # Update unified dementia row/column
            matrix.loc["Dementia in Alzheimer's disease"] = row_max
            matrix["Dementia in Alzheimer's disease"] = row_max

            logger.info("Dementia diseases merged successfully")
        else:
            logger.warning(f"Only found {len(existing_labels)} dementia labels, skipping merge")

        elapsed = time.time() - start_time
        logger.info(f"Dementia merging completed in {elapsed:.2f}s")

        return matrix

    def convert_to_icd_codes(self, matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Convert disease names to ICD-10 codes.

        Args:
            matrix: Matrix with disease names

        Returns:
            Matrix with ICD-10 codes as column/index names
        """
        start_time = time.time()
        logger.info("Converting disease names to ICD-10 codes...")

        # Create reverse mapping
        code_mapping = {name: code for name, code in self.icd_dict.items()
                       if name in matrix.columns}

        missing = set(matrix.columns) - set(code_mapping.keys())
        if missing:
            logger.warning(f"Could not find ICD codes for {len(missing)} diseases:")
            for disease in list(missing)[:10]:
                logger.warning(f"  - {disease}")

        # Convert columns to codes
        matrix_coded = matrix.copy()
        matrix_coded.columns = [code_mapping.get(x, x) for x in matrix_coded.columns]

        elapsed = time.time() - start_time
        logger.info(f"Conversion completed in {elapsed:.2f}s")

        return matrix_coded

    def analyze_matrix(self, matrix: pd.DataFrame, name: str = "Matrix"):
        """
        Log statistics about the matrix.

        Args:
            matrix: Matrix to analyze
            name: Name for logging
        """
        logger.info(f"--- {name} Statistics ---")
        logger.info(f"Shape: {matrix.shape}")

        all_values = matrix.values.flatten()
        total_values = len(all_values)
        zero_count = np.count_nonzero(all_values == 0)
        non_zero_values = all_values[all_values != 0]

        zero_percentage = (zero_count / total_values) * 100
        logger.info(f"Sparsity: {zero_percentage:.2f}% zeros")

        if len(non_zero_values) > 0:
            logger.info(f"Non-zero statistics: "
                       f"min={non_zero_values.min():.4f}, "
                       f"max={non_zero_values.max():.4f}, "
                       f"mean={non_zero_values.mean():.4f}, "
                       f"median={np.median(non_zero_values):.4f}")

    def save_harmonized_matrices(self,
                                 matrix_names: pd.DataFrame,
                                 matrix_codes: pd.DataFrame,
                                 output_dir: str = "."):
        """
        Save harmonized matrices.

        Args:
            matrix_names: Matrix with disease names
            matrix_codes: Matrix with ICD-10 codes
            output_dir: Output directory
        """
        start_time = time.time()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save harmonized matrices (adjacency matrices with row names = column names)
        names_file = output_path / "harmonized_open_target.csv"
        logger.info(f"Saving harmonized matrix (names) to {names_file}...")
        matrix_names.to_csv(names_file, index=True)
        logger.info(f"Saved: {names_file.stat().st_size / 1024:.2f} KB")

        codes_file = output_path / "harmonized_open_target_icd.csv"
        logger.info(f"Saving harmonized matrix (ICD codes) to {codes_file}...")
        matrix_codes.to_csv(codes_file, index=True)
        logger.info(f"Saved: {codes_file.stat().st_size / 1024:.2f} KB")

        elapsed = time.time() - start_time
        logger.info(f"Matrices saved in {elapsed:.2f}s")

    def run_harmonization(self,
                         disease_disease_matrix: pd.DataFrame,
                         output_dir: str = ".") -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute complete harmonization pipeline.

        Args:
            disease_disease_matrix: OpenTargets disease-disease matrix
            output_dir: Directory for output files

        Returns:
            Tuple of (harmonized_matrix_names, harmonized_matrix_codes)
        """
        total_start = time.time()
        logger.info("="*60)
        logger.info("Starting graph harmonization pipeline")
        logger.info("="*60)

        self.analyze_matrix(disease_disease_matrix, "Original OpenTargets Matrix")

        # Step 1: Create embeddings
        diseases = list(disease_disease_matrix.columns)
        icd_embeddings, disease_embeddings = self.create_embeddings(diseases)

        # Step 2: Map diseases to ICD-10
        diseases_mapping = self.map_all_diseases(diseases, icd_embeddings, disease_embeddings)

        # Step 3: Harmonize matrix
        harmonized_matrix = self.harmonize_matrix(disease_disease_matrix, diseases_mapping)
        self.analyze_matrix(harmonized_matrix, "Harmonized Matrix (before dementia merge)")

        # Step 4: Merge dementia diseases
        harmonized_matrix = self.merge_dementia_diseases(harmonized_matrix)
        self.analyze_matrix(harmonized_matrix, "Harmonized Matrix (after dementia merge)")

        # Step 5: Convert to ICD codes
        harmonized_matrix_coded = self.convert_to_icd_codes(harmonized_matrix)
        self.analyze_matrix(harmonized_matrix_coded, "Final Harmonized Matrix (ICD codes)")

        # Step 6: Save results
        self.save_harmonized_matrices(harmonized_matrix, harmonized_matrix_coded, output_dir)

        total_elapsed = time.time() - total_start
        logger.info("="*60)
        logger.info(f"Harmonization completed successfully in {total_elapsed/60:.2f} minutes")
        logger.info("="*60)

        return harmonized_matrix, harmonized_matrix_coded


def main(use_cache: bool = True, skip_to_harmonization: bool = False):
    """
    Main entry point.

    Args:
        use_cache: Enable caching and checkpointing (default: True)
        skip_to_harmonization: Skip OpenTargets processing if cache exists (default: False)
    """
    # Configuration
    DATA_DIR = "data/open_targets_25_09/association_type/"
    OUTPUT_DIR = "data/open_targets_25_09"
    ICD_FILE = "data/DiagAll_Eng.csv"

    logger.info("🚀 Starting with configuration:")
    logger.info(f"   - Cache enabled: {use_cache}")
    logger.info(f"   - Skip to harmonization: {skip_to_harmonization}")

    # Step 1: Process OpenTargets data
    processor = OpenTargetsProcessor(DATA_DIR, OUTPUT_DIR, use_cache=use_cache)
    disease_disease_matrix = processor.run_pipeline(skip_to_harmonization=skip_to_harmonization)

    # Step 2: Harmonize with ICD-10
    harmonizer = GraphHarmonizer(
        icd_file=ICD_FILE,
        similarity_threshold=0.5
    )
    harmonized_names, harmonized_codes = harmonizer.run_harmonization(
        disease_disease_matrix,
        OUTPUT_DIR
    )

    return disease_disease_matrix, harmonized_names, harmonized_codes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process OpenTargets data and harmonize with ICD-10",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # First run (compute everything, ~45-60 mins)
  python prepare_open_targets.py

  # Resume from checkpoint if interrupted
  python prepare_open_targets.py

  # Skip OpenTargets processing, use cached matrix (fast!)
  python prepare_open_targets.py --skip-to-harmonization

  # Force recompute (ignore cache)
  python prepare_open_targets.py --no-cache
        """
    )

    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching (force recompute everything)'
    )

    parser.add_argument(
        '--skip-to-harmonization',
        action='store_true',
        help='Skip OpenTargets processing and load cached matrix (only run harmonization)'
    )

    args = parser.parse_args()

    main(
        use_cache=not args.no_cache,
        skip_to_harmonization=args.skip_to_harmonization
    )
