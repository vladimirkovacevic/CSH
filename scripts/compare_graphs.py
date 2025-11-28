"""
Graph Comparison Script
========================
This script compares harmonized OpenTargets disease-disease networks (genetic predisposition)
with clinical hospital graphs (co-occurrence patterns) to identify validated associations.

Key analyses:
1. Edge overlap: Which genetic associations also appear in clinical data?
2. Top-K analysis: For diseases with strong genetic associations, check clinical evidence
3. Statistical validation: Significance of overlap between genetic and clinical networks
4. Disease module analysis: Identify disease clusters in both networks
5. PubMed validation: Cross-reference with literature evidence

Author: Vladimir K.
Date: 2025-01-27
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob

import numpy as np
import pandas as pd
import pyreadr
import icd10
from scipy.stats import hypergeom, spearmanr, pearsonr
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
from tqdm import tqdm
import networkx as nx
try:
    import community as community_louvain
except ImportError:
    community_louvain = None
    import warnings
    warnings.warn("python-louvain not installed. Community detection will be skipped.")

# Configure logging
def setup_logger(name: str, log_file: str) -> logging.Logger:
    """Setup logger to avoid duplicate handlers."""
    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger

logger = setup_logger(__name__, 'compare_graphs.log')


class GraphComparator:
    """Compare genetic predisposition and clinical co-occurrence networks."""

    def __init__(self,
                 opentargets_file: str,
                 clinical_dir: str,
                 diagnosis_file: str,
                 icd_file: str = "data/DiagAll_Eng.csv"):
        """
        Initialize the comparator.

        Args:
            opentargets_file: Path to harmonized OpenTargets matrix (ICD codes)
            clinical_dir: Directory containing clinical graph .rds files
            diagnosis_file: Path to diagnosis metadata CSV
            icd_file: Path to ICD-10 mapping file
        """
        self.opentargets_file = Path(opentargets_file)
        self.clinical_dir = Path(clinical_dir)
        self.diagnosis_file = Path(diagnosis_file)
        self.icd_file = Path(icd_file)

        logger.info("Initializing GraphComparator")
        logger.info(f"OpenTargets file: {opentargets_file}")
        logger.info(f"Clinical graphs directory: {clinical_dir}")

        # Load data
        self.opentargets_matrix = self._load_opentargets()
        self.diagnosis_df = self._load_diagnosis_metadata()
        self.icd_dict = self._load_icd_mapping()
        self.clinical_files = self._discover_clinical_files()

    def _load_opentargets(self) -> pd.DataFrame:
        """Load harmonized OpenTargets matrix."""
        start_time = time.time()
        logger.info(f"Loading OpenTargets matrix from {self.opentargets_file}...")

        df = pd.read_csv(self.opentargets_file)
        df.index = df.columns

        elapsed = time.time() - start_time
        logger.info(f"Loaded OpenTargets matrix: shape={df.shape} in {elapsed:.2f}s")
        self._log_matrix_stats(df, "OpenTargets (Genetic)")

        return df

    def _load_diagnosis_metadata(self) -> pd.DataFrame:
        """Load diagnosis metadata."""
        start_time = time.time()
        logger.info(f"Loading diagnosis metadata from {self.diagnosis_file}...")

        df = pd.read_csv(self.diagnosis_file)
        df = df[df['age_id'] < 9].iloc[:1080, :]
        df['ID'] = range(len(df))
        df['English description'] = [
            icd10.find(x[0]).description if icd10.find(x[0]) else x[1]
            for x in df[['icd_code', 'descr']].values
        ]
        df['icd_global'] = df['icd_global'].str.replace('__', '-')

        elapsed = time.time() - start_time
        logger.info(f"Loaded {len(df)} diagnoses in {elapsed:.2f}s")

        return df

    def _load_icd_mapping(self) -> Dict:
        """Load ICD-10 code to description mapping."""
        start_time = time.time()
        logger.info(f"Loading ICD-10 mapping from {self.icd_file}...")

        df_icd = pd.read_csv(self.icd_file)
        df_icd = df_icd.drop_duplicates(subset=["ShortDescription"])
        icd_dict = dict(zip(df_icd["ShortDescription"], df_icd["Code"]))

        elapsed = time.time() - start_time
        logger.info(f"Loaded {len(icd_dict)} ICD mappings in {elapsed:.2f}s")

        return icd_dict

    def _discover_clinical_files(self) -> List[Path]:
        """Discover all clinical graph .rds files.

        Raises:
            FileNotFoundError: If clinical directory doesn't exist
            ValueError: If no clinical files found
        """
        if not self.clinical_dir.exists():
            raise FileNotFoundError(f"Clinical directory not found: {self.clinical_dir}")

        pattern = str(self.clinical_dir / "*.rds")
        files = sorted(glob.glob(pattern))

        if not files:
            raise ValueError(f"No .rds files found in {self.clinical_dir}")

        logger.info(f"Discovered {len(files)} clinical graph files")
        for f in files:
            logger.info(f"  - {Path(f).name}")

        return [Path(f) for f in files]

    def _log_matrix_stats(self, matrix: pd.DataFrame, name: str):
        """Log statistics about a matrix."""
        values = matrix.values.flatten()
        non_zero = values[values != 0]

        logger.info(f"--- {name} Statistics ---")
        logger.info(f"  Shape: {matrix.shape}")
        logger.info(f"  Sparsity: {(np.sum(values == 0) / len(values) * 100):.2f}% zeros")
        if len(non_zero) > 0:
            logger.info(f"  Non-zero: min={non_zero.min():.4f}, max={non_zero.max():.4f}, "
                       f"mean={non_zero.mean():.4f}, median={np.median(non_zero):.4f}")

    def get_disease_name(self, icd_code: str) -> str:
        """
        Get short disease name from ICD-10 code.

        Args:
            icd_code: ICD-10 code

        Returns:
            Short disease name (max 40 chars)
        """
        # Reverse lookup in ICD dict
        for name, code in self.icd_dict.items():
            if code == icd_code:
                # Truncate long names
                return name[:40] + "..." if len(name) > 40 else name
        return icd_code  # Fallback to code if not found

    def load_clinical_graph(self, rds_file: Path) -> pd.DataFrame:
        """
        Load a clinical graph from .rds file.

        Args:
            rds_file: Path to .rds file

        Returns:
            Clinical co-occurrence matrix

        Raises:
            ValueError: If RDS file is empty or invalid
        """
        start_time = time.time()
        logger.info(f"Loading clinical graph: {rds_file.name}...")

        try:
            result = pyreadr.read_r(rds_file)
        except Exception as e:
            raise ValueError(f"Failed to read {rds_file.name}: {e}")

        if not result or None not in result:
            raise ValueError(f"Empty or invalid RDS file: {rds_file.name}")

        df = result[None]

        if df.empty:
            raise ValueError(f"Empty dataframe in {rds_file.name}")

        df.fillna(0, inplace=True)

        # Validate dimensions
        if len(df) != len(self.diagnosis_df):
            logger.warning(f"Dimension mismatch in {rds_file.name}: "
                          f"expected {len(self.diagnosis_df)}, got {len(df)}")

        df.columns = self.diagnosis_df['icd_code'][:len(df.columns)]
        df.index = self.diagnosis_df['icd_code'][:len(df)]

        # Filter to diseases present in OpenTargets
        common_indices = df.index.intersection(self.opentargets_matrix.index)
        common_columns = df.columns.intersection(self.opentargets_matrix.columns)

        if len(common_indices) == 0 or len(common_columns) == 0:
            logger.warning(f"No common diseases between {rds_file.name} and OpenTargets")

        df = df.loc[common_indices, common_columns]

        elapsed = time.time() - start_time
        logger.info(f"Loaded {rds_file.name}: shape={df.shape} in {elapsed:.2f}s")

        return df

    def compute_edge_overlap(self,
                            genetic_matrix: pd.DataFrame,
                            clinical_matrix: pd.DataFrame,
                            threshold: float = 0.0) -> Dict:
        """
        Compute edge overlap between genetic and clinical networks.

        Args:
            genetic_matrix: OpenTargets genetic associations
            clinical_matrix: Clinical co-occurrence matrix
            threshold: Minimum edge weight to consider

        Returns:
            Dictionary with overlap statistics
        """
        start_time = time.time()
        logger.info("Computing edge overlap between genetic and clinical networks...")

        # Binarize matrices based on threshold
        genetic_binary = (genetic_matrix > threshold).astype(int)
        clinical_binary = (clinical_matrix > threshold).astype(int)

        # Ensure same shape
        common_nodes = sorted(set(genetic_matrix.index) & set(clinical_matrix.index))
        genetic_binary = genetic_binary.loc[common_nodes, common_nodes]
        clinical_binary = clinical_binary.loc[common_nodes, common_nodes]

        # Count edges
        genetic_edges = np.sum(genetic_binary.values) // 2  # Undirected
        clinical_edges = np.sum(clinical_binary.values) // 2
        overlap_edges = np.sum((genetic_binary.values * clinical_binary.values)) // 2

        # Jaccard similarity
        union_edges = genetic_edges + clinical_edges - overlap_edges
        jaccard = overlap_edges / union_edges if union_edges > 0 else 0

        # Hypergeometric test for significance
        n_possible_edges = (len(common_nodes) * (len(common_nodes) - 1)) // 2

        # Ensure valid hypergeometric parameters
        if overlap_edges > 0 and genetic_edges > 0 and clinical_edges > 0:
            try:
                p_value = hypergeom.sf(
                    overlap_edges - 1,
                    n_possible_edges,
                    genetic_edges,
                    clinical_edges
                )
            except (ValueError, ZeroDivisionError) as e:
                logger.warning(f"Hypergeometric test failed: {e}. Setting p_value=1.0")
                p_value = 1.0
        else:
            p_value = 1.0

        elapsed = time.time() - start_time

        results = {
            'n_common_nodes': len(common_nodes),
            'genetic_edges': genetic_edges,
            'clinical_edges': clinical_edges,
            'overlap_edges': overlap_edges,
            'jaccard_similarity': jaccard,
            'p_value': p_value,
            'overlap_percentage_genetic': overlap_edges / genetic_edges * 100 if genetic_edges > 0 else 0,
            'overlap_percentage_clinical': overlap_edges / clinical_edges * 100 if clinical_edges > 0 else 0
        }

        logger.info(f"Edge overlap computed in {elapsed:.2f}s:")
        logger.info(f"  Common nodes: {results['n_common_nodes']}")
        logger.info(f"  Genetic edges: {results['genetic_edges']:,}")
        logger.info(f"  Clinical edges: {results['clinical_edges']:,}")
        logger.info(f"  Overlapping edges: {results['overlap_edges']:,}")
        logger.info(f"  Jaccard similarity: {results['jaccard_similarity']:.4f}")
        logger.info(f"  Overlap % (genetic): {results['overlap_percentage_genetic']:.2f}%")
        logger.info(f"  Overlap % (clinical): {results['overlap_percentage_clinical']:.2f}%")
        logger.info(f"  Hypergeometric p-value: {results['p_value']:.2e}")

        return results

    def compute_correlation(self,
                           genetic_matrix: pd.DataFrame,
                           clinical_matrix: pd.DataFrame,
                           method: str = 'spearman') -> Tuple[float, float]:
        """
        Compute correlation between genetic and clinical edge weights.

        Args:
            genetic_matrix: OpenTargets genetic associations
            clinical_matrix: Clinical co-occurrence matrix
            method: 'spearman' or 'pearson'

        Returns:
            Tuple of (correlation, p_value)
        """
        start_time = time.time()
        logger.info(f"Computing {method} correlation between genetic and clinical edge weights...")

        # Align matrices
        common_nodes = sorted(set(genetic_matrix.index) & set(clinical_matrix.index))
        genetic_aligned = genetic_matrix.loc[common_nodes, common_nodes]
        clinical_aligned = clinical_matrix.loc[common_nodes, common_nodes]

        # Get upper triangle (exclude diagonal)
        triu_indices = np.triu_indices_from(genetic_aligned.values, k=1)
        genetic_values = genetic_aligned.values[triu_indices]
        clinical_values = clinical_aligned.values[triu_indices]

        # Check for valid data
        if len(genetic_values) < 2:
            logger.warning("Insufficient data for correlation (need at least 2 data points)")
            return 0.0, 1.0

        # Compute correlation
        try:
            if method == 'spearman':
                corr, p_value = spearmanr(genetic_values, clinical_values)
            elif method == 'pearson':
                corr, p_value = pearsonr(genetic_values, clinical_values)
            else:
                raise ValueError(f"Unknown method: {method}")

            # Handle NaN results
            if np.isnan(corr) or np.isnan(p_value):
                logger.warning(f"Correlation resulted in NaN, returning 0.0")
                return 0.0, 1.0

        except (ValueError, ZeroDivisionError) as e:
            logger.warning(f"Correlation calculation failed: {e}")
            return 0.0, 1.0

        elapsed = time.time() - start_time
        logger.info(f"{method.capitalize()} correlation: {corr:.4f} (p={p_value:.2e}) in {elapsed:.2f}s")

        return corr, p_value

    def analyze_top_k_neighbors(self,
                                disease_code: str,
                                disease_name: str,
                                k: int = 50) -> pd.DataFrame:
        """
        Analyze top-K genetic associations and their clinical evidence.

        Args:
            disease_code: ICD-10 code of disease of interest
            disease_name: Human-readable disease name
            k: Number of top associations to analyze

        Returns:
            DataFrame with comparison results
        """
        start_time = time.time()
        logger.info(f"Analyzing top-{k} genetic associations for {disease_name} ({disease_code})...")

        if disease_code not in self.opentargets_matrix.index:
            logger.warning(f"Disease {disease_code} not found in OpenTargets matrix")
            return pd.DataFrame()

        # Get top-K genetic associations
        genetic_row = self.opentargets_matrix.loc[disease_code]
        top_indices = np.argpartition(genetic_row.values, -k)[-k:]
        top_indices = top_indices[np.argsort(-genetic_row.values[top_indices])]
        top_diseases = genetic_row.index[top_indices].tolist()
        top_scores = genetic_row.values[top_indices]

        logger.info(f"Top-{k} genetic associations retrieved")

        # Compare with clinical graphs
        results = []
        for clinical_file in self.clinical_files:
            clinical_matrix = self.load_clinical_graph(clinical_file)

            if disease_code not in clinical_matrix.index:
                logger.warning(f"Disease {disease_code} not in {clinical_file.name}")
                continue

            clinical_row = clinical_matrix.loc[disease_code]

            # Check which top-K diseases have clinical evidence
            clinical_values = []
            for disease in top_diseases:
                if disease in clinical_row.index:
                    clinical_values.append(clinical_row[disease])
                else:
                    clinical_values.append(0)

            clinical_values = np.array(clinical_values)
            n_validated = np.sum(clinical_values > 0)

            results.append({
                'clinical_graph': clinical_file.stem.replace('All_OR_', ''),
                'n_validated': n_validated,
                'validation_rate': (n_validated / k * 100) if k > 0 else 0,
                'mean_clinical_strength': clinical_values[clinical_values > 0].mean() if n_validated > 0 else 0
            })

        results_df = pd.DataFrame(results)

        elapsed = time.time() - start_time
        logger.info(f"Top-K analysis completed in {elapsed:.2f}s")
        logger.info(f"Average validation rate: {results_df['validation_rate'].mean():.2f}%")

        return results_df

    def find_top_diseases_by_validation(self,
                                        clinical_matrices: List[pd.DataFrame],
                                        top_n: int = 20,
                                        min_genetic_edges: int = 10) -> pd.DataFrame:
        """
        Find diseases with most validated genetic associations across clinical graphs.

        Args:
            clinical_matrices: List of clinical co-occurrence matrices
            top_n: Number of top diseases to return
            min_genetic_edges: Minimum genetic associations required

        Returns:
            DataFrame with top diseases ranked by validation
        """
        start_time = time.time()
        logger.info(f"Finding top {top_n} diseases by validation rate...")

        disease_stats = []

        for disease_code in tqdm(self.opentargets_matrix.index, desc="Analyzing diseases"):
            # Get genetic associations for this disease
            genetic_row = self.opentargets_matrix.loc[disease_code]
            n_genetic = np.sum(genetic_row > 0)

            if n_genetic < min_genetic_edges:
                continue

            # Count validation across all clinical graphs
            total_validated = 0
            total_graphs_with_disease = 0

            for clinical_matrix in clinical_matrices:
                if disease_code not in clinical_matrix.index:
                    continue

                total_graphs_with_disease += 1
                clinical_row = clinical_matrix.loc[disease_code]

                # Find overlap
                genetic_diseases = set(genetic_row[genetic_row > 0].index)
                clinical_diseases = set(clinical_row[clinical_row > 0].index)
                validated = len(genetic_diseases & clinical_diseases)

                total_validated += validated

            if total_graphs_with_disease == 0:
                continue

            avg_validated = total_validated / total_graphs_with_disease
            validation_rate = (avg_validated / n_genetic * 100) if n_genetic > 0 else 0

            disease_stats.append({
                'icd_code': disease_code,
                'disease_name': self.get_disease_name(disease_code),
                'n_genetic_associations': n_genetic,
                'avg_validated_per_graph': avg_validated,
                'validation_rate': validation_rate,
                'n_clinical_graphs': total_graphs_with_disease
            })

        results_df = pd.DataFrame(disease_stats)
        results_df = results_df.sort_values('avg_validated_per_graph', ascending=False).head(top_n)

        elapsed = time.time() - start_time
        logger.info(f"Found top {len(results_df)} diseases in {elapsed:.2f}s")

        return results_df

    def find_validated_disease_pairs(self,
                                     clinical_matrix: pd.DataFrame,
                                     threshold_genetic: float = 0.3,
                                     threshold_clinical: float = 0.0) -> pd.DataFrame:
        """
        Find disease pairs that have both genetic predisposition and clinical evidence.

        Args:
            clinical_matrix: Clinical co-occurrence matrix
            threshold_genetic: Minimum genetic association score
            threshold_clinical: Minimum clinical co-occurrence score

        Returns:
            DataFrame with validated disease pairs
        """
        start_time = time.time()
        logger.info("Finding validated disease pairs (genetic + clinical evidence)...")

        # Align matrices
        common_nodes = sorted(set(self.opentargets_matrix.index) & set(clinical_matrix.index))
        genetic_aligned = self.opentargets_matrix.loc[common_nodes, common_nodes]
        clinical_aligned = clinical_matrix.loc[common_nodes, common_nodes]

        # Find pairs above thresholds
        validated_pairs = []
        for i in range(len(common_nodes)):
            for j in range(i + 1, len(common_nodes)):
                disease1 = common_nodes[i]
                disease2 = common_nodes[j]

                genetic_score = genetic_aligned.loc[disease1, disease2]
                clinical_score = clinical_aligned.loc[disease1, disease2]

                if genetic_score >= threshold_genetic and clinical_score >= threshold_clinical:
                    validated_pairs.append({
                        'disease1_code': disease1,
                        'disease2_code': disease2,
                        'genetic_score': genetic_score,
                        'clinical_score': clinical_score
                    })

        validated_df = pd.DataFrame(validated_pairs)

        elapsed = time.time() - start_time
        logger.info(f"Found {len(validated_df)} validated disease pairs in {elapsed:.2f}s")

        return validated_df

    def detect_communities(self, min_size: int = 2, max_size: int = 10) -> Dict[int, List[str]]:
        """
        Detect communities in OpenTargets genetic network using Louvain algorithm.

        Args:
            min_size: Minimum community size
            max_size: Maximum community size

        Returns:
            Dictionary mapping community_id -> list of disease codes
        """
        if community_louvain is None:
            logger.error("python-louvain not installed. Cannot detect communities.")
            return {}

        start_time = time.time()
        logger.info(f"Detecting communities in OpenTargets network (size {min_size}-{max_size})...")

        # Convert matrix to networkx graph (only positive edges)
        matrix = self.opentargets_matrix.copy()
        matrix[matrix <= 0] = 0

        G = nx.from_pandas_adjacency(matrix)

        # Remove isolated nodes
        G.remove_nodes_from(list(nx.isolates(G)))

        logger.info(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Try multiple resolutions to find communities in target size range
        best_partition = None
        best_filtered_count = 0
        best_resolution = None

        for resolution in [1.0, 1.5, 2.0, 3.0, 4.0]:
            partition = community_louvain.best_partition(G, weight='weight', resolution=resolution)

            # Group by community
            temp_communities = {}
            for node, comm_id in partition.items():
                temp_communities.setdefault(comm_id, []).append(node)

            # Count communities in target size range
            filtered_count = sum(1 for members in temp_communities.values()
                               if min_size <= len(members) <= max_size)

            logger.info(f"Resolution {resolution}: {len(temp_communities)} communities, "
                       f"{filtered_count} in size range {min_size}-{max_size}")

            if filtered_count > best_filtered_count:
                best_filtered_count = filtered_count
                best_partition = partition
                best_resolution = resolution

        # Use best partition found
        if best_partition is None:
            logger.warning("Could not find communities in target size range with any resolution")
            return {}

        logger.info(f"Using resolution {best_resolution} with {best_filtered_count} communities in range")

        # Group by community with best partition
        communities = {}
        for node, comm_id in best_partition.items():
            communities.setdefault(comm_id, []).append(node)

        # Filter by size
        filtered = {cid: members for cid, members in communities.items()
                   if min_size <= len(members) <= max_size}

        # Calculate modularity
        modularity = community_louvain.modularity(best_partition, G, weight='weight')

        # Log size distribution
        all_sizes = [len(members) for members in communities.values()]
        if all_sizes:
            logger.info(f"Community size distribution: min={min(all_sizes)}, max={max(all_sizes)}, "
                       f"median={sorted(all_sizes)[len(all_sizes)//2]}")

            # Count by size ranges
            size_1 = sum(1 for s in all_sizes if s == 1)
            size_2_10 = sum(1 for s in all_sizes if 2 <= s <= 10)
            size_11_50 = sum(1 for s in all_sizes if 11 <= s <= 50)
            size_51_plus = sum(1 for s in all_sizes if s > 50)

            logger.info(f"Size distribution: 1 disease: {size_1}, "
                       f"2-10: {size_2_10}, 11-50: {size_11_50}, 51+: {size_51_plus}")

        elapsed = time.time() - start_time
        logger.info(f"Detected {len(communities)} total communities (modularity={modularity:.3f})")
        logger.info(f"Filtered to {len(filtered)} communities with size {min_size}-{max_size}")
        logger.info(f"Community detection completed in {elapsed:.2f}s")

        return filtered

    def validate_community_in_clinical(self,
                                       community: List[str],
                                       clinical_matrix: pd.DataFrame,
                                       n_permutations: int = 1000) -> Dict:
        """
        Validate a genetic community in a clinical graph.

        Args:
            community: List of disease codes in community
            clinical_matrix: Clinical co-occurrence matrix
            n_permutations: Number of permutations for statistical test

        Returns:
            Dictionary with validation metrics
        """
        # Filter community to diseases present in clinical matrix
        community_in_clinical = [d for d in community if d in clinical_matrix.index]

        if len(community_in_clinical) < 2:
            return {
                'n_diseases_in_clinical': len(community_in_clinical),
                'internal_cohesion': 0.0,
                'internal_connectivity': 0.0,
                'avg_edge_weight': 0.0,
                'external_cohesion': 0.0,
                'separation_ratio': 0.0,
                'p_value': 1.0,
                'validated': False
            }

        # Calculate internal cohesion (average edge weight within community)
        internal_edges = []
        for i in community_in_clinical:
            for j in community_in_clinical:
                if i != j:
                    weight = clinical_matrix.loc[i, j]
                    if weight > 0:
                        internal_edges.append(weight)

        internal_cohesion = np.mean(internal_edges) if internal_edges else 0.0

        # Calculate internal connectivity (density)
        # Proportion of possible edges that actually exist
        n = len(community_in_clinical)
        max_possible_edges = n * (n - 1)  # Directed or use n*(n-1)/2 for undirected
        actual_edges = sum(1 for e in internal_edges if e > 0)
        internal_connectivity = actual_edges / max_possible_edges if max_possible_edges > 0 else 0.0

        # Calculate average internal edge weight (among existing edges)
        avg_edge_weight = np.mean([e for e in internal_edges if e > 0]) if any(e > 0 for e in internal_edges) else 0.0

        # Calculate external cohesion (average edge weight to outside)
        all_diseases = set(clinical_matrix.index)
        outside_community = list(all_diseases - set(community_in_clinical))

        external_edges = []
        for i in community_in_clinical:
            for j in outside_community:
                weight = clinical_matrix.loc[i, j]
                if weight > 0:
                    external_edges.append(weight)

        external_cohesion = np.mean(external_edges) if external_edges else 0.0

        # Separation ratio
        separation_ratio = internal_cohesion / external_cohesion if external_cohesion > 0 else float('inf')

        # Statistical test: compare to random communities
        null_cohesions = []
        for _ in range(n_permutations):
            # Sample random disease set of same size
            if len(community_in_clinical) <= len(all_diseases):
                random_community = np.random.choice(list(all_diseases),
                                                   size=len(community_in_clinical),
                                                   replace=False)

                random_edges = []
                for i in random_community:
                    for j in random_community:
                        if i != j:
                            weight = clinical_matrix.loc[i, j]
                            if weight > 0:
                                random_edges.append(weight)

                if random_edges:
                    null_cohesions.append(np.mean(random_edges))

        # p-value: proportion of random cohesions >= observed
        if null_cohesions:
            p_value = np.mean(np.array(null_cohesions) >= internal_cohesion)
        else:
            p_value = 1.0

        # Validation criteria
        validated = (p_value < 0.05 and internal_cohesion > 0 and separation_ratio > 1.0)

        return {
            'n_diseases_in_clinical': len(community_in_clinical),
            'internal_cohesion': internal_cohesion,
            'internal_connectivity': internal_connectivity,
            'avg_edge_weight': avg_edge_weight,
            'external_cohesion': external_cohesion,
            'separation_ratio': separation_ratio,
            'p_value': p_value,
            'validated': validated
        }

    def analyze_communities(self, communities: Dict[int, List[str]]) -> pd.DataFrame:
        """
        Analyze and validate communities across all clinical graphs.

        Args:
            communities: Dictionary mapping community_id -> disease list

        Returns:
            DataFrame with community validation results
        """
        start_time = time.time()
        logger.info(f"\nAnalyzing {len(communities)} communities across {len(self.clinical_files)} clinical graphs...")

        results = []

        for comm_id, members in tqdm(communities.items(), desc="Validating communities"):
            # Get community statistics from genetic network
            genetic_matrix = self.opentargets_matrix.loc[members, members]
            internal_density = (genetic_matrix > 0).sum().sum() / (len(members) * (len(members) - 1))
            avg_genetic_similarity = genetic_matrix[genetic_matrix > 0].mean().mean()

            # Validate across clinical graphs
            validations = []
            cohesions = []
            connectivities = []
            edge_weights = []
            p_values = []

            for clinical_file in self.clinical_files:
                clinical_matrix = self.load_clinical_graph(clinical_file)

                validation = self.validate_community_in_clinical(members, clinical_matrix)

                validations.append(validation['validated'])
                cohesions.append(validation['internal_cohesion'])
                connectivities.append(validation['internal_connectivity'])
                edge_weights.append(validation['avg_edge_weight'])
                p_values.append(validation['p_value'])

            # Aggregate validation metrics
            validation_rate = np.mean(validations) * 100  # Percentage
            avg_cohesion = np.mean([c for c in cohesions if c > 0]) if any(c > 0 for c in cohesions) else 0.0
            avg_connectivity = np.mean([c for c in connectivities if c > 0]) if any(c > 0 for c in connectivities) else 0.0
            avg_edge_weight = np.mean([w for w in edge_weights if w > 0]) if any(w > 0 for w in edge_weights) else 0.0
            avg_p_value = np.mean(p_values)

            # Get disease names
            disease_names = [self.get_disease_name(code) for code in members]

            results.append({
                'community_id': comm_id,
                'size': len(members),
                'diseases': ', '.join([f"{code}" for code in members]),
                'disease_names': ', '.join([f"{code}-{name[:20]}" for code, name in zip(members, disease_names)]),
                'genetic_internal_density': internal_density,
                'avg_genetic_similarity': avg_genetic_similarity,
                'validation_rate': validation_rate,
                'avg_cohesion': avg_cohesion,
                'avg_connectivity': avg_connectivity,
                'avg_edge_weight': avg_edge_weight,
                'avg_p_value': avg_p_value,
                'n_validated_graphs': sum(validations),
                'validated_overall': validation_rate >= 50 and avg_p_value < 0.05
            })

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('validation_rate', ascending=False)

        elapsed = time.time() - start_time
        logger.info(f"\nCommunity analysis completed in {elapsed:.2f}s")
        logger.info(f"Validated communities: {results_df['validated_overall'].sum()}/{len(results_df)}")

        return results_df

    def create_comparison_heatmap(self,
                                  disease_code: str,
                                  disease_name: str,
                                  k: int = 50,
                                  output_file: Optional[str] = None):
        """
        Create a heatmap comparing top-K genetic associations across clinical graphs.

        Args:
            disease_code: ICD-10 code of disease of interest
            disease_name: Human-readable disease name
            k: Number of top associations
            output_file: Path to save figure (optional)
        """
        start_time = time.time()
        logger.info(f"Creating comparison heatmap for {disease_name} ({disease_code})...")

        if disease_code not in self.opentargets_matrix.index:
            logger.warning(f"Disease {disease_code} not found in OpenTargets matrix")
            return None

        # Get top-K genetic associations
        genetic_row = self.opentargets_matrix.loc[disease_code]

        # Ensure k doesn't exceed available diseases
        k_actual = min(k, len(genetic_row))
        if k_actual < k:
            logger.warning(f"Requested k={k} but only {k_actual} diseases available")

        if k_actual == 0:
            logger.error(f"No genetic associations found for {disease_code}")
            return None

        top_indices = np.argpartition(genetic_row.values, -k_actual)[-k_actual:]
        top_indices = top_indices[np.argsort(-genetic_row.values[top_indices])]
        top_diseases = genetic_row.index[top_indices].tolist()

        # Collect clinical evidence
        heatmap_data = []
        for clinical_file in self.clinical_files:
            clinical_matrix = self.load_clinical_graph(clinical_file)

            if disease_code not in clinical_matrix.index:
                continue

            clinical_row = clinical_matrix.loc[disease_code]
            clinical_values = [clinical_row[d] if d in clinical_row.index else 0 for d in top_diseases]
            heatmap_data.append(clinical_values)

        if not heatmap_data:
            logger.warning("No clinical data available for heatmap")
            return None

        # Create DataFrame for heatmap with ICD codes and short names
        disease_labels = [f"{code} - {self.get_disease_name(code)}" for code in top_diseases]

        heatmap_df = pd.DataFrame(
            heatmap_data,
            index=[f.stem.replace('All_OR_', '') for f in self.clinical_files],
            columns=disease_labels
        ).T

        # Plot
        fig, ax = plt.subplots(figsize=(14, max(8, k // 3)))
        sns.heatmap(
            heatmap_df > 0,
            cmap=['white', '#2E86AB'],  # Better blue
            cbar=False,
            linewidths=0.5,
            linecolor='gray',
            ax=ax
        )

        ax.set_title(f'{disease_code} - {disease_name}\nTop {k_actual} Genetic Associations - Clinical Validation',
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Clinical Graph (Age Group / Gender)', fontsize=11)
        ax.set_ylabel('Associated Disease (ICD-10 Code - Name)', fontsize=11)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Heatmap saved to {output_file}")

        elapsed = time.time() - start_time
        logger.info(f"Heatmap created in {elapsed:.2f}s")

        return fig

    def generate_pdf_report(self, output_dir: str = "results/graph_comparison",
                           analyzed_diseases: Optional[List[str]] = None):
        """
        Generate comprehensive PDF report with all results and figures.

        Args:
            output_dir: Directory containing results
            analyzed_diseases: List of disease codes that were analyzed (for heatmaps)
        """
        from datetime import datetime

        output_path = Path(output_dir)
        report_file = output_path / f"graph_comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

        logger.info("="*60)
        logger.info("📄 Generating PDF report...")
        logger.info("="*60)

        start_time = time.time()

        with PdfPages(report_file) as pdf:
            # Page 1: Title page
            self._create_title_page(pdf, output_path)

            # Page 2: Executive Summary
            self._create_executive_summary_page(pdf, output_path)

            # Page 3: Edge Overlap Analysis
            self._create_edge_overlap_page(pdf, output_path)

            # Page 4: Correlation Analysis
            self._create_correlation_page(pdf, output_path)

            # Page 5: Top Validated Diseases
            self._create_top_diseases_page(pdf, output_path)

            # Page 6: Community Detection Summary
            self._create_community_summary_page(pdf, output_path)

            # Page 7: Community Table
            self._create_community_table_page(pdf, output_path)

            # Pages 8+: Disease Heatmaps (1 per page, landscape)
            self._add_disease_heatmaps(pdf, output_path, analyzed_diseases)

            # Methods page
            self._create_methods_page(pdf, output_path)

            # Add metadata
            d = pdf.infodict()
            d['Title'] = 'Graph Comparison Analysis Report'
            d['Author'] = 'Vladimir K.'
            d['Subject'] = 'OpenTargets vs Clinical Graph Comparison'
            d['Keywords'] = 'Genetics, Disease Networks, Validation, ICD-10'
            d['CreationDate'] = datetime.now()

        elapsed = time.time() - start_time
        file_size = report_file.stat().st_size / 1024 / 1024

        logger.info("="*60)
        logger.info(f"✅ PDF report generated in {elapsed:.2f}s")
        logger.info(f"   File: {report_file}")
        logger.info(f"   Size: {file_size:.2f} MB")
        logger.info("="*60)

        return report_file

    def _create_title_page(self, pdf: PdfPages, output_path: Path):
        """Create title page for PDF report."""
        from datetime import datetime

        # A4 size: 8.27 x 11.69 inches
        fig = plt.figure(figsize=(8.27, 11.69))
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Title
        ax.text(0.5, 0.85, 'Graph Comparison Analysis Report',
                ha='center', va='center', fontsize=20, fontweight='bold')

        # Subtitle
        ax.text(0.5, 0.78, 'OpenTargets Genetic Networks vs Clinical Co-occurrence Graphs',
                ha='center', va='center', fontsize=12, style='italic')

        # Date
        ax.text(0.5, 0.73, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                ha='center', va='center', fontsize=10)

        # Separator line
        ax.plot([0.2, 0.8], [0.68, 0.68], 'k-', linewidth=1.5)

        # Analysis overview
        try:
            overlap_df = pd.read_csv(output_path / "edge_overlap_summary.csv")
            n_graphs = len(overlap_df)

            summary_text = f"""Analysis Overview:

• Clinical Graphs Analyzed: {n_graphs}
• Comparison Methods: Edge overlap, correlation, validation
• Community Detection: Disease modules (2-10 diseases)
• Statistical Tests: Hypergeometric, permutation testing"""

            ax.text(0.5, 0.48, summary_text, ha='center', va='center', fontsize=10,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
                    linespacing=1.8)
        except:
            pass

        # Footer
        ax.text(0.5, 0.15, 'Vladimir K.\nComputational Health Sciences',
                ha='center', va='center', fontsize=9, style='italic')

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    def _create_executive_summary_page(self, pdf: PdfPages, output_path: Path):
        """Create executive summary page."""
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')

        ax.text(0.5, 0.95, 'Executive Summary', ha='center', va='top', fontsize=18, fontweight='bold')

        try:
            overlap_df = pd.read_csv(output_path / "edge_overlap_summary.csv")
            findings = []

            if 'overlap_percentage' in overlap_df.columns:
                avg_overlap = overlap_df['overlap_percentage'].mean()
                findings.append(f"• Average validation rate: {avg_overlap:.1f}%")

            if 'p_value' in overlap_df.columns:
                significant = (overlap_df['p_value'] < 0.05).sum()
                findings.append(f"• Significant overlap in {significant}/{len(overlap_df)} graphs (p<0.05)")

            if (output_path / "top_validated_diseases.csv").exists():
                top_df = pd.read_csv(output_path / "top_validated_diseases.csv")
                if 'validation_rate' in top_df.columns:
                    avg_val = top_df['validation_rate'].mean()
                    findings.append(f"• Top {len(top_df)} diseases: {avg_val:.1f}% avg validation rate")

            findings_text = "Key Findings:\n\n" + "\n\n".join(findings)
            ax.text(0.1, 0.88, findings_text, ha='left', va='top', fontsize=11,
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        except:
            pass

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def _create_edge_overlap_page(self, pdf: PdfPages, output_path: Path):
        """Create edge overlap analysis page."""
        try:
            overlap_df = pd.read_csv(output_path / "edge_overlap_summary.csv")
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('Edge Overlap Analysis', fontsize=16, fontweight='bold', y=0.98)

            # Jaccard similarity
            ax1 = plt.subplot(3, 1, 1)
            if 'jaccard_similarity' in overlap_df.columns:
                df_sorted = overlap_df.sort_values('jaccard_similarity', ascending=False)
                ax1.barh(range(len(df_sorted)), df_sorted['jaccard_similarity'], color='steelblue')
                ax1.set_yticks(range(len(df_sorted)))
                ax1.set_yticklabels(df_sorted['clinical_graph'], fontsize=8)
                ax1.set_xlabel('Jaccard Similarity')
                ax1.set_title('Network Overlap by Clinical Graph')
                ax1.grid(axis='x', alpha=0.3)

            # Validation percentage
            ax2 = plt.subplot(3, 1, 2)
            if 'overlap_percentage' in overlap_df.columns:
                df_sorted = overlap_df.sort_values('overlap_percentage', ascending=False)
                ax2.barh(range(len(df_sorted)), df_sorted['overlap_percentage'], color='coral')
                ax2.set_yticks(range(len(df_sorted)))
                ax2.set_yticklabels(df_sorted['clinical_graph'], fontsize=8)
                ax2.set_xlabel('Validation Rate (%)')
                ax2.set_title('Percentage of Genetic Edges in Clinical Data')
                ax2.grid(axis='x', alpha=0.3)

            # Statistical significance
            ax3 = plt.subplot(3, 1, 3)
            if 'p_value' in overlap_df.columns:
                overlap_df['neg_log_p'] = -np.log10(overlap_df['p_value'].clip(lower=1e-300))
                df_sorted = overlap_df.sort_values('neg_log_p', ascending=False)
                bars = ax3.barh(range(len(df_sorted)), df_sorted['neg_log_p'])
                ax3.set_yticks(range(len(df_sorted)))
                ax3.set_yticklabels(df_sorted['clinical_graph'], fontsize=8)
                ax3.set_xlabel('-log10(p-value)')
                ax3.set_title('Statistical Significance')
                ax3.axvline(x=-np.log10(0.05), color='red', linestyle='--', label='p=0.05')
                ax3.legend()
                ax3.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not create edge overlap page: {e}")

    def _create_correlation_page(self, pdf: PdfPages, output_path: Path):
        """Create correlation analysis page."""
        try:
            corr_df = pd.read_csv(output_path / "correlation_summary.csv")
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('Correlation Analysis', fontsize=16, fontweight='bold', y=0.98)

            # Spearman correlation
            ax1 = plt.subplot(2, 1, 1)
            if 'spearman_correlation' in corr_df.columns:
                df_sorted = corr_df.sort_values('spearman_correlation', ascending=False)
                ax1.barh(range(len(df_sorted)), df_sorted['spearman_correlation'], color='steelblue')
                ax1.set_yticks(range(len(df_sorted)))
                ax1.set_yticklabels(df_sorted['clinical_graph'], fontsize=8)
                ax1.set_xlabel('Spearman ρ')
                ax1.set_title('Spearman Correlation: Genetic vs Clinical Edge Weights')
                ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                ax1.grid(axis='x', alpha=0.3)

            # Pearson correlation
            ax2 = plt.subplot(2, 1, 2)
            if 'pearson_correlation' in corr_df.columns:
                df_sorted = corr_df.sort_values('pearson_correlation', ascending=False)
                ax2.barh(range(len(df_sorted)), df_sorted['pearson_correlation'], color='coral')
                ax2.set_yticks(range(len(df_sorted)))
                ax2.set_yticklabels(df_sorted['clinical_graph'], fontsize=8)
                ax2.set_xlabel('Pearson r')
                ax2.set_title('Pearson Correlation: Genetic vs Clinical Edge Weights')
                ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                ax2.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not create correlation page: {e}")

    def _create_top_diseases_page(self, pdf: PdfPages, output_path: Path):
        """Create top validated diseases page."""
        try:
            top_df = pd.read_csv(output_path / "top_validated_diseases.csv")
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('Top Validated Diseases', fontsize=16, fontweight='bold', y=0.98)

            # Validation rate
            ax1 = plt.subplot(2, 1, 1)
            if 'validation_rate' in top_df.columns:
                n_show = min(10, len(top_df))
                top_n = top_df.head(n_show)
                labels = [f"{row['icd_code'][:6]} - {row['disease_name'][:30]}" for _, row in top_n.iterrows()]
                bars = ax1.barh(range(len(top_n)), top_n['validation_rate'])
                ax1.set_yticks(range(len(top_n)))
                ax1.set_yticklabels(labels, fontsize=7)
                ax1.set_xlabel('Validation Rate (%)')
                ax1.set_title(f'Top {n_show} Diseases by Validation Rate')
                ax1.grid(axis='x', alpha=0.3)

                for i, val in enumerate(top_n['validation_rate']):
                    if val > 70:
                        bars[i].set_color('darkgreen')
                    elif val > 50:
                        bars[i].set_color('steelblue')
                    else:
                        bars[i].set_color('coral')

            # Genetic vs validated
            ax2 = plt.subplot(2, 1, 2)
            if 'n_genetic_associations' in top_df.columns and 'avg_validated_per_graph' in top_df.columns:
                n_show = min(10, len(top_df))
                top_n = top_df.head(n_show)
                x = np.arange(len(top_n))
                width = 0.35
                ax2.barh(x - width/2, top_n['n_genetic_associations'], width, label='Genetic', color='lightblue')
                ax2.barh(x + width/2, top_n['avg_validated_per_graph'], width, label='Validated', color='darkblue')
                ax2.set_yticks(x)
                ax2.set_yticklabels(labels, fontsize=7)
                ax2.set_xlabel('Number of Associations')
                ax2.set_title('Genetic vs Validated Associations')
                ax2.legend()
                ax2.grid(axis='x', alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
        except Exception as e:
            logger.warning(f"Could not create top diseases page: {e}")

    def _add_disease_heatmaps(self, pdf: PdfPages, output_path: Path,
                              analyzed_diseases: Optional[List[str]] = None):
        """Add disease heatmaps to PDF.

        Args:
            pdf: PdfPages object
            output_path: Path to results directory
            analyzed_diseases: List of disease codes that were analyzed.
                              Only heatmaps for these diseases will be included.
        """
        # Get all heatmap files
        all_heatmap_files = sorted(output_path.glob("heatmap_*.png"))

        # Filter to only analyzed diseases if list provided
        if analyzed_diseases is not None:
            heatmap_files = []
            for heatmap_file in all_heatmap_files:
                disease_code = heatmap_file.stem.replace('heatmap_', '')
                if disease_code in analyzed_diseases:
                    heatmap_files.append(heatmap_file)

            logger.info(f"Adding {len(heatmap_files)} heatmaps to PDF (filtered from {len(all_heatmap_files)} total)")
        else:
            heatmap_files = all_heatmap_files
            logger.info(f"Adding {len(heatmap_files)} heatmaps to PDF...")

        for heatmap_file in heatmap_files:
            # Create page with proper size (8.5 x 11 inches)
            fig, ax = plt.subplots(figsize=(8.5, 11))

            try:
                img = plt.imread(heatmap_file)

                # Calculate aspect ratio and display with proper scaling
                height, width = img.shape[:2]
                aspect = width / height

                # Position image to fit nicely on page
                ax.imshow(img)
                ax.axis('off')

                # Add title
                disease_code = heatmap_file.stem.replace('heatmap_', '')
                fig.suptitle(f'Disease Validation Heatmap: {disease_code}',
                            fontsize=12, fontweight='bold', y=0.98)
            except Exception as e:
                logger.warning(f"Could not load {heatmap_file}: {e}")
                ax.text(0.5, 0.5, f'Error loading heatmap:\n{heatmap_file.name}',
                       ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')

            plt.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    def _create_community_summary_page(self, pdf: PdfPages, output_path: Path):
        """Create community detection summary page."""
        try:
            comm_file = output_path / "community_analysis.csv"
            if not comm_file.exists():
                logger.warning("Community analysis file not found, skipping page")
                return

            comm_df = pd.read_csv(comm_file)

            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('Disease Community Analysis', fontsize=16, fontweight='bold', y=0.98)

            # Plot 1: Validation rate by community
            ax1 = plt.subplot(4, 1, 1)
            if 'validation_rate' in comm_df.columns:
                top_n = min(20, len(comm_df))
                df_sorted = comm_df.head(top_n)

                bars = ax1.barh(range(len(df_sorted)), df_sorted['validation_rate'])
                ax1.set_yticks(range(len(df_sorted)))
                ax1.set_yticklabels([f"C{row['community_id']} (n={row['size']})"
                                    for _, row in df_sorted.iterrows()], fontsize=7)
                ax1.set_xlabel('Validation Rate (%)')
                ax1.set_title(f'Top {top_n} Communities by Validation Rate')
                ax1.grid(axis='x', alpha=0.3)

                # Color by validation
                for i, val in enumerate(df_sorted['validation_rate']):
                    if val >= 75:
                        bars[i].set_color('darkgreen')
                    elif val >= 50:
                        bars[i].set_color('steelblue')
                    else:
                        bars[i].set_color('coral')

            # Plot 2: Size distribution
            ax2 = plt.subplot(4, 1, 2)
            if 'size' in comm_df.columns:
                size_counts = comm_df['size'].value_counts().sort_index()
                ax2.bar(size_counts.index, size_counts.values, color='steelblue', alpha=0.7)
                ax2.set_xlabel('Community Size (number of diseases)')
                ax2.set_ylabel('Number of Communities')
                ax2.set_title('Community Size Distribution')
                ax2.grid(axis='y', alpha=0.3)

            # Plot 3: Validation vs cohesion scatter
            ax3 = plt.subplot(4, 1, 3)
            if 'validation_rate' in comm_df.columns and 'avg_cohesion' in comm_df.columns:
                validated = comm_df['validated_overall'] == True
                ax3.scatter(comm_df.loc[validated, 'avg_cohesion'],
                           comm_df.loc[validated, 'validation_rate'],
                           c='darkgreen', alpha=0.6, s=50, label='Validated')
                ax3.scatter(comm_df.loc[~validated, 'avg_cohesion'],
                           comm_df.loc[~validated, 'validation_rate'],
                           c='lightgray', alpha=0.6, s=50, label='Not Validated')
                ax3.set_xlabel('Average Clinical Cohesion')
                ax3.set_ylabel('Validation Rate (%)')
                ax3.set_title('Validation Rate vs Clinical Cohesion')
                ax3.legend()
                ax3.grid(alpha=0.3)

            # Plot 4: Internal connectivity analysis
            ax4 = plt.subplot(4, 1, 4)
            if 'avg_connectivity' in comm_df.columns and 'avg_edge_weight' in comm_df.columns:
                validated = comm_df['validated_overall'] == True
                ax4.scatter(comm_df.loc[validated, 'avg_connectivity'],
                           comm_df.loc[validated, 'avg_edge_weight'],
                           c='darkgreen', alpha=0.6, s=50, label='Validated')
                ax4.scatter(comm_df.loc[~validated, 'avg_connectivity'],
                           comm_df.loc[~validated, 'avg_edge_weight'],
                           c='lightgray', alpha=0.6, s=50, label='Not Validated')
                ax4.set_xlabel('Internal Connectivity (density)')
                ax4.set_ylabel('Average Edge Weight')
                ax4.set_title('Clinical Network Connectivity of Communities')
                ax4.legend()
                ax4.grid(alpha=0.3)

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            logger.error(f"Error creating community summary page: {e}")

    def _create_community_table_page(self, pdf: PdfPages, output_path: Path):
        """Create table of top validated communities."""
        try:
            comm_file = output_path / "community_analysis.csv"
            if not comm_file.exists():
                return

            comm_df = pd.read_csv(comm_file)

            fig = plt.figure(figsize=(8.5, 11))
            ax = fig.add_subplot(111)
            ax.axis('off')

            ax.text(0.5, 0.97, 'Top Validated Disease Communities',
                   ha='center', va='top', fontsize=14, fontweight='bold')

            # Show top 10 communities
            top_n = min(10, len(comm_df))
            table_df = comm_df.head(top_n).copy()

            # Prepare table data
            display_cols = []
            col_names = []

            if 'community_id' in table_df.columns:
                display_cols.append(table_df['community_id'].astype(str))
                col_names.append('ID')

            if 'size' in table_df.columns:
                display_cols.append(table_df['size'].astype(int))
                col_names.append('Size')

            if 'validation_rate' in table_df.columns:
                display_cols.append(table_df['validation_rate'].round(1).astype(str) + '%')
                col_names.append('Val%')

            if 'avg_cohesion' in table_df.columns:
                display_cols.append(table_df['avg_cohesion'].round(3))
                col_names.append('Cohesion')

            if 'avg_connectivity' in table_df.columns:
                display_cols.append(table_df['avg_connectivity'].round(3))
                col_names.append('Connect')

            if 'avg_edge_weight' in table_df.columns:
                display_cols.append(table_df['avg_edge_weight'].round(3))
                col_names.append('EdgeWt')

            if 'avg_p_value' in table_df.columns:
                display_cols.append(table_df['avg_p_value'].apply(lambda x: f"{x:.3f}"))
                col_names.append('p-value')

            if 'disease_names' in table_df.columns:
                display_cols.append(table_df['disease_names'].str[:60])
                col_names.append('Diseases')

            if display_cols:
                table_data = np.column_stack(display_cols)

                table = ax.table(cellText=table_data,
                               colLabels=col_names,
                               cellLoc='left',
                               loc='upper center',
                               bbox=[0.05, 0.05, 0.9, 0.88])

                table.auto_set_font_size(False)
                table.set_fontsize(7)
                table.scale(1, 1.8)

                # Style header
                for i in range(len(col_names)):
                    table[(0, i)].set_facecolor('lightblue')
                    table[(0, i)].set_text_props(weight='bold')

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            logger.error(f"Error creating community table page: {e}")

    def _create_methods_page(self, pdf: PdfPages, output_path: Path):
        """Create methods explanation page."""
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')

        ax.text(0.5, 0.97, 'Methodology', ha='center', va='top', fontsize=18, fontweight='bold')

        methods_text = """
HOW TOP VALIDATED DISEASES ARE SELECTED:

1. For each disease in OpenTargets matrix:
   • Count genetic associations (n_genetic_associations)
     → Number of diseases connected via shared genes in OpenTargets

   • For each clinical graph (age group × gender):
     - Find which genetic associations also appear in clinical data
     - Count validated associations (genetic ∩ clinical)

   • Calculate metrics:
     - avg_validated_per_graph = total_validated / n_clinical_graphs
     - validation_rate = (avg_validated / n_genetic) × 100%

2. Rank diseases by avg_validated_per_graph (descending)

3. Select top 10 diseases

Example:
  Disease X has 50 genetic associations (from OpenTargets)
  Across 8 clinical graphs: 120 total validations (15 per graph average)
  → avg_validated_per_graph = 15
  → validation_rate = 15/50 × 100% = 30%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

WHAT IS "VALIDATED"?

Validated = A genetic association (from OpenTargets) that ALSO appears
            in clinical co-occurrence graphs (hospital data)

• Genetic Association: Two diseases share genes (OpenTargets)
• Clinical Co-occurrence: Two diseases appear together in patients (hospital)
• Validated: Genetic association confirmed by clinical evidence

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GENETIC VS VALIDATED ASSOCIATIONS:

• Genetic Associations: Total disease pairs connected via shared genes
                        (from OpenTargets platform)

• Validated Associations: Subset of genetic associations that also appear
                          in clinical co-occurrence graphs (average per graph)

High validation rate = Genetic predictions match clinical reality

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

HEATMAP INTERPRETATION:

• Rows: Top genetic associations for a disease
• Columns: Clinical graphs (age groups × gender)
• Blue cell: Association validated in that clinical graph
• White cell: Association not found in clinical data

More blue = Better clinical validation of genetic predictions

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DATA SOURCES:

• OpenTargets Platform (v25.09): Genetic disease associations
• Clinical Graphs: Hospital co-occurrence data (ICD-10 coded)
  - Stratified by age groups and gender
  - Multiple graphs provide validation across demographics
        """

        ax.text(0.05, 0.92, methods_text.strip(), ha='left', va='top', fontsize=8,
                family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    def run_comprehensive_comparison(self,
                                     auto_discover: bool = True,
                                     target_diseases: Optional[Dict[str, str]] = None,
                                     top_n_diseases: int = 10,
                                     output_dir: str = "results/graph_comparison"):
        """
        Run comprehensive comparison analysis.

        Args:
            auto_discover: Automatically find most validated diseases
            target_diseases: Dict of {ICD_code: disease_name} to analyze (overrides auto_discover)
            top_n_diseases: Number of top diseases to analyze if auto_discover=True
            output_dir: Directory for output files
        """
        total_start = time.time()
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("="*60)
        logger.info("Starting comprehensive graph comparison")
        logger.info("="*60)

        # Load all clinical matrices
        clinical_matrices = []
        all_overlap_results = []
        all_correlation_results = []

        for clinical_file in self.clinical_files:
            logger.info(f"\n--- Analyzing {clinical_file.name} ---")

            clinical_matrix = self.load_clinical_graph(clinical_file)
            clinical_matrices.append(clinical_matrix)
            graph_name = clinical_file.stem.replace('All_OR_', '')

            # Edge overlap analysis
            overlap_results = self.compute_edge_overlap(
                self.opentargets_matrix,
                clinical_matrix
            )
            overlap_results['clinical_graph'] = graph_name
            all_overlap_results.append(overlap_results)

            # Correlation analysis
            spearman_corr, spearman_p = self.compute_correlation(
                self.opentargets_matrix,
                clinical_matrix,
                method='spearman'
            )
            pearson_corr, pearson_p = self.compute_correlation(
                self.opentargets_matrix,
                clinical_matrix,
                method='pearson'
            )

            all_correlation_results.append({
                'clinical_graph': graph_name,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p,
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p
            })

        # Save summary results
        overlap_df = pd.DataFrame(all_overlap_results)
        overlap_df.to_csv(output_path / 'edge_overlap_summary.csv', index=False)
        logger.info(f"Saved edge overlap summary to {output_path / 'edge_overlap_summary.csv'}")

        correlation_df = pd.DataFrame(all_correlation_results)
        correlation_df.to_csv(output_path / 'correlation_summary.csv', index=False)
        logger.info(f"Saved correlation summary to {output_path / 'correlation_summary.csv'}")

        # Determine which diseases to analyze
        if target_diseases is None and auto_discover:
            logger.info("\n" + "="*60)
            logger.info("🔍 AUTO-DISCOVERING most validated diseases...")
            logger.info("="*60)

            top_diseases_df = self.find_top_diseases_by_validation(
                clinical_matrices,
                top_n=top_n_diseases,
                min_genetic_edges=10
            )

            # Save top diseases report
            top_diseases_df.to_csv(output_path / 'top_validated_diseases.csv', index=False)
            logger.info(f"\n✅ Top {len(top_diseases_df)} validated diseases:")
            for idx, row in top_diseases_df.iterrows():
                logger.info(f"  {idx+1}. {row['icd_code']} - {row['disease_name']}: "
                          f"{row['avg_validated_per_graph']:.1f} validated/graph "
                          f"({row['validation_rate']:.1f}%)")

            # Convert to target_diseases format
            target_diseases = {
                row['icd_code']: row['disease_name']
                for _, row in top_diseases_df.iterrows()
            }
        elif target_diseases is None:
            logger.warning("No target diseases specified and auto_discover=False. Using defaults.")
            target_diseases = {}

        # Analyze specific diseases
        logger.info("\n" + "="*60)
        logger.info(f"Analyzing {len(target_diseases)} diseases in detail...")
        logger.info("="*60)

        for disease_code, disease_name in target_diseases.items():
            logger.info(f"\n--- Analyzing {disease_code} - {disease_name} ---")

            # Top-K analysis
            topk_results = self.analyze_top_k_neighbors(disease_code, disease_name, k=10)
            if not topk_results.empty:
                topk_results.to_csv(
                    output_path / f'topk_validation_{disease_code}.csv',
                    index=False
                )

            # Create heatmap
            self.create_comparison_heatmap(
                disease_code,
                disease_name,
                k=10,  # Top 10 associations per disease
                output_file=str(output_path / f'heatmap_{disease_code}.png')
            )

        # Community detection and validation
        if community_louvain is not None:
            logger.info("\n" + "="*60)
            logger.info("🔍 COMMUNITY DETECTION ANALYSIS")
            logger.info("="*60)

            communities = self.detect_communities(min_size=2, max_size=10)

            if communities:
                community_results = self.analyze_communities(communities)
                community_results.to_csv(output_path / 'community_analysis.csv', index=False)
                logger.info(f"✅ Community analysis saved to {output_path / 'community_analysis.csv'}")

                logger.info(f"\n📊 Community Detection Summary:")
                logger.info(f"   - Total communities detected: {len(communities)}")
                logger.info(f"   - Validated communities: {community_results['validated_overall'].sum()}")
                logger.info(f"   - Average validation rate: {community_results['validation_rate'].mean():.1f}%")

                if len(community_results) > 0:
                    logger.info(f"\n🏆 Top 5 Validated Communities:")
                    for idx, row in community_results.head(5).iterrows():
                        logger.info(f"   {idx+1}. Community {row['community_id']} "
                                  f"(size={row['size']}, validation={row['validation_rate']:.1f}%)")
                        logger.info(f"      Diseases: {row['disease_names'][:100]}")
        else:
            logger.warning("python-louvain not installed. Skipping community detection.")

        total_elapsed = time.time() - total_start
        logger.info("\n" + "="*60)
        logger.info(f"Comprehensive comparison completed in {total_elapsed/60:.2f} minutes")
        logger.info(f"Results saved to {output_dir}")
        logger.info("="*60)

        # Generate PDF report with analyzed diseases list
        analyzed_disease_codes = list(target_diseases.keys())
        self.generate_pdf_report(output_dir=output_dir, analyzed_diseases=analyzed_disease_codes)


def main(auto_discover: bool = True, top_n: int = 10):
    """
    Main entry point.

    Args:
        auto_discover: Automatically find most validated diseases (default: True)
        top_n: Number of top diseases to analyze if auto_discover=True (default: 10)
    """
    # Start timing
    total_start_time = time.time()

    # Configuration
    OPENTARGETS_FILE = "harmonized_open_target_icd.csv"
    CLINICAL_DIR = "data/dementia_age_groups"
    DIAGNOSIS_FILE = "data/Diagnosis_global_10years.csv"
    ICD_FILE = "data/DiagAll_Eng.csv"
    OUTPUT_DIR = "results/graph_comparison"

    logger.info("🚀 Starting with configuration:")
    logger.info(f"   - Auto-discover diseases: {auto_discover}")
    logger.info(f"   - Top N diseases: {top_n}")

    # Run comparison
    comparator = GraphComparator(
        opentargets_file=OPENTARGETS_FILE,
        clinical_dir=CLINICAL_DIR,
        diagnosis_file=DIAGNOSIS_FILE,
        icd_file=ICD_FILE
    )

    # Run comprehensive analysis
    comparator.run_comprehensive_comparison(
        auto_discover=auto_discover,
        top_n_diseases=top_n,
        output_dir=OUTPUT_DIR
    )

    # Display total execution time
    total_elapsed = time.time() - total_start_time
    hours = int(total_elapsed // 3600)
    minutes = int((total_elapsed % 3600) // 60)
    seconds = total_elapsed % 60

    logger.info("\n" + "=" * 60)
    logger.info("✅ ANALYSIS COMPLETE")
    logger.info("=" * 60)
    if hours > 0:
        logger.info(f"⏱️  Total execution time: {hours}h {minutes}m {seconds:.1f}s")
    elif minutes > 0:
        logger.info(f"⏱️  Total execution time: {minutes}m {seconds:.1f}s")
    else:
        logger.info(f"⏱️  Total execution time: {seconds:.1f}s")
    logger.info("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare genetic predisposition and clinical co-occurrence networks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover top 10 most validated diseases (default)
  python compare_graphs.py

  # Auto-discover top 5 diseases (faster, smaller PDF)
  python compare_graphs.py --top-n 5

  # Auto-discover top 20 diseases (more comprehensive)
  python compare_graphs.py --top-n 20

  # Manually specify diseases (advanced)
  # Edit target_diseases in main() function
        """
    )

    parser.add_argument(
        '--no-auto-discover',
        action='store_true',
        help='Disable auto-discovery (must manually specify diseases in code)'
    )

    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top diseases to analyze (default: 10)'
    )

    args = parser.parse_args()

    main(
        auto_discover=not args.no_auto_discover,
        top_n=args.top_n
    )
