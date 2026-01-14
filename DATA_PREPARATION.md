# Data Preparation Pipeline: Harmonizing Open Targets and Clinical Graphs

## Overview
Pipeline harmonizes genetic disease associations from Open Targets 25.09 with clinical hospital diagnosis data using ICD-10 mapping.

---

## Part 1: Open Targets Disease-Disease Network (prepare_open_targets.ipynb)

### 1.1 Data Acquisition
- **Source**: Open Targets Platform 25.09
- **Dataset**: `association_by_datasource_direct` parquet files
- **Initial records**: 4,200,235 disease-target associations

### 1.2 Data Filtering
**Removed datatype**: `literature` (reduces bias from publication frequency)

**Retained datatypes** (N=1,779,799):
- `genetic_association`: 714,130
- `animal_model`: 690,354
- `rna_expression`: 166,503
- `somatic_mutation`: 96,504
- `known_drug`: 74,187
- `affected_pathway`: 38,121

### 1.3 Score Aggregation
**Method**: Bounded harmonic sum per (disease, target) pair:
```
S = sum(scores)
customAssociationScore = S / (1.0 + S)
```
**Output**: 1,725,160 unique disease-target pairs

### 1.4 Disease Name Filtering
**Excluded keywords** (case-insensitive):
- measurement, level, concentration, trait, response, ratio, presence, status
- microdeletion, microduplication, deletion, duplication

**Rationale**: Remove non-disease entities and genomic alterations
**Result**: 1,468,469 associations retained

### 1.5 Disease-Disease Correlation Matrix
**Method**:
1. Pivot to disease × gene matrix (diseases as rows, genes as columns)
2. Compute Pearson correlation between disease gene vectors using `np.corrcoef()`
3. Clean matrix:
   - Set diagonal to 0 (remove self-loops)
   - Set negative correlations to 0
   - Fill NaN with 0

**Output matrix**:
- Dimensions: 14,417 × 14,417 diseases
- Sparsity: 81.58% zeros
- Non-zero edges: 19,142,995
- Saved as: `data/open_targets_25_09/dis_dis_nolit_25_09.csv`

**Note**: No percentile thresholding applied at this stage

---

## Part 2: ICD-10 Harmonization (map_icd.py)

### 2.1 Semantic Disease Mapping
**Model**: SentenceTransformer `NeuML/pubmedbert-base-embeddings`

**Process**:
1. Encode ICD-10 descriptions (from `DiagAll_Eng.csv`) as embeddings
2. Encode Open Targets disease names as embeddings
3. Match using cosine similarity
4. **Threshold**: Similarity > 0.5 (else marked "UNKNOWN")
5. Prioritize exact matches before semantic matching

### 2.2 Matrix Filtering
1. Remove diseases mapped to "UNKNOWN"
2. Filter correlation matrix to valid diseases only
3. Rename diseases using ICD-10 descriptions

### 2.3 Handling Duplicate ICD-10 Codes
**Strategy**: Multiple Open Targets diseases may map to same ICD-10 code

**Resolution**: Take maximum correlation across duplicates:
```python
filtered_open_target_df[col + '_temp'] = cols_to_mean.max(axis=1)
filtered_open_target_df.loc[col + '_temp', :] = rows_to_mean.max(axis=0)
```

### 2.4 Dementia Code Merging
**Target codes**:
- F00: Dementia in Alzheimer's disease
- F01: Vascular dementia
- G30: Alzheimer's disease

**Operation**: OR aggregation (max) across rows and columns:
```python
row_or = harmonized_open_targets.loc[target_labels].max(axis=0)
harmonized_open_targets.loc['Dementia in Alzheimer\'s disease'] = row_or
```

**Rationale**: Unify genetic risk factors across dementia subtypes

### 2.5 Output Files
1. `harmonized_open_target.csv` - ICD-10 disease names
2. `harmonized_open_target_icd.csv` - ICD-10 codes as labels

---

## Part 3: Clinical Graph Validation (map_icd.py)

### 3.1 Hospital Data Integration
**Source**: `Diagnosis_global_10years.csv`
- Age filter: `age_id < 9`
- Records: 1,080 diagnosis codes
- Clinical graphs: RDS files by age group and gender (`data/dementia_age_groups/*.rds`)

### 3.2 Top-K Disease Selection
**Parameter**: TOP_K = 50 (configurable: tested 5, 20, 50, 100)

**Method**: For each dementia code (F00, G30):
1. Extract top-K correlated diseases from Open Targets matrix
2. Check presence in clinical correlation matrices (hospital data)
3. Identify which age groups and genders show clinical correlation

### 3.3 Literature Validation
**Method**: PubMed co-occurrence counts via Entrez API
- Query: `"[dementia_code] AND [correlated_disease]"`
- Max results: 20,000 articles
- Retry logic: 5 attempts with exponential backoff (0.5s initial delay)

### 3.4 Visualization
**Output**: Heatmap per dementia code showing:
- Rows: Top-K genetically correlated diseases + ICD-10 code + PubMed count
- Columns: Age groups (20s-70s) × Gender (Female, Male)
- Color: White (not detected), Pink (Female), Blue (Male)
- Saved as: `dementia_heatmaps_{TOP_K}.pdf`

---

## Key Parameters Summary

| Parameter | Value | Location |
|-----------|-------|----------|
| Open Targets version | 25.09 | prepare_open_targets.ipynb |
| Excluded datatype | literature | prepare_open_targets.ipynb |
| Semantic similarity threshold | 0.5 | map_icd.py:68 |
| Correlation method | Pearson | prepare_open_targets.ipynb |
| Duplicate resolution | max | map_icd.py:110-112 |
| Top-K diseases | 50 | map_icd.py:243 |
| Hospital age filter | age_id < 9 | map_icd.py:177 |
| PubMed retry attempts | 5 | map_icd.py:197 |

---

## Data Flow Diagram

```
Open Targets 25.09
    ↓ [Filter: remove literature]
Disease-Target Associations (1.78M)
    ↓ [Bounded harmonic sum]
Unique Disease-Target Pairs (1.73M)
    ↓ [Filter: exclude measurements]
Clean Associations (1.47M)
    ↓ [Pivot + Pearson correlation]
Disease-Disease Matrix (14,417 × 14,417)
    ↓ [Semantic mapping to ICD-10]
Harmonized Matrix (ICD-10 labels)
    ↓ [Merge dementia codes]
Final Matrix for Comparison
    ↓ [Top-K selection + Clinical validation]
Heatmaps + PubMed Validation
```

---

## Validation Checkpoints

1. **Genetic data quality**: Removed literature associations to reduce publication bias
2. **Disease filtering**: Excluded non-disease entities (measurements, traits)
3. **Correlation validity**: Removed negative correlations and self-loops
4. **Mapping accuracy**: Semantic similarity threshold 0.5 + exact match priority
5. **Clinical concordance**: Cross-reference with hospital diagnosis correlations
6. **Literature support**: PubMed co-occurrence counts as independent validation

---

## Notes for Scientists

- **Sparsity**: 81.58% of genetic correlations are zero (highly sparse network)
- **Thresholding**: No percentile cutoff applied - full correlation matrix preserved
- **Duplicate handling**: Max operation may amplify signals across similar diseases
- **Dementia merging**: OR operation assumes shared genetic architecture across subtypes
- **Clinical validation**: Presence in hospital data confirms real-world co-occurrence patterns
