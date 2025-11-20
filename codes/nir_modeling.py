import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cross_decomposition import PLSRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path):
    """
    Load data from file
    """
    print("Loading data ...")
    df = pd.read_csv(file_path)

    return df

def get_spectral_columns(df):
    """Get spectral columns"""
    spectral_cols = []
    for col in df.columns:
        try:
            col_clean = col.strip('"').replace("wl_", "")
            if col_clean.replace('.', '').isdigit() or '.' in col_clean:
                wavelength = float(col_clean)
                if 2121 <= wavelength <= 2339:
                    spectral_cols.append(col)
        except:
            continue
    return spectral_cols

def analyze_farm_specific_variance(df, farm_id, spectral_cols):
    """Analyze feature variance for specific farm"""
    print(f"Analyzing spectral feature variance for farm {farm_id}...")
    
    # Filter data for current farm
    farm_df = df[df['FARM_ID'] == farm_id].copy()
    
    if len(farm_df) < 5:
        print(f"  Farm {farm_id} has too few samples, skipping")
        return None
    
    # Calculate variance for each feature
    feature_variances = {}
    for col in spectral_cols:
        if col in farm_df.columns:
            variance = farm_df[col].var()
            feature_variances[col] = variance
    
    # Convert to DataFrame for analysis
    variance_df = pd.DataFrame(list(feature_variances.items()), 
                              columns=['feature', 'variance'])
    variance_df = variance_df.sort_values('variance', ascending=False)
    
    return variance_df

def select_farm_specific_features(variance_df, keep_percentile=0.5):
    """
    Select farm-specific features based on variance
    
    Args:
        variance_df: DataFrame containing features and variances
        keep_percentile: Percentile to keep (0.5 means keep 50%)
    """
    if variance_df is None or len(variance_df) == 0:
        return []
    
    # Select based on percentile
    threshold = variance_df['variance'].quantile(1 - keep_percentile)
    selected_features = variance_df[variance_df['variance'] >= threshold]['feature'].tolist()
    
    print(f"  Original feature count: {len(variance_df)}")
    print(f"  Selected feature count: {len(selected_features)}")
    print(f"  Retention ratio: {len(selected_features)/len(variance_df)*100:.1f}%")
    print(f"  Variance threshold: {threshold:.6f}")
    
    return selected_features

def enhanced_collate_fn(batch):
    """
    Enhanced collate function - handles complete timeline data
    """
    content_sequences, metadata_sequences, interval_sequences, missing_masks, labels, cow_ids = zip(*batch)
    
    # Since all sequences are now the same length (complete timeline), no padding needed
    # But still convert to tensor
    content_tensor = torch.stack(content_sequences)      # [batch, timeline_length, content_dim]
    metadata_tensor = torch.stack(metadata_sequences)    # [batch, timeline_length, metadata_dim]
    interval_tensor = torch.stack(interval_sequences)    # [batch, timeline_length]
    missing_mask_tensor = torch.stack(missing_masks)     # [batch, timeline_length] True=missing
    
    return (
        content_tensor,
        metadata_tensor, 
        interval_tensor,
        ~missing_mask_tensor,  # Invert mask: True=valid data, False=missing data (consistent with original convention)
        torch.LongTensor([label.item() for label in labels]),
        cow_ids
    )

def match_dim_distribution(mastitis_group, healthy_group, random_state=None, window_start=None, window_end=None):
    """
    Sample data from healthy group with the same DIM distribution as mastitis group
    New: Ensure matching consecutive DIM values within time window
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Get DIM distribution of mastitis group
    mastitis_dim_counts = mastitis_group['DIM'].value_counts()
    
    matched_healthy = []
    
    # If in time window mode, need to match consecutive DIM values
    if window_start is not None and window_end is not None:
        print(f"Time window mode: {window_start}-{window_end} days ago, matching consecutive DIM values")
        
        # Get DIM range of mastitis cows within time window
        mastitis_dims_in_window = mastitis_group['DIM'].values
        dim_min = mastitis_dims_in_window.min()
        dim_max = mastitis_dims_in_window.max()
        
        print(f"Mastitis cow DIM range: {dim_min}-{dim_max}")
        
        # Find healthy cows with consecutive DIM values
        healthy_cow_ids = healthy_group['ANIMAL_SOURCE_ID'].unique()
        suitable_healthy_cows = []
        
        for cow_id in healthy_cow_ids:
            cow_data = healthy_group[healthy_group['ANIMAL_SOURCE_ID'] == cow_id]
            cow_dims = sorted(cow_data['DIM'].values)
            
            # Check if there are consecutive DIM values within mastitis cow DIM range
            for i in range(len(cow_dims) - 4):  # Need at least 5 consecutive points
                consecutive_dims = cow_dims[i:i+5]  # Take 5 consecutive DIM values
                
                # Check if all 5 DIM values are within mastitis cow DIM range
                if (all(dim_min <= dim <= dim_max for dim in consecutive_dims) and
                    consecutive_dims[-1] - consecutive_dims[0] <= 20):  # DIM span no more than 20 days
                    
                    # Select these 5 consecutive data points
                    selected_data = []
                    for dim in consecutive_dims:
                        dim_data = cow_data[cow_data['DIM'] == dim]
                        if len(dim_data) > 0:
                            selected_data.append(dim_data.iloc[0])
                    
                    if len(selected_data) == 5:  # Ensure complete 5 points
                        suitable_healthy_cows.append(pd.DataFrame(selected_data))
                        break  # One set is enough
        
        # Select appropriate number of healthy cows
        if len(suitable_healthy_cows) > 0:
            # Calculate how many healthy cows needed to match mastitis cow count
            n_mastitis_cows = len(mastitis_group['ANIMAL_SOURCE_ID'].unique())
            n_healthy_needed = min(len(suitable_healthy_cows), n_mastitis_cows)
            
            # Select indices instead of DataFrame
            selected_indices = np.random.choice(
                range(len(suitable_healthy_cows)), 
                size=n_healthy_needed, 
                replace=False
            )
            
            for idx in selected_indices:
                matched_healthy.append(suitable_healthy_cows[idx])
            
            print(f"Found {len(suitable_healthy_cows)} healthy cows with consecutive DIM values, selected {n_healthy_needed}")
        else:
            print("Warning: No healthy cows with consecutive DIM values found, falling back to original matching strategy")
            # Fall back to original strategy
            for dim_value, count in mastitis_dim_counts.items():
                healthy_with_dim = healthy_group[healthy_group['DIM'] == dim_value]
                if len(healthy_with_dim) >= count:
                    sampled = healthy_with_dim.sample(n=count, random_state=random_state)
                else:
                    sampled = healthy_with_dim
                matched_healthy.append(sampled)
    
    else:
        # Original single-day matching strategy
        for dim_value, count in mastitis_dim_counts.items():
            healthy_with_dim = healthy_group[healthy_group['DIM'] == dim_value]
            if len(healthy_with_dim) >= count:
                sampled = healthy_with_dim.sample(n=count, random_state=random_state)
            else:
                sampled = healthy_with_dim
            matched_healthy.append(sampled)
    
    if matched_healthy:
        return pd.concat(matched_healthy, ignore_index=True)
    else:
        return pd.DataFrame()
    
def analyze_data_distribution(data, time_point, is_window=False, window_start=None, window_end=None):
    """
    Analyze data distribution
    """
    mastitis_data = data[data['group'] == 'mastitis']
    healthy_data = data[data['group'] == 'healthy']
    
    # Calculate DIM distribution
    mastitis_dim_stats = {
        'mean': mastitis_data['DIM'].mean(),
        'std': mastitis_data['DIM'].std(),
        'min': mastitis_data['DIM'].min(),
        'max': mastitis_data['DIM'].max(),
        'count': len(mastitis_data)
    }
    
    healthy_dim_stats = {
        'mean': healthy_data['DIM'].mean(),
        'std': healthy_data['DIM'].std(),
        'min': healthy_data['DIM'].min(),
        'max': healthy_data['DIM'].max(),
        'count': len(healthy_data)
    }
    
    # Calculate missing values (if time series data)
    missing_info = {}
    if is_window and window_start is not None and window_end is not None:
        # Two-stage strategy: count missing values at each time point within window
        total_cows = len(data['ANIMAL_SOURCE_ID'].unique())
        for interval in range(window_start, window_end + 1):
            interval_data = data[data['interval'] == interval]
            present_cows = len(interval_data['ANIMAL_SOURCE_ID'].unique())
            missing_count = total_cows - present_cows
            missing_info[f'missing_day_{interval}'] = missing_count
    
    return {
        'mastitis_dim_stats': mastitis_dim_stats,
        'healthy_dim_stats': healthy_dim_stats,
        'missing_info': missing_info,
        'time_point_info': f"{window_start}-{window_end}" if is_window else str(time_point)
    }

def fill_missing_by_group_dim_parity(data, feature_cols):
    """
    Unified function to fill missing values grouped by group+dim+parity
    """
    print(f"Filling missing values for {len(feature_cols)} features grouped by group+dim+parity...")
    
    # Create data copy
    data_filled = data.copy()
    
    # Create grouping key: group_dim_parity
    data_filled['group_dim_parity_key'] = (
        data_filled['group'].astype(str) + '_' + 
        data_filled['DIM'].astype(str) + '_' + 
        data_filled['LACTATION_NO'].astype(str)
    )
    
    filled_count = 0
    total_missing = 0
    
    for col in feature_cols:
        if col not in data_filled.columns:
            continue
            
        col_values = data_filled[col].values
        missing_mask = np.isnan(col_values)
        col_missing_count = missing_mask.sum()
        total_missing += col_missing_count
        
        if col_missing_count > 0:
            print(f"  {col}: {col_missing_count} missing values")
            
            # Calculate median by group
            group_medians = {}
            for key in data_filled['group_dim_parity_key'].unique():
                group_data = data_filled[data_filled['group_dim_parity_key'] == key]
                group_values = group_data[col].values
                if len(group_values) > 0 and not np.isnan(group_values).all():
                    group_medians[key] = np.nanmedian(group_values)
            
            # Global median as fallback
            global_median = np.nanmedian(col_values)
            
            # Fill missing values
            for i, is_missing in enumerate(missing_mask):
                if is_missing:
                    key = data_filled.iloc[i]['group_dim_parity_key']
                    if key in group_medians:
                        col_values[i] = group_medians[key]
                        filled_count += 1
                    else:
                        col_values[i] = global_median
                        filled_count += 1
            
            data_filled[col] = col_values
    
    # Clean up auxiliary columns
    data_filled = data_filled.drop(columns=['group_dim_parity_key'])
    
    print(f"  Total filled {filled_count}/{total_missing} missing values")
    return data_filled

def create_enhanced_cow_sequences_for_two_stage(mastitis_data, healthy_data, spectral_cols=None, milk_composition_cols=None, 
                                               scaler_type='minmax', max_interval=30, min_interval=0,
                                               content_scaler=None, metadata_scaler=None):
    """
    Create enhanced cow-level time series data for two-stage modeling - specifically handles DIM matching for healthy and mastitis cows
    
    Args:
        mastitis_data: Mastitis cow data (with time series)
        healthy_data: Healthy cow data (single sampling)
        max_interval: Maximum time interval (e.g., 30 days ago)
        min_interval: Minimum time interval (e.g., diagnosis day)
    """
    cow_sequences = {}
    all_features_list = []
    
    # Define complete timeline
    full_timeline = list(range(max_interval, min_interval - 1, -1))  # [30, 29, 28, ..., 1, 0]
    timeline_length = len(full_timeline)
    
    print(f"Complete timeline: {max_interval} days ago to {min_interval} days ago, {timeline_length} time points")
    
    # 1. Process mastitis cow time series data
    print("Processing mastitis cow time series data...")
    mastitis_cow_sequences = {}
    
    # First collect all mastitis data for unified standardization
    for cow_id in mastitis_data['ANIMAL_SOURCE_ID'].unique():
        cow_data = mastitis_data[mastitis_data['ANIMAL_SOURCE_ID'] == cow_id].copy()
        cow_data = cow_data.sort_values('DATE_KEY')
        
        if len(cow_data) >= 1:
            all_features_list.append(cow_data)
    
    # Merge all data for unified standardization
    all_data_combined = pd.concat(all_features_list, ignore_index=True)
    print(f"  Mastitis cow unified standardization, sample count: {len(all_data_combined)}")
    
    if content_scaler is None or metadata_scaler is None:
        # Training set: create new scaler
        print(f"    Creating new scaler (training set)")
        content_scaled, metadata_scaled, content_scaler, metadata_scaler = prepare_enhanced_features(
            all_data_combined, spectral_cols, milk_composition_cols, scaler_type
        )
    else:
        # Test set: use training set scaler
        print(f"    Using training set scaler (test set)")
        content_scaled, metadata_scaled, _, _ = prepare_enhanced_features(
            all_data_combined, spectral_cols, milk_composition_cols, scaler_type,
            content_scaler=content_scaler, metadata_scaler=metadata_scaler
        )
    
    # Create complete timeline sequences for each mastitis cow
    start_idx = 0
    for cow_id in mastitis_data['ANIMAL_SOURCE_ID'].unique():
        cow_data = mastitis_data[mastitis_data['ANIMAL_SOURCE_ID'] == cow_id].copy()
        cow_data = cow_data.sort_values('DATE_KEY')
        
        if len(cow_data) >= 1:
            end_idx = start_idx + len(cow_data)
            
            # Get existing time points and features for this cow
            existing_intervals = cow_data['interval'].values
            existing_content = content_scaled[start_idx:end_idx]
            existing_metadata = metadata_scaled[start_idx:end_idx]
            existing_groups = cow_data['group'].values
            existing_dates = cow_data['DATE_KEY'].values
            
            # Create complete timeline arrays
            content_dim = existing_content.shape[1]
            metadata_dim = existing_metadata.shape[1]
            
            # Initialize complete sequences (filled with 0)
            full_content = np.zeros((timeline_length, content_dim))
            full_metadata = np.zeros((timeline_length, metadata_dim))
            full_groups = np.array(['missing'] * timeline_length)
            full_dates = np.array([f'missing_day_{interval}' for interval in full_timeline])
            full_masks = np.ones(timeline_length, dtype=bool)  # True=missing/filled, False=real data
            
            # Fill existing data into corresponding time positions
            for i, interval in enumerate(existing_intervals):
                if interval in full_timeline:
                    timeline_idx = full_timeline.index(interval)
                    full_content[timeline_idx] = existing_content[i]
                    full_metadata[timeline_idx] = existing_metadata[i]
                    full_groups[timeline_idx] = existing_groups[i]
                    full_dates[timeline_idx] = existing_dates[i]
                    full_masks[timeline_idx] = False  # False=real data
            
            # Fill metadata for missing days using mean of existing data
            missing_count = full_masks.sum()
            if missing_count > 0:
                existing_metadata_mean = existing_metadata.mean(axis=0)
                missing_indices = np.where(full_masks)[0]
                for idx in missing_indices:
                    full_metadata[idx] = existing_metadata_mean
            
            sequence_data = {
                'dates': full_dates,
                'intervals': np.array(full_timeline),
                'content': full_content,
                'metadata': full_metadata,
                'group': full_groups,
                'missing_mask': full_masks,
                'real_data_count': (~full_masks).sum(),
                'cow_type': 'mastitis'
            }
            
            mastitis_cow_sequences[cow_id] = sequence_data
            start_idx = end_idx
    
    # 2. Process healthy cow data - based on DIM matching, use real time series
    print("Processing healthy cow data (based on DIM matching, using time series)...")
    
    # Get DIM distribution of mastitis cows
    mastitis_dims = []
    for cow_id, seq_data in mastitis_cow_sequences.items():
        # Get DIM values of real data points
        real_indices = ~seq_data['missing_mask']
        if real_indices.any():
            # Find corresponding DIM values (need to get from original data)
            cow_data = mastitis_data[mastitis_data['ANIMAL_SOURCE_ID'] == cow_id]
            for i, interval in enumerate(seq_data['intervals']):
                if not seq_data['missing_mask'][i]:
                    interval_data = cow_data[cow_data['interval'] == interval]
                    if len(interval_data) > 0:
                        mastitis_dims.append(interval_data.iloc[0]['DIM'])
    
    mastitis_dim_counts = pd.Series(mastitis_dims).value_counts()
    #print(f"Mastitis cow DIM distribution: {mastitis_dim_counts}")
    
    # Match healthy cows for each DIM value, select healthy cows with time series
    matched_healthy_cows = []
    
    # Get DIM range of mastitis cows within time window
    mastitis_dims_in_window = mastitis_data['DIM'].values
    dim_min = mastitis_dims_in_window.min()
    dim_max = mastitis_dims_in_window.max()
    
    print(f"Mastitis cow DIM range: {dim_min}-{dim_max}")
    
    # Find healthy cows with consecutive DIM values
    healthy_cow_ids = healthy_data['ANIMAL_SOURCE_ID'].unique()
    suitable_healthy_cows = []
    
    for cow_id in healthy_cow_ids:
        cow_data = healthy_data[healthy_data['ANIMAL_SOURCE_ID'] == cow_id]
        cow_dims = sorted(cow_data['DIM'].values)
        
        # Check if there are consecutive DIM values within mastitis cow DIM range
        for i in range(len(cow_dims) - 4):  # Need at least 5 consecutive points
            consecutive_dims = cow_dims[i:i+5]  # Take 5 consecutive DIM values
            
            # Check if all 5 DIM values are within mastitis cow DIM range
            if (all(dim_min <= dim <= dim_max for dim in consecutive_dims) and
                consecutive_dims[-1] - consecutive_dims[0] <= 20):  # DIM span no more than 20 days
                
                # Select these 5 consecutive data points
                selected_data = []
                for dim in consecutive_dims:
                    dim_data = cow_data[cow_data['DIM'] == dim]
                    if len(dim_data) > 0:
                        selected_data.append(dim_data.iloc[0])
                
                if len(selected_data) == 5:  # Ensure complete 5 points
                    suitable_healthy_cows.append(pd.DataFrame(selected_data))
                    break  # One set is enough
    
    # Select appropriate number of healthy cows
    if len(suitable_healthy_cows) > 0:
        # Calculate how many healthy cows needed to match mastitis cow count
        n_mastitis_cows = len(mastitis_cow_sequences)
        n_healthy_needed = min(len(suitable_healthy_cows), n_mastitis_cows)
        
        # Select indices instead of DataFrame
        selected_indices = np.random.choice(
            range(len(suitable_healthy_cows)), 
            size=n_healthy_needed, 
            replace=False
        )
        
        for idx in selected_indices:
            matched_healthy_cows.append(suitable_healthy_cows[idx])
        
        print(f"Found {len(suitable_healthy_cows)} healthy cows with consecutive DIM values, selected {n_healthy_needed}")
    else:
        print("Warning: No healthy cows with consecutive DIM values found, falling back to original matching strategy")
        # Fall back to original strategy
        for dim_value, count in mastitis_dim_counts.items():
            # Find healthy cows with this DIM value
            healthy_with_dim = healthy_data[healthy_data['DIM'] == dim_value]
            
            # Check if these healthy cows have time series
            healthy_cow_ids_with_dim = healthy_with_dim['ANIMAL_SOURCE_ID'].unique()
            healthy_cows_with_sequences = []
            
            for cow_id in healthy_cow_ids_with_dim:
                cow_data = healthy_data[healthy_data['ANIMAL_SOURCE_ID'] == cow_id]
                if len(cow_data) > 1:  # Healthy cows with time series
                    healthy_cows_with_sequences.append(cow_id)
            
            # Select from healthy cows with time series
            if len(healthy_cows_with_sequences) >= count:
                selected_cow_ids = np.random.choice(healthy_cows_with_sequences, size=count, replace=False)
            else:
                selected_cow_ids = healthy_cows_with_sequences
            
            # Get complete time series data for selected healthy cows
            for cow_id in selected_cow_ids:
                cow_data = healthy_data[healthy_data['ANIMAL_SOURCE_ID'] == cow_id].copy()
                cow_data = cow_data.sort_values('DATE_KEY')
                matched_healthy_cows.append(cow_data)
    
    if matched_healthy_cows:
        # Merge all healthy cow data for unified standardization
        all_healthy_data = pd.concat(matched_healthy_cows, ignore_index=True)
        print(f"Matched healthy cow count: {len(matched_healthy_cows)}")
        print(f"Healthy cow total sample count: {len(all_healthy_data)}")
        
        # Create time series data for healthy cows - use same scaler as mastitis cows
        healthy_content_scaled, healthy_metadata_scaled, _, _ = prepare_enhanced_features(
            all_healthy_data, spectral_cols, milk_composition_cols, scaler_type,
            content_scaler=content_scaler, metadata_scaler=metadata_scaler
        )
        
        # Create time series for each healthy cow
        start_idx = 0
        for cow_data in matched_healthy_cows:
            cow_id = cow_data['ANIMAL_SOURCE_ID'].iloc[0]
            end_idx = start_idx + len(cow_data)
            
            # Get time series data for this cow
            cow_content = healthy_content_scaled[start_idx:end_idx]
            cow_metadata = healthy_metadata_scaled[start_idx:end_idx]
            cow_dates = cow_data['DATE_KEY'].values
            cow_dims = cow_data['DIM'].values
            
            # Create complete timeline arrays
            content_dim = cow_content.shape[1]
            metadata_dim = cow_metadata.shape[1]
            
            # Initialize complete sequences (filled with 0)
            full_content = np.zeros((timeline_length, content_dim))
            full_metadata = np.zeros((timeline_length, metadata_dim))
            full_groups = np.array(['missing'] * timeline_length)
            full_dates = np.array([f'missing_day_{interval}' for interval in full_timeline])
            full_masks = np.ones(timeline_length, dtype=bool)  # True=missing/filled, False=real data
            
            # Map healthy cow time series data to timeline
            # Since healthy cows don't have interval information, we need to infer time position from DIM
            # Map healthy cow DIM values to different positions on timeline
            for i, dim in enumerate(cow_dims):
                # Correction: use sequential mapping since we've ensured DIM value continuity
                # Directly map the i-th DIM value to the i-th time position
                timeline_idx = i
                
                # Ensure index is within valid range
                timeline_idx = max(0, min(timeline_idx, timeline_length - 1))
                
                # For consecutive DIM values, place directly in order
                # Since we've ensured DIM value continuity
                full_content[timeline_idx] = cow_content[i]
                full_metadata[timeline_idx] = cow_metadata[i]
                full_groups[timeline_idx] = 'healthy'
                full_dates[timeline_idx] = cow_dates[i]
                full_masks[timeline_idx] = False  # False=real data
            
            # Fill metadata for missing days using mean of existing data
            missing_count = full_masks.sum()
            if missing_count > 0 and missing_count < timeline_length:
                existing_metadata_mean = cow_metadata.mean(axis=0)
                missing_indices = np.where(full_masks)[0]
                for idx in missing_indices:
                    full_metadata[idx] = existing_metadata_mean
            
            sequence_data = {
                'dates': full_dates,
                'intervals': np.array(full_timeline),
                'content': full_content,
                'metadata': full_metadata,
                'group': full_groups,
                'missing_mask': full_masks,
                'real_data_count': (~full_masks).sum(),
                'cow_type': 'healthy'
            }
            
            cow_sequences[f"healthy_{cow_id}"] = sequence_data
            start_idx = end_idx
    
    # 3. Add mastitis cow sequences
    cow_sequences.update(mastitis_cow_sequences)
    
    print(f"Created complete timeline sequence data for {len(cow_sequences)} cows")
    print(f"  Mastitis cows: {len(mastitis_cow_sequences)}, Healthy cows: {len(cow_sequences) - len(mastitis_cow_sequences)}")
    
    return cow_sequences, content_scaler, metadata_scaler

def prepare_enhanced_features(data, spectral_cols, milk_composition_cols, scaler_type='minmax', 
                             content_scaler=None, metadata_scaler=None):
    """
    Unified feature preprocessing: includes dim, parity, milkweightlbs, cells, spectral/milk composition
    All missing values are filled grouped by group+dim+parity
    """
    print("Starting unified feature preprocessing...")
    
    # 1. Define feature columns to fill
    features_to_fill = ['TOTAL_YIELD']
    
    if spectral_cols:
        features_to_fill.extend(spectral_cols)
        content_cols = spectral_cols
        content_type = "spectral"
    elif milk_composition_cols:
        available_comp_cols = [col for col in milk_composition_cols if col in data.columns]
        features_to_fill.extend(available_comp_cols)
        content_cols = available_comp_cols
        content_type = "composition"
    else:
        raise ValueError("Must provide either spectral_cols or milk_composition_cols")
    
    # 2. Unified missing value filling (grouped by group+dim+parity)
    data_filled = fill_missing_by_group_dim_parity(data, features_to_fill)
    
    # 3. Parity re-encoding into 3 groups + One-hot encoding
    def encode_parity(parity_value):
        if parity_value == 1:
            return [1, 0, 0]  # First parity
        elif parity_value == 2:
            return [0, 1, 0]  # Second parity
        else:  # 3 and above
            return [0, 0, 1]  # Third parity and above
    
    # Apply parity encoding
    parity_encoded = np.array([encode_parity(p) for p in data_filled['LACTATION_NO']])
    
    # 4. Combine metadata features
    metadata_features = np.column_stack([
        data_filled['DIM'].values.reshape(-1, 1),           # DIM
        data_filled['TOTAL_YIELD'].values.reshape(-1, 1), # Milk weight (filled)
        parity_encoded                                       # Parity one-hot (3D)
    ])
    
    # 5. Get content features (filled)
    content_features = data_filled[content_cols].values
    
    # 6. Standardization
    if content_scaler is None or metadata_scaler is None:
        # Training set: create new scaler
        if scaler_type == 'minmax':
            metadata_scaler = MinMaxScaler(feature_range=(0, 1))
            content_scaler = MinMaxScaler(feature_range=(0, 1))
            print(f"Using MinMaxScaler, feature range: [0, 1]")
        elif scaler_type == 'standard':
            metadata_scaler = StandardScaler()
            content_scaler = StandardScaler()
            print(f"Using StandardScaler, feature range: mean 0, std 1")
        else:
            raise ValueError("scaler_type must be 'minmax' or 'standard'")
        
        # Training set: fit_transform
        metadata_scaled = metadata_scaler.fit_transform(metadata_features)
        content_scaled = content_scaler.fit_transform(content_features)
    else:
        # Test set: transform using training set scaler
        metadata_scaled = metadata_scaler.transform(metadata_features)
        content_scaled = content_scaler.transform(content_features)
    
    print(f"{content_type} data dimension: {content_scaled.shape[1]}")
    print(f"Metadata dimension: {metadata_scaled.shape[1]} (DIM(1) + MilkWeight(1) + Parity(3))")
    
    return content_scaled, metadata_scaled, content_scaler, metadata_scaler

class EnhancedSingleDayDataset(Dataset):
    """
    Enhanced single-day dataset
    """
    def __init__(self, data, spectral_cols=None, milk_composition_cols=None, scaler_type='minmax', 
                 content_scaler=None, metadata_scaler=None):
        self.features = []
        self.labels = []
        
        # Prepare features - use provided scaler or create new one
        if content_scaler is None or metadata_scaler is None:
            # Training set: create new scaler
            print(f"  Creating new scaler (training set)")
            content_features, metadata_features, content_scaler, metadata_scaler = prepare_enhanced_features(
                data, spectral_cols, milk_composition_cols, scaler_type
            )
        else:
            # Test set: use training set scaler
            print(f"  Using training set scaler (test set)")
            content_features, metadata_features, _, _ = prepare_enhanced_features(
                data, spectral_cols, milk_composition_cols, scaler_type, 
                content_scaler=content_scaler, metadata_scaler=metadata_scaler
            )
        
        all_features = np.column_stack([metadata_features, content_features])
        
        for idx, row in data.iterrows():
            label = 1 if row['group'] == 'mastitis' else 0
            self.features.append(all_features[data.index.get_loc(idx)])
            self.labels.append(label)
        
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        
        # Save scaler for test set use
        self.content_scaler = content_scaler
        self.metadata_scaler = metadata_scaler
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.LongTensor([self.labels[idx]])
    
class SingleDayTransformer(nn.Module):
    """
    Single-day Transformer model - consistent with inner layer structure of two-stage strategy
    """
    def __init__(self, input_dim, d_model=64, nhead=8, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # Consistent with SpectralEncoder
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        x = self.input_projection(x.unsqueeze(1))  # [batch, 1, d_model]
        x = self.transformer(x)  # [batch, 1, d_model]
        x = x.squeeze(1)  # [batch, d_model]
        return self.classifier(x)

class SingleDayLSTM(nn.Module):
    """
    Single-day LSTM model - parameter count balanced with Transformer
    """
    def __init__(self, input_dim, hidden_dim=96, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 2)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, input_dim]
        lstm_out, _ = self.lstm(x)  # [batch, 1, hidden_dim]
        x = lstm_out.squeeze(1)  # [batch, hidden_dim]
        return self.classifier(x)
    
class EnhancedTimeSeriesDataset(Dataset):
    """
    Enhanced time series dataset - supports complete timeline
    """
    def __init__(self, cow_sequences, min_real_data_points=3):
        """
        Args:
            min_real_data_points: Minimum number of real data points required per cow
        """
        self.content_sequences = []     
        self.metadata_sequences = []    
        self.interval_sequences = []    
        self.missing_masks = []         # New: missing data mask
        self.labels = []               
        self.cow_ids = []             
        
        filtered_count = 0
        for cow_id, seq_data in cow_sequences.items():
            # Check real data point count
            if seq_data['real_data_count'] >= min_real_data_points:
                self.content_sequences.append(seq_data['content'])
                self.metadata_sequences.append(seq_data['metadata'])
                self.interval_sequences.append(seq_data['intervals'])
                self.missing_masks.append(seq_data['missing_mask'])
                
                # Generate cow-level labels (based on real data points, excluding filled 'missing')
                real_groups = seq_data['group'][~seq_data['missing_mask']]  # Only take real data groups
                label = 1 if len(real_groups) > 0 and any('mastiti' in g for g in real_groups) else 0
                
                # # Debug: output first few label generation cases
                # if len(self.labels) < 5:  # Only output first 5
                #     print(f"  Debug label generation - cow_id: {cow_id}, real_groups: {real_groups}, label: {label}")
                self.labels.append(label)
                self.cow_ids.append(cow_id)
            else:
                filtered_count += 1
        
        print(f"Filtered out {filtered_count} cows (real data points < {min_real_data_points})")
        print(f"Retained {len(self.content_sequences)} cows for training")
        
        # Debug: check label distribution
        if len(self.labels) > 0:
            label_counts = pd.Series(self.labels).value_counts()
            print(f"Label distribution: {label_counts}")
            print(f"Mastitis cow count: {label_counts.get(1, 0)}, Healthy cow count: {label_counts.get(0, 0)}")
        else:
            print("Warning: No valid labels!")
    
    def __len__(self):
        return len(self.content_sequences)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.content_sequences[idx]), 
            torch.FloatTensor(self.metadata_sequences[idx]),
            torch.FloatTensor(self.interval_sequences[idx]),
            torch.BoolTensor(self.missing_masks[idx]),      # New: missing mask
            torch.LongTensor([self.labels[idx]]), 
            self.cow_ids[idx]
        )
    
class SpectralEncoder(nn.Module):
    """
    Stage 1: Spectral encoder
    """
    def __init__(self, spectral_dim, embedding_dim=64, encoder_type='transformer'):
        super().__init__()
        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim
        
        if encoder_type == 'mlp':
            self.encoder = nn.Sequential(
                nn.Linear(spectral_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 96),
                nn.ReLU(), 
                nn.Dropout(0.1),
                nn.Linear(96, embedding_dim),
                nn.ReLU()
            )
        
        elif encoder_type == '1dcnn':
            self.conv_layers = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            self.fc = nn.Linear(64, embedding_dim)
            
        elif encoder_type == 'transformer':
            self.input_projection = nn.Linear(spectral_dim, embedding_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=embedding_dim * 2,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
            self.pooling = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, x):
        if self.encoder_type == 'mlp':
            return self.encoder(x)
            
        elif self.encoder_type == '1dcnn':
            x = x.unsqueeze(1)
            x = self.conv_layers(x)
            x = x.squeeze(-1)
            return self.fc(x)
            
        elif self.encoder_type == 'transformer':
            x = self.input_projection(x.unsqueeze(1))
            x = self.transformer(x)
            return x.squeeze(1)

class LSTMSpectralEncoder(nn.Module):
    """
    LSTM-based spectral encoder
    """
    def __init__(self, spectral_dim, embedding_dim=64):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Use LSTM with similar parameter count
        # Transformer has ~25K parameters, LSTM uses similar parameter count
        # Calculation: spectral_dim * hidden_size * 4 (bidirectional) * num_layers + hidden_size * 2
        hidden_size = 64  # Increase hidden size to match Transformer parameter count
        self.lstm = nn.LSTM(
            input_size=spectral_dim,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True  # hidden_size*2 dimensional output
        )
        
        # Final projection layer ensures consistent output dimension
        self.output_projection = nn.Linear(hidden_size * 2, embedding_dim)
    
    def forward(self, x):
        # x: [batch, spectral_dim]
        x = x.unsqueeze(1)  # [batch, 1, spectral_dim] 
        lstm_out, _ = self.lstm(x)  # [batch, 1, 64]
        x = lstm_out.squeeze(1)  # [batch, 64]
        x = self.output_projection(x)  # [batch, embedding_dim]
        return x
    
class EnhancedTwoStageModel(nn.Module):
    """
    Enhanced two-stage model - supports outer Transformer with Positional Embedding
    """
    def __init__(self, spectral_dim, metadata_dim, spectral_embedding_dim=64, 
                 spectral_encoder_type='transformer', temporal_model_type='transformer'):
        super().__init__()
        
        # Stage 1: spectral encoder (support transformer and lstm)
        if spectral_encoder_type == 'lstm':
            self.spectral_encoder = LSTMSpectralEncoder(
                spectral_dim=spectral_dim, embedding_dim=spectral_embedding_dim
            )
        else:
            # Default to SpectralEncoder for transformer/mlp/1dcnn
            self.spectral_encoder = SpectralEncoder(
                spectral_dim, spectral_embedding_dim, spectral_encoder_type
            )
        
        # Fusion dimension
        self.fusion_dim = spectral_embedding_dim + metadata_dim
        
        # Ensure projection attribute always exists for both branches
        self.projection = None
        
        if temporal_model_type == 'transformer':
            # Ensure fusion_dim divisible by nhead=8
            nhead = 8
            if self.fusion_dim % nhead != 0:
                self.fusion_dim = ((self.fusion_dim // nhead) + 1) * nhead
                self.projection = nn.Linear(spectral_embedding_dim + metadata_dim, self.fusion_dim)
            else:
                self.projection = None
            
            # Positional Encoding
            self.max_seq_len = 100
            self.pos_embedding = nn.Parameter(torch.randn(1, self.max_seq_len, self.fusion_dim))
            
            # Transformer temporal encoder (parameter controlled)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.fusion_dim,
                nhead=nhead,
                dim_feedforward=self.fusion_dim,
                dropout=0.1,
                batch_first=True
            )
            self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            
            # Temporal attention for pooling
            self.temporal_attention = nn.Linear(self.fusion_dim, 1)
            classifier_input_dim = self.fusion_dim
        else:
            # LSTM temporal encoder (parameter balanced with transformer)
            # Calculate parameter count similar to Transformer
            # Transformer: fusion_dim * fusion_dim * 2 (attention) + fusion_dim * fusion_dim (ffn) + fusion_dim * 2 (layers)
            # LSTM: fusion_dim * lstm_hidden * 4 (bidirectional) * num_layers + lstm_hidden * 2
            lstm_hidden = 48  # Increase hidden size
            self.lstm = nn.LSTM(
                input_size=self.fusion_dim,
                hidden_size=lstm_hidden,
                num_layers=2,
                batch_first=True,
                dropout=0.1,
                bidirectional=True
            )
            self.temporal_attention = nn.Linear(lstm_hidden * 2, 1)
            classifier_input_dim = lstm_hidden * 2
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )
        
        self.temporal_model_type = temporal_model_type
    
    def forward(self, spectral_sequences, metadata_sequences, mask=None, return_attention=False):
        batch_size, seq_len, spectral_dim = spectral_sequences.shape
        
        # Stage 1: Spectral encoding
        spectral_flat = spectral_sequences.view(-1, spectral_dim)
        spectral_embeddings_flat = self.spectral_encoder(spectral_flat)
        spectral_embeddings = spectral_embeddings_flat.view(batch_size, seq_len, -1)
        
        # Stage 2: Temporal modeling
        if self.temporal_model_type == 'transformer':
            # Feature fusion
            fused_features = torch.cat([spectral_embeddings, metadata_sequences], dim=-1)
            
            # If needed, perform dimension projection
            if self.projection is not None:
                fused_features = self.projection(fused_features)
            
            # Add positional encoding
            if seq_len <= self.max_seq_len:
                pos_emb = self.pos_embedding[:, :seq_len, :]
                fused_features = fused_features + pos_emb
            
            # Transformer encoding
            if mask is not None:
                attn_mask = (mask == 0)
            else:
                attn_mask = None
            
            temporal_out = self.temporal_transformer(fused_features, src_key_padding_mask=attn_mask)
            
            # Attention pooling
            if mask is not None:
                attn_scores = self.temporal_attention(temporal_out).squeeze(-1)
                attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
                attn_weights = torch.softmax(attn_scores, dim=1)
                pooled = torch.sum(temporal_out * attn_weights.unsqueeze(-1), dim=1)
            else:
                attn_scores = self.temporal_attention(temporal_out).squeeze(-1)
                attn_weights = torch.softmax(attn_scores, dim=1)
                pooled = torch.sum(temporal_out * attn_weights.unsqueeze(-1), dim=1)
            
            logits = self.classifier(pooled)
            
            if return_attention:
                return logits, attn_weights, temporal_out
            else:
                return logits
        
        else:  # LSTM
            # Feature fusion
            fused_features = torch.cat([spectral_embeddings, metadata_sequences], dim=-1)
            
            # If needed, perform dimension projection
            if self.projection is not None:
                fused_features = self.projection(fused_features)
            
            # LSTM temporal modeling
            lstm_out, _ = self.lstm(fused_features)
            
            # Attention pooling
            if mask is not None:
                attn_scores = self.temporal_attention(lstm_out).squeeze(-1)
                attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
                attn_weights = torch.softmax(attn_scores, dim=1)
                pooled = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)
            else:
                attn_scores = self.temporal_attention(lstm_out).squeeze(-1)
                attn_weights = torch.softmax(attn_scores, dim=1)
                pooled = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)
            
            logits = self.classifier(pooled)
            
            if return_attention:
                return logits, attn_weights, lstm_out
            else:
                return logits

def train_and_evaluate_single_day(model, train_loader, test_loader, device, num_epochs=100, class_weights=None):
    """
    Train and evaluate single-day model
    """
    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training
    model.train()
    for epoch in range(num_epochs):
        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.squeeze().to(device)
            
            # Ensure labels are 1D tensor
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            
            # Skip empty batches
            if features.size(0) == 0 or labels.size(0) == 0:
                continue
            
            # Ensure batch sizes match
            if features.size(0) != labels.size(0):
                min_size = min(features.size(0), labels.size(0))
                features = features[:min_size]
                labels = labels[:min_size]
            
            if features.size(0) == 0:
                continue
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # Evaluation
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.squeeze().to(device)
            
            # Ensure labels are 1D tensor
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            
            # Skip empty batches
            if features.size(0) == 0 or labels.size(0) == 0:
                continue
            
            # Ensure batch sizes match
            if features.size(0) != labels.size(0):
                min_size = min(features.size(0), labels.size(0))
                features = features[:min_size]
                labels = labels[:min_size]
            
            if features.size(0) == 0:
                continue
            
            outputs = model(features)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    if len(set(all_labels)) > 1 and len(all_labels) > 0:
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        auc = roc_auc_score(all_labels, np.array(all_probabilities)[:, 1])
        return accuracy, precision, recall, auc
    else:
        return 0.0, 0.0, 0.0, 0.0
    
def debug_two_stage_data(train_loader, test_loader):
    """
    Debug two-stage data flow
    """
    print("\n=== Debug Two-Stage Data Flow ===")
    
    # Check training data
    print("Check training data:")
    train_batch_count = 0
    for batch_idx, batch_data in enumerate(train_loader):
        if batch_idx >= 2:  # Only check first 2 batches
            break
        
        content_seq, metadata_seq, interval_seq, masks, labels, cow_ids = batch_data
        print(f"  Batch {batch_idx}:")
        print(f"    content_seq shape: {content_seq.shape}")
        print(f"    metadata_seq shape: {metadata_seq.shape}")
        print(f"    masks shape: {masks.shape}")
        print(f"    labels shape: {labels.shape}")
        print(f"    labels values: {labels}")
        print(f"    mask example: {masks[0] if len(masks) > 0 else 'empty'}")
        train_batch_count += 1
    
    print(f"Training data total batch count: {len(list(train_loader))}")
    
    # Check test data
    print("\nCheck test data:")
    test_batch_count = 0
    for batch_idx, batch_data in enumerate(test_loader):
        if batch_idx >= 2:  # Only check first 2 batches
            break
        
        content_seq, metadata_seq, interval_seq, masks, labels, cow_ids = batch_data
        print(f"  Batch {batch_idx}:")
        print(f"    content_seq shape: {content_seq.shape}")
        print(f"    labels values: {labels}")
        test_batch_count += 1
    
    print(f"Test data total batch count: {len(list(test_loader))}")

def train_and_evaluate_two_stage(model, train_loader, test_loader, device, num_epochs=100, val_split=0.2, class_weights=None):
    """
    Train and evaluate two-stage model - includes validation loop
    """
    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    epoch_losses = []  # Record training loss for each epoch
    val_losses = []    # Record validation loss for each epoch
    
    # Split training data into train and validation sets
    train_data = []
    for batch in train_loader:
        train_data.append(batch)
    
    # Randomly split training data into train and validation sets
    train_indices = list(range(len(train_data)))
    if len(train_indices) > 1:
        train_idx, val_idx = train_test_split(train_indices, test_size=val_split, random_state=42)
        train_batches = [train_data[i] for i in train_idx]
        val_batches = [train_data[i] for i in val_idx]
    else:
        train_batches = train_data
        val_batches = train_data  # If data is too small, use training set as validation set
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for content_seq, metadata_seq, interval_seq, masks, labels, cow_ids in train_batches:
            content_seq = content_seq.to(device)
            metadata_seq = metadata_seq.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            # Skip empty batches or single-sample batches
            if content_seq.size(0) <= 1 or labels.size(0) == 0:
                continue
            
            # Ensure batch sizes match
            if content_seq.size(0) != labels.size(0):
                min_size = min(content_seq.size(0), labels.size(0))
                content_seq = content_seq[:min_size]
                metadata_seq = metadata_seq[:min_size]
                masks = masks[:min_size]
                labels = labels[:min_size]
            
            if content_seq.size(0) <= 1:
                continue
            
            optimizer.zero_grad()
            outputs = model(content_seq, metadata_seq, masks)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        # Record average training loss
        if num_batches > 0:
            avg_epoch_loss = epoch_loss / num_batches
            epoch_losses.append(avg_epoch_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_num_batches = 0
        
        with torch.no_grad():
            for content_seq, metadata_seq, interval_seq, masks, labels, cow_ids in val_batches:
                content_seq = content_seq.to(device)
                metadata_seq = metadata_seq.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                
                # Skip empty batches or single-sample batches
                if content_seq.size(0) <= 1 or labels.size(0) == 0:
                    continue
                
                # Ensure batch sizes match
                if content_seq.size(0) != labels.size(0):
                    min_size = min(content_seq.size(0), labels.size(0))
                    content_seq = content_seq[:min_size]
                    metadata_seq = metadata_seq[:min_size]
                    masks = masks[:min_size]
                    labels = labels[:min_size]
                
                if content_seq.size(0) <= 1:
                    continue
                
                outputs = model(content_seq, metadata_seq, masks)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_num_batches += 1
        
        # Record average validation loss
        if val_num_batches > 0:
            avg_val_loss = val_loss / val_num_batches
            val_losses.append(avg_val_loss)
        else:
            val_losses.append(0.0)
    
    # Evaluation
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for content_seq, metadata_seq, interval_seq, masks, labels, cow_ids in test_loader:
            content_seq = content_seq.to(device)
            metadata_seq = metadata_seq.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            
            # Skip empty batches
            if content_seq.size(0) == 0 or labels.size(0) == 0:
                continue
            
            # Ensure batch sizes match
            if content_seq.size(0) != labels.size(0):
                min_size = min(content_seq.size(0), labels.size(0))
                content_seq = content_seq[:min_size]
                metadata_seq = metadata_seq[:min_size]
                masks = masks[:min_size]
                labels = labels[:min_size]
            
            if content_seq.size(0) == 0:
                continue
            
            outputs = model(content_seq, metadata_seq, masks)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    if len(set(all_labels)) > 1 and len(all_labels) > 0:
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        auc = roc_auc_score(all_labels, np.array(all_probabilities)[:, 1])
        return accuracy, precision, recall, auc, epoch_losses, val_losses
    else:
        return 0.0, 0.0, 0.0, 0.0, epoch_losses, val_losses

def save_results_to_csv(all_results, data_distributions, detailed_metrics, training_losses, time_window=5, step=1):
    """
    Save results to CSV files
    """
    print(f"\nSaving results to CSV files (window={time_window}, step={step})...")

    # Map internal strategy keys to desired output names
    def map_strategy_output_name(name: str) -> str:
        # Normalize temporal-LSTM variant naming in outputs
        if name in ('two_stage_spectral_transformer_lstm', 'two_stage_spectral_transformer_lstm_temporal'):
            return 'two_stage_spectral_transformer_lstm_temporal'
        return name
    
    # 1. Data distribution and sample count statistics
    distribution_records = []
    for strategy_name, time_points in data_distributions.items():
        for time_point, dist_info in time_points.items():
            record = {
                'strategy': map_strategy_output_name(strategy_name),
                'time_point': time_point,
                'mastitis_count': dist_info['mastitis_dim_stats']['count'],
                'healthy_count': dist_info['healthy_dim_stats']['count'],
                'mastitis_dim_mean': dist_info['mastitis_dim_stats']['mean'],
                'mastitis_dim_std': dist_info['mastitis_dim_stats']['std'],
                'healthy_dim_mean': dist_info['healthy_dim_stats']['mean'],
                'healthy_dim_std': dist_info['healthy_dim_stats']['std'],
            }
            
            # Add missing value information (if any)
            if dist_info['missing_info']:
                for interval, missing_count in dist_info['missing_info'].items():
                    record[f'missing_interval_{interval}'] = missing_count
            
            distribution_records.append(record)
    
    distribution_df = pd.DataFrame(distribution_records)
    distribution_df.to_csv(f'enhanced_data_distributions_w{time_window}_s{step}.csv', index=False)
    
    # 2. Detailed performance metrics (results from each repetition)
    detailed_records = []
    for strategy_name, time_points in detailed_metrics.items():
        for time_point, metrics_list in time_points.items():
            for metrics in metrics_list:
                record = {
                    'strategy': map_strategy_output_name(strategy_name),
                    'time_point': time_point,
                    'iteration': metrics['iteration'],
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'auc': metrics['auc']
                }
                detailed_records.append(record)
    
    detailed_df = pd.DataFrame(detailed_records)
    detailed_df.to_csv(f'enhanced_detailed_metrics_w{time_window}_s{step}.csv', index=False)
    
    # 3. Summary results
    summary_records = []
    for strategy_name, time_points in all_results.items():
        for time_point, results in time_points.items():
            record = {
                'strategy': map_strategy_output_name(strategy_name),
                'time_point': time_point,
                'accuracy_mean': results['accuracy_mean'],
                'accuracy_std': results['accuracy_std'],
                'precision_mean': results['precision_mean'],
                'recall_mean': results['recall_mean'],
                'auc_mean': results['auc_mean']
            }
            summary_records.append(record)
    
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(f'enhanced_summary_results_w{time_window}_s{step}.csv', index=False)
    
    # 4. Training loss and validation loss records (two-stage strategy only)
    loss_records = []
    for strategy_name, time_points in training_losses.items():
        for time_point, iterations in time_points.items():
            for iter_data in iterations:
                iteration = iter_data['iteration']
                epoch_losses = iter_data['epoch_losses']
                val_losses = iter_data.get('val_losses', [])
                
                # Ensure training loss and validation loss have consistent length
                max_epochs = max(len(epoch_losses), len(val_losses))
                for epoch in range(max_epochs):
                    train_loss = epoch_losses[epoch] if epoch < len(epoch_losses) else 0.0
                    val_loss = val_losses[epoch] if epoch < len(val_losses) else 0.0
                    
                    record = {
                        'strategy': map_strategy_output_name(strategy_name),
                        'time_point': time_point,
                        'iteration': iteration,
                        'epoch': epoch + 1,
                        'train_loss': train_loss,
                        'val_loss': val_loss
                    }
                    loss_records.append(record)
    
    if loss_records:
        loss_df = pd.DataFrame(loss_records)
        loss_df.to_csv(f'enhanced_training_losses_w{time_window}_s{step}.csv', index=False)
        print("CSV files saved:")
        print(f"- enhanced_data_distributions_single_w{time_window}_s{step}.csv: Data distribution and sample statistics")
        print(f"- enhanced_detailed_metrics_single_w{time_window}_s{step}.csv: Detailed metrics from each repetition")
        print(f"- enhanced_summary_results_single_w{time_window}_s{step}.csv: Summary results")
        print(f"- enhanced_training_losses_single_w{time_window}_s{step}.csv: Two-stage strategy training loss and validation loss records")
    else:
        print("CSV files saved:")
        print(f"- enhanced_data_distributions_single_w{time_window}_s{step}.csv: Data distribution and sample statistics")
        print(f"- enhanced_detailed_metrics_single_w{time_window}_s{step}.csv: Detailed metrics from each repetition")
        print(f"- enhanced_summary_results_single_w{time_window}_s{step}.csv: Summary results")

def print_enhanced_summary(all_results, model_info=None):
    """
    Print enhanced experiment summary
    
    Args:
        all_results: All experiment results
        model_info: Model information dictionary, containing model size and running time
    """
    print("\n" + "=" * 80)
    print("Enhanced Baseline Experiment Results Summary")
    print("=" * 80)
    
    # 1. Spectral vs milk composition comparison (two-stage modeling)
    print("\n1. Spectral vs Milk Composition Comparison (Two-Stage Modeling):")
    
    # Merge two-stage results from LSTM and Transformer (update: add transformer_lstm_temporal)
    spectral_results = {}
    spectral_results.update(all_results.get('two_stage_spectral_transformer', {}))
    spectral_results.update(all_results.get('two_stage_spectral_transformer_lstm', {}))
    
    composition_results = {}
    composition_results.update(all_results.get('two_stage_composition_transformer_3', {}))
    # Optional: if composition_lstm is added in the future, include it
    if 'two_stage_composition_lstm' in all_results:
        composition_results.update(all_results.get('two_stage_composition_lstm', {}))
    
    if spectral_results and composition_results:
        spectral_avg = np.mean([r['accuracy_mean'] for r in spectral_results.values()])
        composition_avg = np.mean([r['accuracy_mean'] for r in composition_results.values()])
        
        print(f"  Spectral features two-stage:     {spectral_avg:.3f}")
        print(f"  Milk composition two-stage:      {composition_avg:.3f}")
        print(f"  Difference:                      {spectral_avg - composition_avg:+.3f}")
        
        if spectral_avg > composition_avg:
            print("  → Original spectral features perform better in two-stage modeling")
        else:
            print("  → Milk composition features perform better in two-stage modeling")
    
    # 2. Single-day modeling architecture comparison
    print("\n2. Single-Day Modeling Architecture Comparison (Spectral Features):")
    
    single_day_methods = {
        'Transformer': 'single_day_spectral_transformer',
        'LSTM': 'single_day_spectral_lstm',
        'Random Forest': 'single_day_spectral_rf',
        'PLS-DA': 'single_day_spectral_plsda',
        'LDA': 'single_day_spectral_lda'
    }
    
    for method_name, strategy_key in single_day_methods.items():
        if strategy_key in all_results:
            avg_acc = np.mean([r['accuracy_mean'] for r in all_results[strategy_key].values()])
            print(f"  {method_name:15s}: {avg_acc:.3f}")
            
            # Add detailed information for deep learning models
            if method_name in ['Transformer', 'LSTM'] and model_info and strategy_key in model_info:
                info = model_info[strategy_key]
                print(f"    └─ Model size: {info.get('model_size', 'N/A'):>8} parameters")
                print(f"    └─ Running time: {info.get('running_time', 'N/A'):>8} seconds")
    
    # 3. Temporal information value validation
    print("\n3. Temporal Information Value Validation:")
    
    if 'single_day_spectral_transformer' in all_results and spectral_results:
        single_avg = np.mean([r['accuracy_mean'] for r in all_results['single_day_spectral_transformer'].values()])
        two_stage_avg = np.mean([r['accuracy_mean'] for r in spectral_results.values()])
        improvement = two_stage_avg - single_avg
        
        print(f"  Single-day Transformer:    {single_avg:.3f}")
        print(f"  Two-stage spectral:        {two_stage_avg:.3f}")
        print(f"  Temporal information gain: {improvement:+.3f} ({improvement/single_avg*100:+.1f}%)")
    
    # 4. Two-stage model architecture comparison
    print("\n4. Two-Stage Model Architecture Comparison (Spectral Features):")
    
    # Three two-stage models
    two_stage_models = {
        'Transformer-Transformer': 'two_stage_spectral_transformer',
        'Transformer-LSTM': 'two_stage_spectral_transformer_lstm',
        'LSTM-Transformer': 'two_stage_spectral_lstm_transformer'
    }
    
    model_performances = {}
    
    for model_name, strategy_key in two_stage_models.items():
        if strategy_key in all_results and all_results[strategy_key]:
            avg_acc = np.mean([r['accuracy_mean'] for r in all_results[strategy_key].values()])
            model_performances[model_name] = avg_acc
            
            print(f"  {model_name:20s}: {avg_acc:.3f}")
            
            # Add model information
            if model_info and strategy_key in model_info:
                info = model_info[strategy_key]
                print(f"    └─ Model size: {info.get('model_size', 'N/A'):>8} parameters")
                print(f"    └─ Running time: {info.get('running_time', 'N/A'):>8} seconds")
    
    # Find best model
    if model_performances:
        best_model = max(model_performances, key=model_performances.get)
        best_performance = model_performances[best_model]
        print(f"\n  → Best two-stage model: {best_model} ({best_performance:.3f})")
    
    # 5. Overall best method
    print("\n5. Overall Best Method:")
    
    best_method = None
    best_performance = 0
    
    for strategy_name, time_points in all_results.items():
        if time_points:
            avg_performance = np.mean([r['accuracy_mean'] for r in time_points.values()])
            if avg_performance > best_performance:
                best_performance = avg_performance
                best_method = strategy_name
    
    if best_method:
        method_display = best_method.replace('_', ' ').title()
        print(f"  {method_display}: {best_performance:.3f}")
    
    print("\n" + "=" * 80)

def save_results_to_csv_with_farm(all_farm_results, all_farm_data_distributions, all_farm_detailed_metrics, all_farm_training_losses, time_window=5, step=1):
    """
    Save results to CSV files (including Farm column)
    """
    print(f"\nSaving results to CSV files (window={time_window}, step={step})...")

    # Map internal strategy keys to desired output names
    def map_strategy_output_name(name: str) -> str:
        # Normalize temporal-LSTM variant naming in outputs
        if name in ('two_stage_spectral_transformer_lstm', 'two_stage_spectral_transformer_lstm_temporal'):
            return 'two_stage_spectral_transformer_lstm_temporal'
        return name
    
    # 1. Data distribution and sample count statistics
    distribution_records = []
    for farm_id, farm_data_distributions in all_farm_data_distributions.items():
        for strategy_name, time_points in farm_data_distributions.items():
            for time_point, dist_info in time_points.items():
                record = {
                    'Farm': farm_id,
                    'strategy': map_strategy_output_name(strategy_name),
                    'time_point': time_point,
                    'mastitis_count': dist_info['mastitis_dim_stats']['count'],
                    'healthy_count': dist_info['healthy_dim_stats']['count'],
                    'mastitis_dim_mean': dist_info['mastitis_dim_stats']['mean'],
                    'mastitis_dim_std': dist_info['mastitis_dim_stats']['std'],
                    'healthy_dim_mean': dist_info['healthy_dim_stats']['mean'],
                    'healthy_dim_std': dist_info['healthy_dim_stats']['std'],
                }
                
                # Add missing value information (if any)
                if dist_info['missing_info']:
                    for interval, missing_count in dist_info['missing_info'].items():
                        record[f'missing_interval_{interval}'] = missing_count
                
                distribution_records.append(record)
    
    distribution_df = pd.DataFrame(distribution_records)
    distribution_df.to_csv(f'enhanced_data_distributions_w{time_window}_s{step}.csv', index=False)
    
    # 2. Detailed performance metrics (results from each repetition)
    detailed_records = []
    for farm_id, farm_detailed_metrics in all_farm_detailed_metrics.items():
        for strategy_name, time_points in farm_detailed_metrics.items():
            for time_point, metrics_list in time_points.items():
                for metrics in metrics_list:
                    record = {
                        'Farm': farm_id,
                        'strategy': map_strategy_output_name(strategy_name),
                        'time_point': time_point,
                        'iteration': metrics['iteration'],
                        'accuracy': metrics['accuracy'],
                        'precision': metrics['precision'],
                        'recall': metrics['recall'],
                        'auc': metrics['auc']
                    }
                    detailed_records.append(record)
    
    detailed_df = pd.DataFrame(detailed_records)
    detailed_df.to_csv(f'enhanced_detailed_metrics_w{time_window}_s{step}.csv', index=False)
    
    # 3. Summary results
    summary_records = []
    for farm_id, farm_results in all_farm_results.items():
        for strategy_name, time_points in farm_results.items():
            for time_point, results in time_points.items():
                record = {
                    'Farm': farm_id,
                    'strategy': map_strategy_output_name(strategy_name),
                    'time_point': time_point,
                    'accuracy_mean': results['accuracy_mean'],
                    'accuracy_std': results['accuracy_std'],
                    'precision_mean': results['precision_mean'],
                    'recall_mean': results['recall_mean'],
                    'auc_mean': results['auc_mean']
                }
                summary_records.append(record)
    
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(f'enhanced_summary_results_w{time_window}_s{step}.csv', index=False)
    
    # 4. Training loss and validation loss records (two-stage strategy only)
    loss_records = []
    for farm_id, farm_training_losses in all_farm_training_losses.items():
        for strategy_name, time_points in farm_training_losses.items():
            for time_point, iterations in time_points.items():
                for iter_data in iterations:
                    iteration = iter_data['iteration']
                    epoch_losses = iter_data['epoch_losses']
                    val_losses = iter_data.get('val_losses', [])
                    
                    # Ensure training loss and validation loss have consistent length
                    max_epochs = max(len(epoch_losses), len(val_losses))
                    for epoch in range(max_epochs):
                        train_loss = epoch_losses[epoch] if epoch < len(epoch_losses) else 0.0
                        val_loss = val_losses[epoch] if epoch < len(val_losses) else 0.0
                        
                        record = {
                            'Farm': farm_id,
                            'strategy': map_strategy_output_name(strategy_name),
                            'time_point': time_point,
                            'iteration': iteration,
                            'epoch': epoch + 1,
                            'train_loss': train_loss,
                            'val_loss': val_loss
                        }
                        loss_records.append(record)
    
    if loss_records:
        loss_df = pd.DataFrame(loss_records)
        loss_df.to_csv(f'enhanced_training_losses_w{time_window}_s{step}.csv', index=False)
        print("CSV files saved:")
        print(f"- enhanced_data_distributions_w{time_window}_s{step}.csv: Data distribution and sample statistics")
        print(f"- enhanced_detailed_metrics_w{time_window}_s{step}.csv: Detailed metrics from each repetition")
        print(f"- enhanced_summary_results_w{time_window}_s{step}.csv: Summary results")
        print(f"- enhanced_training_losses_w{time_window}_s{step}.csv: Two-stage strategy training loss and validation loss records")
    else:
        print("CSV files saved:")
        print(f"- enhanced_data_distributions_w{time_window}_s{step}.csv: Data distribution and sample statistics")
        print(f"- enhanced_detailed_metrics_w{time_window}_s{step}.csv: Detailed metrics from each repetition")
        print(f"- enhanced_summary_results_w{time_window}_s{step}.csv: Summary results")

def print_enhanced_summary_with_farm(all_farm_results, all_farm_model_info=None):
    """
    Print enhanced experiment summary (including Farm information)
    
    Args:
        all_farm_results: All farm experiment results
        all_farm_model_info: All farm model information dictionary, containing model size and running time
    """
    print("\n" + "=" * 80)
    print("Enhanced Baseline Experiment Results Summary (by Farm)")
    print("=" * 80)
    
    # Statistics by farm
    for farm_id, farm_results in all_farm_results.items():
        print(f"\nFarm {farm_id}:")
        print("-" * 40)
        
        # 1. Spectral vs milk composition comparison (two-stage modeling)
        print("\n1. Spectral vs Milk Composition Comparison (Two-Stage Modeling):")
        
        # Merge two-stage results from LSTM and Transformer (update: add transformer_lstm_temporal)
        spectral_results = {}
        spectral_results.update(farm_results.get('two_stage_spectral_transformer', {}))
        spectral_results.update(farm_results.get('two_stage_spectral_transformer_lstm', {}))
        
        composition_results = {}
        composition_results.update(farm_results.get('two_stage_composition_transformer_3', {}))
        # Optional: if composition_lstm is added in the future, include it
        if 'two_stage_composition_lstm' in farm_results:
            composition_results.update(farm_results.get('two_stage_composition_lstm', {}))
        
        if spectral_results and composition_results:
            spectral_avg = np.mean([r['accuracy_mean'] for r in spectral_results.values()])
            composition_avg = np.mean([r['accuracy_mean'] for r in composition_results.values()])
            
            print(f"  Spectral features two-stage:     {spectral_avg:.3f}")
            print(f"  Milk composition two-stage:      {composition_avg:.3f}")
            print(f"  Difference:                      {spectral_avg - composition_avg:+.3f}")
            
            if spectral_avg > composition_avg:
                print("  → Original spectral features perform better in two-stage modeling")
            else:
                print("  → Milk composition features perform better in two-stage modeling")
        
        # 2. Single-day modeling architecture comparison
        print("\n2. Single-Day Modeling Architecture Comparison (Spectral Features):")
        
        single_day_methods = {
            'Transformer': 'single_day_spectral_transformer',
            'LSTM': 'single_day_spectral_lstm',
            'Random Forest': 'single_day_spectral_rf',
            'PLS-DA': 'single_day_spectral_plsda',
            'LDA': 'single_day_spectral_lda'
        }
        
        for method_name, strategy_key in single_day_methods.items():
            if strategy_key in farm_results:
                avg_acc = np.mean([r['accuracy_mean'] for r in farm_results[strategy_key].values()])
                print(f"  {method_name:15s}: {avg_acc:.3f}")
                
                # Add detailed information for deep learning models
                if method_name in ['Transformer', 'LSTM'] and all_farm_model_info and farm_id in all_farm_model_info and strategy_key in all_farm_model_info[farm_id]:
                    info = all_farm_model_info[farm_id][strategy_key]
                    print(f"    └─ Model size: {info.get('model_size', 'N/A'):>8} parameters")
                    print(f"    └─ Running time: {info.get('running_time', 'N/A'):>8} seconds")
        
        # 3. Temporal information value validation
        print("\n3. Temporal Information Value Validation:")
        
        if 'single_day_spectral_transformer' in farm_results and spectral_results:
            single_avg = np.mean([r['accuracy_mean'] for r in farm_results['single_day_spectral_transformer'].values()])
            two_stage_avg = np.mean([r['accuracy_mean'] for r in spectral_results.values()])
            improvement = two_stage_avg - single_avg
            
            print(f"  Single-day Transformer:    {single_avg:.3f}")
            print(f"  Two-stage spectral:        {two_stage_avg:.3f}")
            print(f"  Temporal information gain: {improvement:+.3f} ({improvement/single_avg*100:+.1f}%)")
        
        # 4. Two-stage model architecture comparison
        print("\n4. Two-Stage Model Architecture Comparison (Spectral Features):")
        
        # Three two-stage models
        two_stage_models = {
            'Transformer-Transformer': 'two_stage_spectral_transformer',
            'Transformer-LSTM': 'two_stage_spectral_transformer_lstm',
            'LSTM-Transformer': 'two_stage_spectral_lstm_transformer'
        }
        
        model_performances = {}
        
        for model_name, strategy_key in two_stage_models.items():
            if strategy_key in farm_results and farm_results[strategy_key]:
                avg_acc = np.mean([r['accuracy_mean'] for r in farm_results[strategy_key].values()])
                model_performances[model_name] = avg_acc
                
                print(f"  {model_name:20s}: {avg_acc:.3f}")
                
                # Add model information
                if all_farm_model_info and farm_id in all_farm_model_info and strategy_key in all_farm_model_info[farm_id]:
                    info = all_farm_model_info[farm_id][strategy_key]
                    print(f"    └─ Model size: {info.get('model_size', 'N/A'):>8} parameters")
                    print(f"    └─ Running time: {info.get('running_time', 'N/A'):>8} seconds")
        
        # Find best model
        if model_performances:
            best_model = max(model_performances, key=model_performances.get)
            best_performance = model_performances[best_model]
            print(f"\n  → Best two-stage model: {best_model} ({best_performance:.3f})")
        
        # 5. Overall best method
        print("\n5. Overall Best Method:")
        
        best_method = None
        best_performance = 0
        
        for strategy_name, time_points in farm_results.items():
            if time_points:
                avg_performance = np.mean([r['accuracy_mean'] for r in time_points.values()])
                if avg_performance > best_performance:
                    best_performance = avg_performance
                    best_method = strategy_name
        
        if best_method:
            method_display = best_method.replace('_', ' ').title()
            print(f"  {method_display}: {best_performance:.3f}")
    
    # Cross-farm summary
    print(f"\n{'='*80}")
    print("Cross-Farm Summary")
    print("=" * 80)
    
    # Calculate average performance of each strategy across all farms
    all_strategies = set()
    for farm_results in all_farm_results.values():
        all_strategies.update(farm_results.keys())
    
    strategy_performance = {}
    for strategy in all_strategies:
        performances = []
        for farm_results in all_farm_results.values():
            if strategy in farm_results and farm_results[strategy]:
                avg_perf = np.mean([r['accuracy_mean'] for r in farm_results[strategy].values()])
                performances.append(avg_perf)
        
        if performances:
            strategy_performance[strategy] = {
                'mean': np.mean(performances),
                'std': np.std(performances),
                'farms': len(performances)
            }
    
    # Sort by average performance
    sorted_strategies = sorted(strategy_performance.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    print("\nAverage performance of all strategies across farms:")
    for strategy, perf in sorted_strategies:
        strategy_display = strategy.replace('_', ' ').title()
        print(f"  {strategy_display:30s}: {perf['mean']:.3f} ± {perf['std']:.3f} ({perf['farms']} farms)")
    
    print("\n" + "=" * 80)

def modeling(time_window=5, step=1, num_epochs=100, input_file='Data/ft-mir-pathogen_neg.csv'):
    """
    Run enhanced baseline comparison experiment
    
    Important fix: Resolved data leakage issue
    - Single-day strategy: Training and test sets now use the same scaler
    - Two-stage strategy: Split by cow level first, then standardize training and test sets separately
    - Ensure standardization statistics are learned only from training data
    - Avoid using test set information for feature engineering
    
    New: Run experiments separately by FARM_ID, add Farm column to output files
    
    Args:
        time_window (int): Time window size for two-stage modeling, default 5 days
        step (int): Sliding window step size, default 1 day
        num_epochs (int): Number of training epochs, default 100
    """
    print(f"Enhanced Baseline Comparison Experiment - Includes more features and milk composition comparison")
    print(f"Input file: {input_file}")
    print(f"Time window: {time_window} days, Step: {step} days, Training epochs: {num_epochs}")
    print("=" * 80)
    
    # 1. Data preparation
    df = load_data(input_file)
    
    # Get spectral columns (2121-2339 cm⁻¹)
    spectral_cols = get_spectral_columns(df)
    print(f"Found {len(spectral_cols)} spectral features")

    # Define columns containing only 3 main components
    milk_composition_3_cols = ["FAT", "PROTEIN", "LACTOSE"]
    
    # Get all FARM_IDs
    farm_ids = sorted(df['FARM_ID'].unique())
    print(f"Found {len(farm_ids)} farms: {farm_ids}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Define experiment strategies
    strategies = {
        'single_day_spectral_plsda': {
            'type': 'single_day',
            'model': 'plsda',
            'features': 'spectral'
        },
        'single_day_spectral_lda': {
            'type': 'single_day',
            'model': 'lda',
            'features': 'spectral'
        },
        'single_day_spectral_rf': {
            'type': 'single_day',
            'model': 'random_forest',
            'features': 'spectral'
        },
        'single_day_spectral_lstm': {
            'type': 'single_day',
            'model': 'lstm',
            'features': 'spectral'
        },
        'single_day_spectral_transformer': {
            'type': 'single_day',
            'model': 'transformer',
            'features': 'spectral'
        },
        'two_stage_spectral_transformer': {
            'type': 'two_stage',
            'model': 'transformer_transformer',
            'features': 'spectral'
        },
        'two_stage_composition_transformer_3': {
            'type': 'two_stage',
            'model': 'transformer_transformer',
            'features': 'composition_3'
        },
        'two_stage_spectral_transformer_lstm': {
            'type': 'two_stage',
            'model': 'transformer_lstm_temporal',
            'features': 'spectral'
        },
        'two_stage_spectral_lstm_transformer': {
            'type': 'two_stage',
            'model': 'lstm_transformer',
            'features': 'spectral'
        }
    }
    
    # 3. Store results for all farms
    all_farm_results = {}
    all_farm_data_distributions = {}
    all_farm_detailed_metrics = {}
    all_farm_training_losses = {}
    all_farm_model_info = {}
    
    # 4. Run experiments for each farm
    for farm_id in farm_ids:
        print(f"\n{'='*80}")
        print(f"Processing farm: {farm_id}")
        print(f"{'='*80}")
        
        # Filter data for current farm
        farm_df = df[df['FARM_ID'] == farm_id].copy()
        
        if len(farm_df) < 5:  # Skip if farm data is too small
            print(f"  Skip farm {farm_id}: Too few samples ({len(farm_df)})")
            continue
        
        print(f"  Farm {farm_id} data: {len(farm_df)} samples")
        print(f"  Mastitis samples: {len(farm_df[farm_df['group'] == 'mastitis'])}")
        print(f"  Healthy samples: {len(farm_df[farm_df['group'] == 'healthy'])}")
        
        # Farm-specific feature selection
        variance_df = analyze_farm_specific_variance(df, farm_id, spectral_cols)
        farm_specific_features = select_farm_specific_features(variance_df, keep_percentile=1)
        
        if len(farm_specific_features) == 0:
            print(f"  Skip farm {farm_id}: No valid features")
            continue
        
        print(f"  Using {len(farm_specific_features)} features")
        
        # Separate data
        mastitis_data = farm_df[farm_df['group'] == 'mastitis'].copy()
        healthy_data = farm_df[farm_df['group'] == 'healthy'].copy()
        
        # Store results for current farm
        all_results = {}
        data_distributions = {}
        detailed_metrics = {}
        training_losses = {}
        model_info = {}
        
        # 5. Run experiments
        for strategy_name, config in strategies.items():
            print(f"\n{'='*60}")
            print(f"Running strategy: {strategy_name}")
            print(f"{'='*60}")
            
            all_results[strategy_name] = {}
            data_distributions[strategy_name] = {}
            detailed_metrics[strategy_name] = {}
            training_losses[strategy_name] = {}
            
            if config['type'] == 'single_day':
                # Single-day modeling - includes all time points from 30 to 0
                time_points = list(range(30, -1, -1))  # 30, 29, 28, ..., 1, 0
                
                for time_point in time_points:
                    print(f"\nTime point {time_point} days ago:")
                    
                    # Filter data
                    point_mastitis = mastitis_data[mastitis_data['interval'] == time_point].copy()
                    
                    if len(point_mastitis) < 2:  # Lower threshold from 5 to 2
                        print(f"  Skip: Too few samples ({len(point_mastitis)})")
                        continue
                    
                    # Perform 20 iterations with 5-fold cross validation
                    metrics_list = []
                    
                    for iteration in range(2):  # Increase from 2 to 20 iterations for better statistical reliability
                        try:
                            # DIM matching
                            matched_healthy = match_dim_distribution(
                                point_mastitis, healthy_data, random_state=42+iteration
                            )
                            combined_data = pd.concat([point_mastitis, matched_healthy], ignore_index=True)
                            
                            # Analyze data distribution (only record in first iteration)
                            if iteration == 0:
                                data_distributions[strategy_name][time_point] = analyze_data_distribution(
                                    combined_data, time_point, is_window=False
                                )
                            
                            # Select feature columns
                            if config['features'] == 'spectral':
                                feature_cols = farm_specific_features  # Use farm-specific features
                                comp_cols = None
                            elif config['features'] == 'composition_3':
                                feature_cols = None
                                comp_cols = milk_composition_3_cols
                            
                            # 5-fold cross validation
                            kfold = KFold(n_splits=5, shuffle=True, random_state=42+iteration)
                            fold_metrics = []
                            
                            for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(combined_data)):
                                train_data = combined_data.iloc[train_idx].copy()
                                test_data = combined_data.iloc[test_idx].copy()
                                
                                # Train based on model type
                                if config['model'] in ['transformer', 'lstm']:
                                    # Deep learning models
                                    train_dataset = EnhancedSingleDayDataset(
                                        train_data, feature_cols, comp_cols, 'minmax'
                                    )
                                    test_dataset = EnhancedSingleDayDataset(
                                        test_data, feature_cols, comp_cols, 'minmax',
                                        content_scaler=train_dataset.content_scaler,
                                        metadata_scaler=train_dataset.metadata_scaler
                                    )
                                    
                                    # Calculate class weights for weighted loss
                                    train_labels = train_dataset.labels
                                    class_counts = np.bincount(train_labels)
                                    total_samples = len(train_labels)
                                    class_weights = [total_samples / (2 * count) if count > 0 else 0.0 for count in class_counts]
                                    
                                    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
                                    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
                                    
                                    input_dim = train_dataset.features.shape[1]
                                    
                                    if config['model'] == 'transformer':
                                        model = SingleDayTransformer(input_dim, d_model=64, nhead=8).to(device)
                                    else:  # lstm
                                        model = SingleDayLSTM(input_dim, hidden_dim=96, num_layers=2).to(device)
                                    
                                    # Calculate model size
                                    total_params = sum(p.numel() for p in model.parameters())
                                    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                                    
                                    # Record start time
                                    import time
                                    start_time = time.time()
                                    
                                    accuracy, precision, recall, auc = train_and_evaluate_single_day(
                                        model, train_loader, test_loader, device, num_epochs=num_epochs, class_weights=class_weights
                                    )
                                    
                                    # Calculate running time
                                    end_time = time.time()
                                    running_time = end_time - start_time
                                    
                                    # Store model info (only record in first iteration and first fold)
                                    if iteration == 0 and fold_idx == 0:
                                        model_info[strategy_name] = {
                                            'model_size': f"{total_params:,}",
                                            'trainable_params': f"{trainable_params:,}",
                                            'running_time': f"{running_time:.1f}"
                                        }
                                    
                                    fold_metrics.append({
                                        'accuracy': accuracy,
                                        'precision': precision,
                                        'recall': recall,
                                        'auc': auc
                                    })
                                
                                else:
                                    # Traditional machine learning models
                                    train_dataset = EnhancedSingleDayDataset(
                                        train_data, feature_cols, comp_cols, 'minmax'
                                    )
                                    test_dataset = EnhancedSingleDayDataset(
                                        test_data, feature_cols, comp_cols, 'minmax',
                                        content_scaler=train_dataset.content_scaler,
                                        metadata_scaler=train_dataset.metadata_scaler
                                    )
                                    
                                    X_train, y_train = train_dataset.features, train_dataset.labels
                                    X_test, y_test = test_dataset.features, test_dataset.labels
                                
                                    if config['model'] == 'random_forest':
                                        model = RandomForestClassifier(n_estimators=50, random_state=42+iteration)
                                        model.fit(X_train, y_train)
                                    elif config['model'] == 'plsda':
                                        # PLS-DA implementation - fixed version with enhanced numerical stability
                                        try:
                                            # Check input data validity
                                            if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
                                                print(f"    Warning: Training data contains NaN values, skipping PLS-DA")
                                                continue
                                            
                                            if np.any(np.isinf(X_train)) or np.any(np.isinf(y_train)):
                                                print(f"    Warning: Training data contains infinite values, skipping PLS-DA")
                                                continue
                                            
                                            # Check data dimensions
                                            if X_train.shape[0] < 3:
                                                print(f"    Warning: Too few training samples ({X_train.shape[0]}), skipping PLS-DA")
                                                continue
                                            
                                            # For binary classification, usually 1 component is sufficient
                                            n_components = 14
                                            
                                            if n_components < 1:
                                                print(f"    Warning: Cannot determine PLS component number, skipping PLS-DA")
                                                continue
                                            
                                            model = PLSRegression(n_components=n_components, scale=False)
                                            
                                            # Use original labels (0,1) directly, no LabelEncoder needed
                                            model.fit(X_train, y_train.reshape(-1, 1))
                                            
                                            # Check if model trained successfully
                                            if hasattr(model, 'coef_') and np.any(np.isnan(model.coef_)):
                                                print(f"    Warning: PLS-DA model coefficients contain NaN, skipping")
                                                continue
                                            
                                            # Add predict and predict_proba methods for compatibility
                                            def predict_method(X):
                                                try:
                                                    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                                                        return np.zeros(X.shape[0], dtype=int)
                                                    
                                                    scores = model.predict(X).flatten()
                                                    
                                                    # Check if prediction scores are valid
                                                    if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
                                                        return np.zeros(len(scores), dtype=int)
                                                    
                                                    # Use 0.5 as threshold for binary classification
                                                    return (scores > 0.5).astype(int)
                                                except Exception as e:
                                                    print(f"      PLS-DA prediction failed: {e}")
                                                    return np.zeros(X.shape[0], dtype=int)
                                            
                                            def predict_proba_method(X):
                                                try:
                                                    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
                                                        # Return uniform probabilities
                                                        return np.column_stack([np.ones(X.shape[0]) * 0.5, np.ones(X.shape[0]) * 0.5])
                                                    
                                                    scores = model.predict(X).flatten()
                                                    
                                                    # Check if prediction scores are valid
                                                    if np.any(np.isnan(scores)) or np.any(np.isinf(scores)):
                                                        return np.column_stack([np.ones(len(scores)) * 0.5, np.ones(len(scores)) * 0.5])
                                                    
                                                    # Clip scores to [0,1] range
                                                    scores_clipped = np.clip(scores, 0, 1)
                                                    
                                                    # Create probabilities for both classes
                                                    probs = np.column_stack([1 - scores_clipped, scores_clipped])
                                                    
                                                    # Ensure probabilities sum to 1, prevent division by zero
                                                    row_sums = probs.sum(axis=1, keepdims=True)
                                                    row_sums[row_sums == 0] = 1.0  # Prevent division by zero
                                                    probs = probs / row_sums
                                                    
                                                    # Final check
                                                    if np.any(np.isnan(probs)) or np.any(np.isinf(probs)):
                                                        return np.column_stack([np.ones(len(scores)) * 0.5, np.ones(len(scores)) * 0.5])
                                                    
                                                    return probs
                                                except Exception as e:
                                                    print(f"      PLS-DA probability prediction failed: {e}")
                                                    return np.column_stack([np.ones(X.shape[0]) * 0.5, np.ones(X.shape[0]) * 0.5])
                                            
                                            model.predict = predict_method
                                            model.predict_proba = predict_proba_method
                                            
                                        except Exception as e:
                                            print(f"    PLS-DA model creation failed: {e}")
                                            continue
                                    elif config['model'] == 'lda':
                                        # LDA implementation
                                        try:
                                            # Check input data validity
                                            if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
                                                print(f"    Warning: Training data contains NaN values, skipping LDA")
                                                continue
                                            
                                            if np.any(np.isinf(X_train)) or np.any(np.isinf(y_train)):
                                                print(f"    Warning: Training data contains infinite values, skipping LDA")
                                                continue
                                            
                                            # Check data dimensions
                                            if X_train.shape[0] < 3:
                                                print(f"    Warning: Too few training samples ({X_train.shape[0]}), skipping LDA")
                                                continue
                                            
                                            model = LinearDiscriminantAnalysis()
                                            model.fit(X_train, y_train)
                                            
                                            # Check if model trained successfully
                                            if hasattr(model, 'coef_') and np.any(np.isnan(model.coef_)):
                                                print(f"    Warning: LDA model coefficients contain NaN, skipping")
                                                continue
                                            
                                        except Exception as e:
                                            print(f"    LDA model creation failed: {e}")
                                            continue
                                    else:
                                        print(f"    Unknown model type: {config['model']}")
                                        continue
                                    
                                    predictions = model.predict(X_test)
                                    probabilities = model.predict_proba(X_test)
                                    
                                    # Add numerical stability checks
                                    try:
                                        # Check if prediction results are valid
                                        if np.any(np.isnan(predictions)) or np.any(np.isnan(probabilities)):
                                            print(f"    Warning: Prediction results contain NaN values")
                                            accuracy = precision = recall = auc = 0.0
                                        else:
                                            accuracy = accuracy_score(y_test, predictions)
                                            precision = precision_score(y_test, predictions, average='weighted')
                                            recall = recall_score(y_test, predictions, average='weighted')
                                            
                                            # Additional check for AUC calculation
                                            if len(set(y_test)) > 1 and not np.any(np.isnan(probabilities[:, 1])):
                                                try:
                                                    auc = roc_auc_score(y_test, probabilities[:, 1])
                                                    # Check if AUC is NaN
                                                    if np.isnan(auc) or np.isinf(auc):
                                                        auc = 0.0
                                                except Exception as auc_e:
                                                    print(f"    AUC calculation failed: {auc_e}")
                                                    auc = 0.0
                                            else:
                                                auc = 0.0
                                            
                                            # Final check for all metrics
                                            if np.isnan(accuracy) or np.isinf(accuracy):
                                                accuracy = 0.0
                                            if np.isnan(precision) or np.isinf(precision):
                                                precision = 0.0
                                            if np.isnan(recall) or np.isinf(recall):
                                                recall = 0.0
                                            if np.isnan(auc) or np.isinf(auc):
                                                auc = 0.0
                                        
                                    except Exception as eval_e:
                                        print(f"    Evaluation metric calculation failed: {eval_e}")
                                        accuracy = precision = recall = auc = 0.0
                                    
                                    fold_metrics.append({
                                        'accuracy': accuracy,
                                        'precision': precision,
                                        'recall': recall,
                                        'auc': auc
                                    })
                            
                            # Average metrics across folds
                            if fold_metrics:
                                avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
                                avg_precision = np.mean([m['precision'] for m in fold_metrics])
                                avg_recall = np.mean([m['recall'] for m in fold_metrics])
                                avg_auc = np.mean([m['auc'] for m in fold_metrics])
                                
                                metrics_list.append({
                                    'accuracy': avg_accuracy,
                                    'precision': avg_precision,
                                    'recall': avg_recall,
                                    'auc': avg_auc,
                                    'iteration': iteration
                                })
                            
                        except Exception as e:
                            print(f"    Iteration {iteration} failed: {e}")
                            continue
                    
                    # Save results
                    if metrics_list:
                        # Calculate mean and handle NaN
                        accuracy_values = [m['accuracy'] for m in metrics_list]
                        precision_values = [m['precision'] for m in metrics_list]
                        recall_values = [m['recall'] for m in metrics_list]
                        auc_values = [m['auc'] for m in metrics_list]
                        
                        # Replace NaN values with 0
                        accuracy_values = [0.0 if np.isnan(v) or np.isinf(v) else v for v in accuracy_values]
                        precision_values = [0.0 if np.isnan(v) or np.isinf(v) else v for v in precision_values]
                        recall_values = [0.0 if np.isnan(v) or np.isinf(v) else v for v in recall_values]
                        auc_values = [0.0 if np.isnan(v) or np.isinf(v) else v for v in auc_values]
                        
                        all_results[strategy_name][time_point] = {
                            'accuracy_mean': np.mean(accuracy_values) if accuracy_values else 0.0,
                            'accuracy_std': np.std(accuracy_values) if accuracy_values else 0.0,
                            'precision_mean': np.mean(precision_values) if precision_values else 0.0,
                            'recall_mean': np.mean(recall_values) if recall_values else 0.0,
                            'auc_mean': np.mean(auc_values) if auc_values else 0.0
                        }
                        detailed_metrics[strategy_name][time_point] = metrics_list
                        
                        print(f"  Completed: ACC={all_results[strategy_name][time_point]['accuracy_mean']:.3f}, "
                              f"AUC={all_results[strategy_name][time_point]['auc_mean']:.3f}")
            
            else:
                # Two-stage modeling - use sliding window
                windows = []
                for start in range(30 - time_window + 1, -1, -step):
                    end = start + time_window - 1
                    if end <= 30:
                        windows.append((start, end))
                
                for window_start, window_end in windows:
                    window_center = (window_start + window_end) / 2
                    print(f"\nWindow {window_start}-{window_end} days ago (center: {window_center}):")
                    
                    # Filter data
                    window_mastitis = mastitis_data[
                        (mastitis_data['interval'] >= window_start) & 
                        (mastitis_data['interval'] <= window_end)
                    ].copy()
                    
                    if len(window_mastitis) < 10:
                        print(f"  Skip: Too few samples ({len(window_mastitis)})")
                        continue
                    
                    # Perform 20 iterations
                    metrics_list = []
                    
                    for iteration in range(2):  # Increase from 2 to 20 iterations for better statistical reliability
                        try:
                            # DIM matching
                            matched_healthy = match_dim_distribution(
                                window_mastitis, healthy_data, random_state=42+iteration,
                                window_start=window_start, window_end=window_end
                            )
                            combined_data = pd.concat([window_mastitis, matched_healthy], ignore_index=True)
                            
                            # Analyze data distribution (only record in first iteration)
                            if iteration == 0:
                                data_distributions[strategy_name][window_center] = analyze_data_distribution(
                                    combined_data, window_center, is_window=True, 
                                    window_start=window_start, window_end=window_end
                                )
                            
                            # Select feature columns
                            if config['features'] == 'spectral':
                                feature_cols = farm_specific_features  # Use farm-specific features
                                comp_cols = None
                            elif config['features'] == 'composition_3':
                                feature_cols = None
                                comp_cols = milk_composition_3_cols
                            
                            # Fix: Split by cow level first, then standardize
                            print(f"  Fix standardization process: split first then standardize")
                            
                            # Step 1: Split raw data by cow level
                            all_cow_ids = list(window_mastitis['ANIMAL_SOURCE_ID'].unique()) + list(healthy_data['ANIMAL_SOURCE_ID'].unique())
                            print(f"    Total cows: {len(all_cow_ids)}")
                            print(f"    Mastitis cows: {len(window_mastitis['ANIMAL_SOURCE_ID'].unique())}")
                            print(f"    Healthy cows: {len(healthy_data['ANIMAL_SOURCE_ID'].unique())}")
                            
                            # 5-fold cross validation on cow IDs
                            kfold = KFold(n_splits=5, shuffle=True, random_state=42+iteration)
                            fold_metrics = []
                            
                            for fold_idx, (train_cow_idx, test_cow_idx) in enumerate(kfold.split(all_cow_ids)):
                                train_cow_ids = [all_cow_ids[i] for i in train_cow_idx]
                                test_cow_ids = [all_cow_ids[i] for i in test_cow_idx]
                                
                                print(f"    Fold {fold_idx+1}/5: Train cows: {len(train_cow_ids)}, Test cows: {len(test_cow_ids)}")
                                
                                # Step 2: Separate training and test set raw data
                                train_mastitis = window_mastitis[window_mastitis['ANIMAL_SOURCE_ID'].isin(train_cow_ids)].copy()
                                test_mastitis = window_mastitis[window_mastitis['ANIMAL_SOURCE_ID'].isin(test_cow_ids)].copy()
                                
                                train_healthy = healthy_data[healthy_data['ANIMAL_SOURCE_ID'].isin(train_cow_ids)].copy()
                                test_healthy = healthy_data[healthy_data['ANIMAL_SOURCE_ID'].isin(test_cow_ids)].copy()
                                
                                print(f"      Train: {len(train_mastitis)} mastitis samples, {len(train_healthy)} healthy samples")
                                print(f"      Test: {len(test_mastitis)} mastitis samples, {len(test_healthy)} healthy samples")
                                
                                # Step 3: Create scaler using only training set data
                                train_cow_sequences, content_scaler, metadata_scaler = create_enhanced_cow_sequences_for_two_stage(
                                    train_mastitis, train_healthy, feature_cols, comp_cols, 'minmax',
                                    max_interval=window_end, min_interval=window_start
                                )
                                
                                # Step 4: Process test set data using training set scaler
                                test_cow_sequences, _, _ = create_enhanced_cow_sequences_for_two_stage(
                                    test_mastitis, test_healthy, feature_cols, comp_cols, 'minmax',
                                    max_interval=window_end, min_interval=window_start,
                                    content_scaler=content_scaler, metadata_scaler=metadata_scaler
                                )
                                
                                # Step 5: Merge training and test set sequences
                                all_cow_sequences = {**train_cow_sequences, **test_cow_sequences}
                                
                                # Calculate window size and set min_real_data_points to 25% of window
                                window_size = window_end - window_start + 1
                                min_real_data_points = max(1, int(window_size * 0.25))
                                
                                dataset = EnhancedTimeSeriesDataset(all_cow_sequences, min_real_data_points=min_real_data_points)
                                
                                if len(dataset) < 10:
                                    continue
                                
                                # Step 6: Re-split dataset (based on cow ID)
                                # Note: Healthy cow cow_id is modified to "healthy_original_cow_id"
                                train_indices = []
                                test_indices = []
                                
                                for i, cow_id in enumerate(dataset.cow_ids):
                                    # Convert cow_id to string for checking
                                    cow_id_str = str(cow_id)
                                    
                                    # Check if healthy cow (starts with "healthy_")
                                    if cow_id_str.startswith("healthy_"):
                                        original_cow_id = cow_id_str.replace("healthy_", "")
                                        # Convert original_cow_id back to integer for comparison
                                        try:
                                            original_cow_id_int = int(original_cow_id)
                                            if original_cow_id_int in train_cow_ids:
                                                train_indices.append(i)
                                            elif original_cow_id_int in test_cow_ids:
                                                test_indices.append(i)
                                        except ValueError:
                                            # If conversion fails, skip
                                            continue
                                    else:
                                        # Mastitis cow, use original cow_id directly
                                        if cow_id in train_cow_ids:
                                            train_indices.append(i)
                                        elif cow_id in test_cow_ids:
                                            test_indices.append(i)
                                
                                train_dataset = torch.utils.data.Subset(dataset, train_indices)
                                test_dataset = torch.utils.data.Subset(dataset, test_indices)
                                
                                # Check training and test set label distribution
                                train_labels = [dataset.labels[i] for i in train_indices]
                                test_labels = [dataset.labels[i] for i in test_indices]
                                
                                train_label_counts = pd.Series(train_labels).value_counts()
                                test_label_counts = pd.Series(test_labels).value_counts()
                                
                                # Check if there are enough samples
                                if len(train_labels) == 0 or len(test_labels) == 0:
                                    print("      Warning: Training or test set is empty!")
                                    continue
                                
                                if len(set(train_labels)) < 2 or len(set(test_labels)) < 2:
                                    print("      Warning: Training or test set has only one class!")
                                    continue
                                
                                # Calculate class weights for weighted loss
                                class_counts = np.bincount(train_labels)
                                total_samples = len(train_labels)
                                class_weights = [total_samples / (2 * count) if count > 0 else 0.0 for count in class_counts]
                                
                                train_loader = DataLoader(
                                    train_dataset, batch_size=16, shuffle=True, 
                                    collate_fn=enhanced_collate_fn
                                )
                                test_loader = DataLoader(
                                    test_dataset, batch_size=16, shuffle=False, 
                                    collate_fn=enhanced_collate_fn
                                )
                                
                                content_dim = dataset.content_sequences[0].shape[1]
                                metadata_dim = dataset.metadata_sequences[0].shape[1]
                                
                                # Create two-stage model
                                if config['model'] == 'transformer_transformer':
                                    model = EnhancedTwoStageModel(
                                        spectral_dim=content_dim,
                                        metadata_dim=metadata_dim,
                                        spectral_embedding_dim=64,
                                        spectral_encoder_type='transformer',
                                        temporal_model_type='transformer'
                                    ).to(device)
                                elif config['model'] == 'transformer_lstm_temporal':
                                    model = EnhancedTwoStageModel(
                                        spectral_dim=content_dim,
                                        metadata_dim=metadata_dim,
                                        spectral_embedding_dim=64,
                                        spectral_encoder_type='transformer',
                                        temporal_model_type='lstm'
                                    ).to(device)
                                elif config['model'] == 'lstm_transformer':
                                    model = EnhancedTwoStageModel(
                                        spectral_dim=content_dim,
                                        metadata_dim=metadata_dim,
                                        spectral_embedding_dim=64,
                                        spectral_encoder_type='lstm',
                                        temporal_model_type='transformer'
                                    ).to(device)
                                
                                # Calculate model size
                                total_params = sum(p.numel() for p in model.parameters())
                                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                                
                                # Record start time
                                import time
                                start_time = time.time()
                                
                                accuracy, precision, recall, auc, epoch_losses, val_losses = train_and_evaluate_two_stage(
                                    model, train_loader, test_loader, device, num_epochs=num_epochs, class_weights=class_weights
                                )
                                
                                # Calculate running time
                                end_time = time.time()
                                running_time = end_time - start_time
                                
                                # Store model info (only record in first iteration and first fold)
                                if iteration == 0 and fold_idx == 0:
                                    model_info[strategy_name] = {
                                        'model_size': f"{total_params:,}",
                                        'trainable_params': f"{trainable_params:,}",
                                        'running_time': f"{running_time:.1f}"
                                    }
                                
                                fold_metrics.append({
                                    'accuracy': accuracy,
                                    'precision': precision,
                                    'recall': recall,
                                    'auc': auc
                                })
                            
                            # Average metrics across folds
                            if fold_metrics:
                                avg_accuracy = np.mean([m['accuracy'] for m in fold_metrics])
                                avg_precision = np.mean([m['precision'] for m in fold_metrics])
                                avg_recall = np.mean([m['recall'] for m in fold_metrics])
                                avg_auc = np.mean([m['auc'] for m in fold_metrics])
                                
                                metrics_list.append({
                                    'accuracy': avg_accuracy,
                                    'precision': avg_precision,
                                    'recall': avg_recall,
                                    'auc': avg_auc,
                                    'iteration': iteration
                                })
                            
                        except Exception as e:
                            print(f"    Iteration {iteration} failed: {e}")
                            continue
                    
                    # Save results
                    if metrics_list:
                        # Calculate mean and handle NaN
                        accuracy_values = [m['accuracy'] for m in metrics_list]
                        precision_values = [m['precision'] for m in metrics_list]
                        recall_values = [m['recall'] for m in metrics_list]
                        auc_values = [m['auc'] for m in metrics_list]
                        
                        # Replace NaN values with 0
                        accuracy_values = [0.0 if np.isnan(v) or np.isinf(v) else v for v in accuracy_values]
                        precision_values = [0.0 if np.isnan(v) or np.isinf(v) else v for v in precision_values]
                        recall_values = [0.0 if np.isnan(v) or np.isinf(v) else v for v in recall_values]
                        auc_values = [0.0 if np.isnan(v) or np.isinf(v) else v for v in auc_values]
                        
                        all_results[strategy_name][window_center] = {
                            'accuracy_mean': np.mean(accuracy_values) if accuracy_values else 0.0,
                            'accuracy_std': np.std(accuracy_values) if accuracy_values else 0.0,
                            'precision_mean': np.mean(precision_values) if precision_values else 0.0,
                            'recall_mean': np.mean(recall_values) if recall_values else 0.0,
                            'auc_mean': np.mean(auc_values) if auc_values else 0.0
                        }
                        detailed_metrics[strategy_name][window_center] = metrics_list
                        
                        print(f"  Completed: ACC={all_results[strategy_name][window_center]['accuracy_mean']:.3f}, "
                              f"AUC={all_results[strategy_name][window_center]['auc_mean']:.3f}")
            
            # Store results for current farm
            all_farm_results[farm_id] = all_results
            all_farm_data_distributions[farm_id] = data_distributions
            all_farm_detailed_metrics[farm_id] = detailed_metrics
            all_farm_training_losses[farm_id] = training_losses
            all_farm_model_info[farm_id] = model_info
    
    # 6. Save results to CSV files (including Farm column)
    save_results_to_csv_with_farm(all_farm_results, all_farm_data_distributions, 
                                 all_farm_detailed_metrics, all_farm_training_losses, 
                                 time_window, step)
    
    # 7. Output summary
    print_enhanced_summary_with_farm(all_farm_results, all_farm_model_info)
    
    return all_farm_results, all_farm_data_distributions, all_farm_detailed_metrics


def plot_single_day_trends(all_farm_results):
    """
    Plot performance trends for single-day modeling across all farms
    """
    print(f"\n{'='*80}")
    print("Plotting Single-Day Modeling Performance Trends")
    print(f"{'='*80}")
    
    # Filter single-day modeling strategies
    single_day_strategies = [
        'single_day_spectral_lda',
        'single_day_spectral_rf', 
        'single_day_spectral_plsda',
        'single_day_spectral_lstm',
        'single_day_spectral_transformer'
    ]
    
    # Get all farm IDs
    farm_ids = sorted(all_farm_results.keys())
    
    # Create trend plots for each single-day strategy
    for strategy in single_day_strategies:
        print(f"\nProcessing strategy: {strategy}")
        
        # Check if strategy exists
        strategy_exists = False
        for farm_id in farm_ids:
            if strategy in all_farm_results[farm_id]:
                strategy_exists = True
                break
        
        if not strategy_exists:
            print(f"  Skip: Strategy {strategy} does not exist")
            continue
        
        # Collect performance data from all farms
        farm_performance_data = {}
        
        for farm_id in farm_ids:
            if strategy in all_farm_results[farm_id]:
                farm_data = all_farm_results[farm_id][strategy]
                
                # Extract time points and performance data
                time_points = []
                auc_values = []
                acc_values = []
                
                for time_point, results in farm_data.items():
                    if isinstance(time_point, (int, float)):  # Ensure it's a numeric value
                        time_points.append(time_point)
                        auc_values.append(results['auc_mean'])
                        acc_values.append(results['accuracy_mean'])
                
                if len(time_points) > 0:
                    # Sort by time point
                    sorted_data = sorted(zip(time_points, auc_values, acc_values))
                    time_points, auc_values, acc_values = zip(*sorted_data)
                    
                    farm_performance_data[farm_id] = {
                        'time_points': time_points,
                        'auc_values': auc_values,
                        'acc_values': acc_values
                    }
        
        if len(farm_performance_data) == 0:
            print(f"  Skip: No valid farm data")
            continue
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Subplot 1: AUC trends for all farms
        ax1 = axes[0, 0]
        for farm_id, data in farm_performance_data.items():
            time_points = data['time_points']
            auc_values = data['auc_values']
            
            # Calculate correlation coefficient
            if len(time_points) > 1:
                corr = np.corrcoef(time_points, auc_values)[0, 1]
                label = f'Farm {farm_id} (r={corr:.3f})'
            else:
                label = f'Farm {farm_id}'
            
            ax1.plot(time_points, auc_values, 'o-', label=label, alpha=0.7, linewidth=2, markersize=4)
        
        ax1.set_xlabel('Time Point (Days Ago)')
        ax1.set_ylabel('AUC')
        ax1.set_title(f'{strategy} - AUC Trends Across All Farms')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Subplot 2: Accuracy trends for all farms
        ax2 = axes[0, 1]
        for farm_id, data in farm_performance_data.items():
            time_points = data['time_points']
            acc_values = data['acc_values']
            
            # Calculate correlation coefficient
            if len(time_points) > 1:
                corr = np.corrcoef(time_points, acc_values)[0, 1]
                label = f'Farm {farm_id} (r={corr:.3f})'
            else:
                label = f'Farm {farm_id}'
            
            ax2.plot(time_points, acc_values, 's-', label=label, alpha=0.7, linewidth=2, markersize=4)
        
        ax2.set_xlabel('Time Point (Days Ago)')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'{strategy} - Accuracy Trends Across All Farms')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        # Subplot 3: Correlation coefficient distribution
        ax3 = axes[1, 0]
        auc_corrs = []
        acc_corrs = []
        farm_labels = []
        
        for farm_id, data in farm_performance_data.items():
            time_points = data['time_points']
            auc_values = data['auc_values']
            acc_values = data['acc_values']
            
            if len(time_points) > 1:
                auc_corr = np.corrcoef(time_points, auc_values)[0, 1]
                acc_corr = np.corrcoef(time_points, acc_values)[0, 1]
                
                auc_corrs.append(auc_corr)
                acc_corrs.append(acc_corr)
                farm_labels.append(f'Farm {farm_id}')
        
        if len(auc_corrs) > 0:
            x = np.arange(len(farm_labels))
            width = 0.35
            
            ax3.bar(x - width/2, auc_corrs, width, label='AUC Correlation', alpha=0.7)
            ax3.bar(x + width/2, acc_corrs, width, label='Accuracy Correlation', alpha=0.7)
            ax3.axhline(y=-0.1, color='k', linestyle='--', alpha=0.5, label='Threshold (-0.1)')
            
            ax3.set_xlabel('Farm')
            ax3.set_ylabel('Correlation Coefficient')
            ax3.set_title(f'{strategy} - Correlation Coefficients by Farm')
            ax3.set_xticks(x)
            ax3.set_xticklabels(farm_labels, rotation=45)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Performance improvement statistics
        ax4 = axes[1, 1]
        auc_improvements = []
        acc_improvements = []
        
        for farm_id, data in farm_performance_data.items():
            time_points = data['time_points']
            auc_values = data['auc_values']
            acc_values = data['acc_values']
            
            if len(auc_values) >= 10:
                # Calculate early vs late performance improvement
                auc_early = np.mean(auc_values[:5])  # First 5 time points
                auc_late = np.mean(auc_values[-5:])  # Last 5 time points
                auc_improvement = auc_late - auc_early
                
                acc_early = np.mean(acc_values[:5])
                acc_late = np.mean(acc_values[-5:])
                acc_improvement = acc_late - acc_early
            else:
                auc_improvement = 0.0
                acc_improvement = 0.0
            
            auc_improvements.append(auc_improvement)
            acc_improvements.append(acc_improvement)
        
        if len(auc_improvements) > 0:
            x = np.arange(len(farm_labels))
            width = 0.35
            
            ax4.bar(x - width/2, auc_improvements, width, label='AUC Improvement', alpha=0.7)
            ax4.bar(x + width/2, acc_improvements, width, label='Accuracy Improvement', alpha=0.7)
            ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Threshold (0)')
            
            ax4.set_xlabel('Farm')
            ax4.set_ylabel('Performance Improvement')
            ax4.set_title(f'{strategy} - Performance Improvements by Farm')
            ax4.set_xticks(x)
            ax4.set_xticklabels(farm_labels, rotation=45)
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        strategy_name = strategy.replace('_', ' ').title()
        plt.savefig(f'single_day_{strategy}_trends.png', dpi=300, bbox_inches='tight')
        print(f"  Plot saved as 'single_day_{strategy}_trends.png'")
        
        # Print statistics
        print(f"\n{strategy} Statistics:")
        if len(auc_corrs) > 0:
            auc_positive_trend = sum(1 for corr in auc_corrs if corr < -0.1)
            acc_positive_trend = sum(1 for corr in acc_corrs if corr < -0.1)
            auc_positive_improvement = sum(1 for imp in auc_improvements if imp > 0)
            acc_positive_improvement = sum(1 for imp in acc_improvements if imp > 0)
            
            total_farms = len(auc_corrs)
            print(f"  Number of farms: {total_farms}")
            print(f"  Farms with correct AUC trend: {auc_positive_trend}/{total_farms} ({auc_positive_trend/total_farms*100:.1f}%)")
            print(f"  Farms with correct accuracy trend: {acc_positive_trend}/{total_farms} ({acc_positive_trend/total_farms*100:.1f}%)")
            print(f"  Farms with AUC improvement: {auc_positive_improvement}/{total_farms} ({auc_positive_improvement/total_farms*100:.1f}%)")
            print(f"  Farms with accuracy improvement: {acc_positive_improvement}/{total_farms} ({acc_positive_improvement/total_farms*100:.1f}%)")
            
            print(f"  Average AUC correlation: {np.mean(auc_corrs):.3f}")
            print(f"  Average accuracy correlation: {np.mean(acc_corrs):.3f}")
            print(f"  Average AUC improvement: {np.mean(auc_improvements):.3f}")
            print(f"  Average accuracy improvement: {np.mean(acc_improvements):.3f}")


def plot_all_single_day_summary(all_farm_results):
    """
    Plot summary comparison for all single-day strategies
    """
    print(f"\n{'='*80}")
    print("Plotting Single-Day Strategy Summary Comparison")
    print(f"{'='*80}")
    
    # Filter single-day modeling strategies
    single_day_strategies = [
        'single_day_spectral_lda',
        'single_day_spectral_rf', 
        'single_day_spectral_plsda',
        'single_day_spectral_lstm',
        'single_day_spectral_transformer'
    ]
    
    # Get all farm IDs
    farm_ids = sorted(all_farm_results.keys())
    
    # Collect statistics for all strategies
    strategy_stats = {}
    
    for strategy in single_day_strategies:
        strategy_stats[strategy] = {
            'auc_corrs': [],
            'acc_corrs': [],
            'auc_improvements': [],
            'acc_improvements': [],
            'farms_with_data': 0
        }
        
        for farm_id in farm_ids:
            if strategy in all_farm_results[farm_id]:
                farm_data = all_farm_results[farm_id][strategy]
                
                # Extract time points and performance data
                time_points = []
                auc_values = []
                acc_values = []
                
                for time_point, results in farm_data.items():
                    if isinstance(time_point, (int, float)):
                        time_points.append(time_point)
                        auc_values.append(results['auc_mean'])
                        acc_values.append(results['accuracy_mean'])
                
                if len(time_points) > 1:
                    # Sort by time point
                    sorted_data = sorted(zip(time_points, auc_values, acc_values))
                    time_points, auc_values, acc_values = zip(*sorted_data)
                    
                    # Calculate correlation coefficient
                    auc_corr = np.corrcoef(time_points, auc_values)[0, 1]
                    acc_corr = np.corrcoef(time_points, acc_values)[0, 1]
                    
                    strategy_stats[strategy]['auc_corrs'].append(auc_corr)
                    strategy_stats[strategy]['acc_corrs'].append(acc_corr)
                    
                    # Calculate performance improvement
                    if len(auc_values) >= 10:
                        auc_early = np.mean(auc_values[:5])
                        auc_late = np.mean(auc_values[-5:])
                        auc_improvement = auc_late - auc_early
                        
                        acc_early = np.mean(acc_values[:5])
                        acc_late = np.mean(acc_values[-5:])
                        acc_improvement = acc_late - acc_early
                    else:
                        auc_improvement = 0.0
                        acc_improvement = 0.0
                    
                    strategy_stats[strategy]['auc_improvements'].append(auc_improvement)
                    strategy_stats[strategy]['acc_improvements'].append(acc_improvement)
                    strategy_stats[strategy]['farms_with_data'] += 1
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Subplot 1: AUC correlation comparison across strategies
    ax1 = axes[0, 0]
    strategy_names = []
    auc_corr_means = []
    auc_corr_stds = []
    
    for strategy in single_day_strategies:
        if strategy_stats[strategy]['farms_with_data'] > 0:
            strategy_name = strategy.replace('single_day_spectral_', '').upper()
            strategy_names.append(strategy_name)
            auc_corrs = strategy_stats[strategy]['auc_corrs']
            auc_corr_means.append(np.mean(auc_corrs))
            auc_corr_stds.append(np.std(auc_corrs))
    
    if len(strategy_names) > 0:
        x = np.arange(len(strategy_names))
        ax1.bar(x, auc_corr_means, yerr=auc_corr_stds, alpha=0.7, capsize=5)
        ax1.axhline(y=-0.1, color='k', linestyle='--', alpha=0.5, label='Threshold (-0.1)')
        ax1.set_xlabel('Strategy')
        ax1.set_ylabel('Average AUC Correlation')
        ax1.set_title('AUC Trend Strength Comparison Across Strategies')
        ax1.set_xticks(x)
        ax1.set_xticklabels(strategy_names, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Accuracy correlation comparison across strategies
    ax2 = axes[0, 1]
    acc_corr_means = []
    acc_corr_stds = []
    
    for strategy in single_day_strategies:
        if strategy_stats[strategy]['farms_with_data'] > 0:
            acc_corrs = strategy_stats[strategy]['acc_corrs']
            acc_corr_means.append(np.mean(acc_corrs))
            acc_corr_stds.append(np.std(acc_corrs))
    
    if len(strategy_names) > 0:
        ax2.bar(x, acc_corr_means, yerr=acc_corr_stds, alpha=0.7, capsize=5)
        ax2.axhline(y=-0.1, color='k', linestyle='--', alpha=0.5, label='Threshold (-0.1)')
        ax2.set_xlabel('Strategy')
        ax2.set_ylabel('Average Accuracy Correlation')
        ax2.set_title('Accuracy Trend Strength Comparison Across Strategies')
        ax2.set_xticks(x)
        ax2.set_xticklabels(strategy_names, rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Subplot 3: AUC performance improvement comparison across strategies
    ax3 = axes[1, 0]
    auc_improvement_means = []
    auc_improvement_stds = []
    
    for strategy in single_day_strategies:
        if strategy_stats[strategy]['farms_with_data'] > 0:
            auc_improvements = strategy_stats[strategy]['auc_improvements']
            auc_improvement_means.append(np.mean(auc_improvements))
            auc_improvement_stds.append(np.std(auc_improvements))
    
    if len(strategy_names) > 0:
        ax3.bar(x, auc_improvement_means, yerr=auc_improvement_stds, alpha=0.7, capsize=5)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Threshold (0)')
        ax3.set_xlabel('Strategy')
        ax3.set_ylabel('Average AUC Performance Improvement')
        ax3.set_title('AUC Performance Improvement Comparison Across Strategies')
        ax3.set_xticks(x)
        ax3.set_xticklabels(strategy_names, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Accuracy performance improvement comparison across strategies
    ax4 = axes[1, 1]
    acc_improvement_means = []
    acc_improvement_stds = []
    
    for strategy in single_day_strategies:
        if strategy_stats[strategy]['farms_with_data'] > 0:
            acc_improvements = strategy_stats[strategy]['acc_improvements']
            acc_improvement_means.append(np.mean(acc_improvements))
            acc_improvement_stds.append(np.std(acc_improvements))
    
    if len(strategy_names) > 0:
        ax4.bar(x, acc_improvement_means, yerr=acc_improvement_stds, alpha=0.7, capsize=5)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Threshold (0)')
        ax4.set_xlabel('Strategy')
        ax4.set_ylabel('Average Accuracy Performance Improvement')
        ax4.set_title('Accuracy Performance Improvement Comparison Across Strategies')
        ax4.set_xticks(x)
        ax4.set_xticklabels(strategy_names, rotation=45)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('single_day_all_strategies_summary.png', dpi=300, bbox_inches='tight')
    print("  Summary plot saved as 'single_day_all_strategies_summary.png'")
    
    # Print summary statistics
    print(f"\nSingle-Day Strategy Summary Statistics:")
    for strategy in single_day_strategies:
        if strategy_stats[strategy]['farms_with_data'] > 0:
            strategy_name = strategy.replace('single_day_spectral_', '').upper()
            auc_corrs = strategy_stats[strategy]['auc_corrs']
            acc_corrs = strategy_stats[strategy]['acc_corrs']
            auc_improvements = strategy_stats[strategy]['auc_improvements']
            acc_improvements = strategy_stats[strategy]['acc_improvements']
            
            auc_positive_trend = sum(1 for corr in auc_corrs if corr < -0.1)
            acc_positive_trend = sum(1 for corr in acc_corrs if corr < -0.1)
            auc_positive_improvement = sum(1 for imp in auc_improvements if imp > 0)
            acc_positive_improvement = sum(1 for imp in acc_improvements if imp > 0)
            
            total_farms = len(auc_corrs)
            print(f"\n{strategy_name}:")
            print(f"  Number of farms: {total_farms}")
            print(f"  Farms with correct AUC trend: {auc_positive_trend}/{total_farms} ({auc_positive_trend/total_farms*100:.1f}%)")
            print(f"  Farms with correct accuracy trend: {acc_positive_trend}/{total_farms} ({acc_positive_trend/total_farms*100:.1f}%)")
            print(f"  Farms with AUC improvement: {auc_positive_improvement}/{total_farms} ({auc_positive_improvement/total_farms*100:.1f}%)")
            print(f"  Farms with accuracy improvement: {acc_positive_improvement}/{total_farms} ({acc_positive_improvement/total_farms*100:.1f}%)")
            print(f"  Average AUC correlation: {np.mean(auc_corrs):.3f}")
            print(f"  Average accuracy correlation: {np.mean(acc_corrs):.3f}")


if __name__ == "__main__":
    import sys
    import argparse
    
    # Create command line argument parser
    parser = argparse.ArgumentParser(description='Run enhanced baseline comparison experiment')
    parser.add_argument('--time_window', type=int, default=7, 
                       help='Time window size for two-stage modeling (default: 7)')
    parser.add_argument('--step', type=int, default=1, 
                       help='Sliding window step size (default: 1)')
    parser.add_argument('--num_epochs', type=int, default=100, 
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--input_file', type=str, default='Data/ft-nir.csv',
                       help='Input data file path (default: Data/ft-nir.csv)')
    
    # Parse command line arguments
    args = parser.parse_args()
    
    print(f"Running parameters:")
    print(f"  - Input file: {args.input_file}")
    print(f"  - Time window: {args.time_window}")
    print(f"  - Step: {args.step}")
    print(f"  - Training epochs: {args.num_epochs}")
    
    results, distributions, metrics = modeling(
        time_window=args.time_window, 
        step=args.step, 
        num_epochs=args.num_epochs,
        input_file=args.input_file
    )
    
    # Plot single-day modeling performance trends
    #plot_single_day_trends(results)
    #plot_all_single_day_summary(results)