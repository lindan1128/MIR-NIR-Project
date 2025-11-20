import argparse
import itertools
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             roc_auc_score)
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')


def load_data(file_path: str) -> pd.DataFrame:
    """Load raw data"""
    print(f"Loading data from {file_path} ...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows")
    return df


def get_spectral_columns(df: pd.DataFrame) -> List[str]:
    """Get spectral columns for 2121-2339 cm⁻¹"""
    spectral_cols = []
    for col in df.columns:
        try:
            col_clean = col.strip('"').replace("wl_", "")
            if col_clean.replace('.', '').isdigit() or '.' in col_clean:
                wavelength = float(col_clean)
                if 2121 <= wavelength <= 2339:
                    spectral_cols.append(col)
        except Exception:
            continue
    return spectral_cols


def analyze_farm_specific_variance(df: pd.DataFrame, spectral_cols: List[str]) -> pd.DataFrame:
    """Calculate spectral variance based on training data"""
    feature_variances = {}
    for col in spectral_cols:
        if col in df.columns:
            feature_variances[col] = df[col].var()
    variance_df = pd.DataFrame(
        list(feature_variances.items()),
        columns=['feature', 'variance']
    )
    variance_df = variance_df.sort_values('variance', ascending=False)
    return variance_df


def select_farm_specific_features(variance_df: pd.DataFrame, keep_percentile: float = 0.5) -> List[str]:
    """Select features by variance threshold, keeping specified percentile"""
    if variance_df is None or variance_df.empty:
        return []
    threshold = variance_df['variance'].quantile(1 - keep_percentile)
    selected = variance_df[variance_df['variance'] >= threshold]['feature'].tolist()
    print(f"  Variance selection → original: {len(variance_df)}, kept: {len(selected)} ({keep_percentile*100:.1f}%), threshold: {threshold:.6f}")
    return selected


def match_dim_distribution(mastitis_group: pd.DataFrame,
                           healthy_group: pd.DataFrame,
                           random_state: int = 42,
                           window_start: int = None,
                           window_end: int = None) -> pd.DataFrame:
    """Match healthy samples by DIM"""
    if mastitis_group.empty or healthy_group.empty:
        return pd.DataFrame()

    np.random.seed(random_state)
    mastitis_dim_counts = mastitis_group['DIM'].value_counts()
    matched = []

    if window_start is not None and window_end is not None:
        mastitis_dims = mastitis_group['DIM'].values
        dim_min, dim_max = mastitis_dims.min(), mastitis_dims.max()
        healthy_cow_ids = healthy_group['ANIMAL_SOURCE_ID'].unique()
        suitable = []
        for cow_id in healthy_cow_ids:
            cow_data = healthy_group[healthy_group['ANIMAL_SOURCE_ID'] == cow_id]
            dims = sorted(cow_data['DIM'].values)
            for i in range(len(dims) - 4):
                consecutive = dims[i:i + 5]
                if (all(dim_min <= d <= dim_max for d in consecutive)
                        and consecutive[-1] - consecutive[0] <= 20):
                    selected = []
                    for dim in consecutive:
                        dim_rows = cow_data[cow_data['DIM'] == dim]
                        if len(dim_rows) > 0:
                            selected.append(dim_rows.iloc[0])
                    if len(selected) == 5:
                        suitable.append(pd.DataFrame(selected))
                        break
        if suitable:
            n_needed = min(len(suitable), len(mastitis_group['ANIMAL_SOURCE_ID'].unique()))
            indices = np.random.choice(range(len(suitable)), size=n_needed, replace=False)
            for idx in indices:
                matched.append(suitable[idx])
        else:
            for dim_value, count in mastitis_dim_counts.items():
                healthy_with_dim = healthy_group[healthy_group['DIM'] == dim_value]
                if len(healthy_with_dim) >= count:
                    sampled = healthy_with_dim.sample(n=count, random_state=random_state)
                else:
                    sampled = healthy_with_dim
                matched.append(sampled)
    else:
        for dim_value, count in mastitis_dim_counts.items():
            healthy_with_dim = healthy_group[healthy_group['DIM'] == dim_value]
            if len(healthy_with_dim) >= count:
                sampled = healthy_with_dim.sample(n=count, random_state=random_state)
            else:
                sampled = healthy_with_dim
            matched.append(sampled)

    if matched:
        return pd.concat(matched, ignore_index=True)
    return pd.DataFrame()


def analyze_data_distribution(data: pd.DataFrame,
                              time_point: float,
                              is_window: bool = False,
                              window_start: int = None,
                              window_end: int = None) -> Dict:
    mastitis_data = data[data['group'] == 'mastitis']
    healthy_data = data[data['group'] == 'healthy']

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

    missing_info = {}
    if is_window and window_start is not None and window_end is not None:
        total_cows = len(data['ANIMAL_SOURCE_ID'].unique())
        for interval in range(window_start, window_end + 1):
            interval_data = data[data['interval'] == interval]
            present_cows = len(interval_data['ANIMAL_SOURCE_ID'].unique())
            missing_info[f'missing_day_{interval}'] = total_cows - present_cows

    return {
        'mastitis_dim_stats': mastitis_dim_stats,
        'healthy_dim_stats': healthy_dim_stats,
        'missing_info': missing_info,
        'time_point_info': f"{window_start}-{window_end}" if is_window else str(time_point)
    }


def fill_missing_by_group_dim_parity(data: pd.DataFrame,
                                     feature_cols: List[str]) -> pd.DataFrame:
    print(f"Filling missing values for {len(feature_cols)} features ...")
    filled = data.copy()
    filled['group_dim_parity_key'] = (
        filled['group'].astype(str) + '_' +
        filled['DIM'].astype(str) + '_' +
        filled['LACTATION_NO'].astype(str)
    )

    for col in feature_cols:
        if col not in filled.columns:
            continue
        values = filled[col].values
        missing_mask = np.isnan(values)
        if missing_mask.sum() == 0:
            continue
        group_medians = {}
        for key in filled['group_dim_parity_key'].unique():
            group_values = filled.loc[filled['group_dim_parity_key'] == key, col].values
            if len(group_values) > 0 and not np.isnan(group_values).all():
                group_medians[key] = np.nanmedian(group_values)
        global_median = np.nanmedian(values)
        for idx, is_missing in enumerate(missing_mask):
            if is_missing:
                key = filled.iloc[idx]['group_dim_parity_key']
                if key in group_medians:
                    values[idx] = group_medians[key]
                else:
                    values[idx] = global_median
        filled[col] = values

    filled = filled.drop(columns=['group_dim_parity_key'])
    return filled


def prepare_enhanced_features(data: pd.DataFrame,
                              spectral_cols: List[str],
                              scaler_type: str = 'minmax',
                              content_scaler: object = None,
                              metadata_scaler: object = None) -> Tuple[np.ndarray, np.ndarray, object, object]:
    features_to_fill = ['TOTAL_YIELD'] + spectral_cols
    data_filled = fill_missing_by_group_dim_parity(data, features_to_fill)

    def encode_parity(parity_value: int) -> List[int]:
        if parity_value == 1:
            return [1, 0, 0]
        if parity_value == 2:
            return [0, 1, 0]
        return [0, 0, 1]

    parity_encoded = np.array([encode_parity(p) for p in data_filled['LACTATION_NO']])
    metadata_features = np.column_stack([
        data_filled['DIM'].values.reshape(-1, 1),
        data_filled['TOTAL_YIELD'].values.reshape(-1, 1),
        parity_encoded
    ])
    content_features = data_filled[spectral_cols].values

    if content_scaler is None or metadata_scaler is None:
        if scaler_type == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            metadata_scaler = MinMaxScaler(feature_range=(0, 1))
            content_scaler = MinMaxScaler(feature_range=(0, 1))
        elif scaler_type == 'standard':
            from sklearn.preprocessing import StandardScaler
            metadata_scaler = StandardScaler()
            content_scaler = StandardScaler()
        else:
            raise ValueError("scaler_type must be 'minmax' or 'standard'")
        metadata_scaled = metadata_scaler.fit_transform(metadata_features)
        content_scaled = content_scaler.fit_transform(content_features)
    else:
        metadata_scaled = metadata_scaler.transform(metadata_features)
        content_scaled = content_scaler.transform(content_features)

    print(f"Spectral dim: {content_scaled.shape[1]}, metadata dim: {metadata_scaled.shape[1]}")
    return content_scaled, metadata_scaled, content_scaler, metadata_scaler


class EnhancedTimeSeriesDataset(Dataset):
    def __init__(self, cow_sequences: Dict, min_real_data_points: int = 3):
        self.content_sequences = []
        self.metadata_sequences = []
        self.interval_sequences = []
        self.missing_masks = []
        self.labels = []
        self.cow_ids = []

        filtered = 0
        for cow_id, seq in cow_sequences.items():
            if seq['real_data_count'] >= min_real_data_points:
                self.content_sequences.append(seq['content'])
                self.metadata_sequences.append(seq['metadata'])
                self.interval_sequences.append(seq['intervals'])
                self.missing_masks.append(seq['missing_mask'])
                real_groups = seq['group'][~seq['missing_mask']]
                label = 1 if len(real_groups) > 0 and any('mastiti' in g for g in real_groups) else 0
                self.labels.append(label)
                self.cow_ids.append(cow_id)
            else:
                filtered += 1
        if filtered > 0:
            print(f"Filtered {filtered} cows (insufficient real data points)")
        if len(self.labels) > 0:
            label_counts = pd.Series(self.labels).value_counts()
            print(f"Dataset label distribution: {label_counts.to_dict()}")
        else:
            print("Warning: empty dataset")

    def __len__(self) -> int:
        return len(self.content_sequences)

    def __getitem__(self, idx: int):
        return (
            torch.FloatTensor(self.content_sequences[idx]),
            torch.FloatTensor(self.metadata_sequences[idx]),
            torch.FloatTensor(self.interval_sequences[idx]),
            torch.BoolTensor(self.missing_masks[idx]),
            torch.LongTensor([self.labels[idx]]),
            self.cow_ids[idx]
        )


def enhanced_collate_fn(batch):
    content_sequences, metadata_sequences, interval_sequences, missing_masks, labels, cow_ids = zip(*batch)
    content_tensor = torch.stack(content_sequences)
    metadata_tensor = torch.stack(metadata_sequences)
    interval_tensor = torch.stack(interval_sequences)
    missing_tensor = torch.stack(missing_masks)
    label_tensor = torch.LongTensor([label.item() for label in labels])
    return content_tensor, metadata_tensor, interval_tensor, ~missing_tensor, label_tensor, cow_ids


def create_enhanced_cow_sequences_for_two_stage(mastitis_data: pd.DataFrame,
                                               healthy_data: pd.DataFrame,
                                               spectral_cols: List[str],
                                               scaler_type: str = 'minmax',
                                               max_interval: int = 30,
                                               min_interval: int = 0,
                                               content_scaler: object = None,
                                               metadata_scaler: object = None):
    cow_sequences = {}
    all_features_list = []
    full_timeline = list(range(max_interval, min_interval - 1, -1))
    timeline_length = len(full_timeline)

    mastitis_cow_sequences = {}
    for cow_id in mastitis_data['ANIMAL_SOURCE_ID'].unique():
        cow_df = mastitis_data[mastitis_data['ANIMAL_SOURCE_ID'] == cow_id].copy()
        cow_df = cow_df.sort_values('DATE_KEY')
        if len(cow_df) > 0:
            all_features_list.append(cow_df)
    if not all_features_list:
        return {}, content_scaler, metadata_scaler
    all_data_combined = pd.concat(all_features_list, ignore_index=True)
    print(f"  Standardising mastitis data ({len(all_data_combined)} samples)")

    content_scaled, metadata_scaled, content_scaler, metadata_scaler = prepare_enhanced_features(
        all_data_combined, spectral_cols, scaler_type, content_scaler, metadata_scaler
    )

    start_idx = 0
    for cow_id in mastitis_data['ANIMAL_SOURCE_ID'].unique():
        cow_df = mastitis_data[mastitis_data['ANIMAL_SOURCE_ID'] == cow_id].copy()
        cow_df = cow_df.sort_values('DATE_KEY')
        if len(cow_df) == 0:
            continue
        end_idx = start_idx + len(cow_df)
        existing_intervals = cow_df['interval'].values
        content = content_scaled[start_idx:end_idx]
        metadata = metadata_scaled[start_idx:end_idx]
        groups = cow_df['group'].values
        dates = cow_df['DATE_KEY'].values

        content_dim = content.shape[1]
        metadata_dim = metadata.shape[1]
        full_content = np.zeros((timeline_length, content_dim))
        full_metadata = np.zeros((timeline_length, metadata_dim))
        full_groups = np.array(['missing'] * timeline_length)
        full_dates = np.array([f'missing_day_{interval}' for interval in full_timeline])
        full_masks = np.ones(timeline_length, dtype=bool)

        for idx, interval in enumerate(existing_intervals):
            if interval in full_timeline:
                t_idx = full_timeline.index(interval)
                full_content[t_idx] = content[idx]
                full_metadata[t_idx] = metadata[idx]
                full_groups[t_idx] = groups[idx]
                full_dates[t_idx] = dates[idx]
                full_masks[t_idx] = False

        if full_masks.sum() > 0:
            metadata_mean = metadata.mean(axis=0)
            missing_indices = np.where(full_masks)[0]
            for idx in missing_indices:
                full_metadata[idx] = metadata_mean

        cow_sequences[cow_id] = {
            'dates': full_dates,
            'intervals': np.array(full_timeline),
            'content': full_content,
            'metadata': full_metadata,
            'group': full_groups,
            'missing_mask': full_masks,
            'real_data_count': (~full_masks).sum(),
            'cow_type': 'mastitis'
        }
        mastitis_cow_sequences[cow_id] = cow_sequences[cow_id]
        start_idx = end_idx

    if healthy_data is not None and not healthy_data.empty:
        healthy_features, healthy_metadata, _, _ = prepare_enhanced_features(
            healthy_data, spectral_cols, scaler_type,
            content_scaler=content_scaler, metadata_scaler=metadata_scaler
        )
        start_idx = 0
        for cow_id in healthy_data['ANIMAL_SOURCE_ID'].unique():
            cow_df = healthy_data[healthy_data['ANIMAL_SOURCE_ID'] == cow_id].copy()
            cow_df = cow_df.sort_values('DATE_KEY')
            end_idx = start_idx + len(cow_df)
            cow_content = healthy_features[start_idx:end_idx]
            cow_metadata = healthy_metadata[start_idx:end_idx]
            cow_dates = cow_df['DATE_KEY'].values
            cow_dims = cow_df['DIM'].values

            content_dim = cow_content.shape[1]
            metadata_dim = cow_metadata.shape[1]
            full_content = np.zeros((timeline_length, content_dim))
            full_metadata = np.zeros((timeline_length, metadata_dim))
            full_groups = np.array(['missing'] * timeline_length)
            full_dates = np.array([f'missing_day_{interval}' for interval in full_timeline])
            full_masks = np.ones(timeline_length, dtype=bool)

            for idx, dim in enumerate(cow_dims):
                timeline_idx = idx
                timeline_idx = max(0, min(timeline_idx, timeline_length - 1))
                full_content[timeline_idx] = cow_content[idx]
                full_metadata[timeline_idx] = cow_metadata[idx]
                full_groups[timeline_idx] = 'healthy'
                full_dates[timeline_idx] = cow_dates[idx]
                full_masks[timeline_idx] = False

            if 0 < full_masks.sum() < timeline_length:
                metadata_mean = cow_metadata.mean(axis=0)
                missing_indices = np.where(full_masks)[0]
                for idx in missing_indices:
                    full_metadata[idx] = metadata_mean

            cow_sequences[f"healthy_{cow_id}"] = {
                'dates': full_dates,
                'intervals': np.array(full_timeline),
                'content': full_content,
                'metadata': full_metadata,
                'group': full_groups,
                'missing_mask': full_masks,
                'real_data_count': (~full_masks).sum(),
                'cow_type': 'healthy'
            }
            start_idx = end_idx

    print(f"Created {len(cow_sequences)} sequences (mastitis: {len(mastitis_cow_sequences)}, healthy: {len(cow_sequences) - len(mastitis_cow_sequences)})")
    return cow_sequences, content_scaler, metadata_scaler


class EnhancedTwoStageModel(nn.Module):
    def __init__(self,
                 spectral_dim: int,
                 metadata_dim: int,
                 spectral_embedding_dim: int = 64,
                 nhead: int = 8):
        super().__init__()
        self.spectral_encoder = SpectralEncoder(spectral_dim, spectral_embedding_dim, encoder_type='transformer')
        fusion_dim = spectral_embedding_dim + metadata_dim
        if fusion_dim % nhead != 0:
            new_dim = ((fusion_dim // nhead) + 1) * nhead
            self.projection = nn.Linear(fusion_dim, new_dim)
            fusion_dim = new_dim
        else:
            self.projection = None
        self.max_seq_len = 100
        self.pos_embedding = nn.Parameter(torch.randn(1, self.max_seq_len, fusion_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fusion_dim,
            nhead=nhead,
            dim_feedforward=fusion_dim,
            dropout=0.1,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.temporal_attention = nn.Linear(fusion_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 2)
        )

    def forward(self, spectral_sequences, metadata_sequences, mask=None):
        batch_size, seq_len, spectral_dim = spectral_sequences.shape
        spectral_flat = spectral_sequences.view(-1, spectral_dim)
        spectral_embeddings = self.spectral_encoder(spectral_flat)
        spectral_embeddings = spectral_embeddings.view(batch_size, seq_len, -1)
        fused = torch.cat([spectral_embeddings, metadata_sequences], dim=-1)
        if self.projection is not None:
            fused = self.projection(fused)
        if seq_len <= self.max_seq_len:
            fused = fused + self.pos_embedding[:, :seq_len, :]
        attn_mask = None
        if mask is not None:
            attn_mask = (mask == 0)
        temporal_out = self.temporal_transformer(fused, src_key_padding_mask=attn_mask)
        if mask is not None:
            attn_scores = self.temporal_attention(temporal_out).squeeze(-1)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
            attn_weights = torch.softmax(attn_scores, dim=1)
        else:
            attn_scores = self.temporal_attention(temporal_out).squeeze(-1)
            attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.sum(temporal_out * attn_weights.unsqueeze(-1), dim=1)
        return self.classifier(pooled)


class SpectralEncoder(nn.Module):
    def __init__(self, spectral_dim: int, embedding_dim: int = 64, encoder_type: str = 'transformer'):
        super().__init__()
        self.encoder_type = encoder_type
        self.embedding_dim = embedding_dim
        if encoder_type == 'transformer':
            self.input_projection = nn.Linear(spectral_dim, embedding_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=embedding_dim * 2,
                dropout=0.1,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        elif encoder_type == 'mlp':
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
        else:
            raise ValueError("Unsupported encoder_type")

    def forward(self, x):
        if self.encoder_type == 'transformer':
            x = self.input_projection(x.unsqueeze(1))
            x = self.transformer(x)
            return x.squeeze(1)
        return self.encoder(x)


def train_two_stage_model(model: nn.Module,
                          train_loader: DataLoader,
                          val_loader: DataLoader,
                          device: torch.device,
                          num_epochs: int = 100,
                          class_weights: List[float] = None) -> Tuple[float, float, float, float, List[float], List[float]]:
    if class_weights is not None:
        class_weights = torch.FloatTensor(class_weights).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    epoch_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        batch_count = 0
        for content_seq, metadata_seq, interval_seq, masks, labels, _ in train_loader:
            content_seq = content_seq.to(device)
            metadata_seq = metadata_seq.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            if content_seq.size(0) <= 1:
                continue
            optimizer.zero_grad()
            outputs = model(content_seq, metadata_seq, masks)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()
            batch_count += 1
        epoch_loss = running_loss / batch_count if batch_count > 0 else 0.0
        epoch_losses.append(epoch_loss)

        model.eval()
        val_running = 0.0
        val_batches = 0
        with torch.no_grad():
            for content_seq, metadata_seq, interval_seq, masks, labels, _ in val_loader:
                content_seq = content_seq.to(device)
                metadata_seq = metadata_seq.to(device)
                masks = masks.to(device)
                labels = labels.to(device)
                if content_seq.size(0) == 0:
                    continue
                outputs = model(content_seq, metadata_seq, masks)
                loss = criterion(outputs, labels)
                val_running += loss.item()
                val_batches += 1
        val_loss = val_running / val_batches if val_batches > 0 else 0.0
        val_losses.append(val_loss)
        print(f"    Epoch {epoch+1}/{num_epochs} → train loss {epoch_loss:.4f}, val loss {val_loss:.4f}")

    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    with torch.no_grad():
        for content_seq, metadata_seq, interval_seq, masks, labels, _ in val_loader:
            content_seq = content_seq.to(device)
            metadata_seq = metadata_seq.to(device)
            masks = masks.to(device)
            labels = labels.to(device)
            if content_seq.size(0) == 0:
                continue
            outputs = model(content_seq, metadata_seq, masks)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    if len(all_labels) == 0 or len(set(all_labels)) < 2:
        return 0.0, 0.0, 0.0, 0.0, epoch_losses, val_losses
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    auc = roc_auc_score(all_labels, np.array(all_probabilities)[:, 1])
    return accuracy, precision, recall, auc, epoch_losses, val_losses


def save_validation_results(summary_records: List[Dict],
                            detailed_records: List[Dict],
                            distribution_records: List[Dict],
                            loss_records: List[Dict],
                            time_window: int,
                            step: int) -> None:
    summary_df = pd.DataFrame(summary_records)
    summary_path = f'validation_summary_results_w{time_window}_s{step}.csv'
    summary_df.to_csv(summary_path, index=False)

    detailed_df = pd.DataFrame(detailed_records)
    detailed_path = f'validation_detailed_metrics_w{time_window}_s{step}.csv'
    detailed_df.to_csv(detailed_path, index=False)

    distribution_df = pd.DataFrame(distribution_records)
    distribution_path = f'validation_data_distributions_w{time_window}_s{step}.csv'
    distribution_df.to_csv(distribution_path, index=False)

    loss_df = pd.DataFrame(loss_records)
    loss_path = f'validation_training_losses_w{time_window}_s{step}.csv'
    loss_df.to_csv(loss_path, index=False)

    print("Saved CSVs:")
    print(f"  - {summary_path}")
    print(f"  - {detailed_path}")
    print(f"  - {distribution_path}")
    print(f"  - {loss_path}")


def print_validation_summary(results_by_val: Dict) -> None:
    print("\n" + "=" * 80)
    print("Leave-One-Study Validation Summary")
    print("=" * 80)
    for val_farm, combos in results_by_val.items():
        print(f"\nValidation farm {val_farm}:")
        for combo_key, windows in combos.items():
            if not windows:
                continue
            acc_values = [metrics['accuracy'] for metrics in windows.values()]
            auc_values = [metrics['auc'] for metrics in windows.values()]
            print(f"  Train farms {combo_key:<20s} → ACC {np.mean(acc_values):.3f}, AUC {np.mean(auc_values):.3f}")
    print("\n" + "=" * 80)


def modeling_leave_one_study(time_window: int = 5,
                             step: int = 1,
                             num_epochs: int = 100,
                             input_file: str = 'Data/ft-mir-pathogen_neg.csv',
                             keep_percentile: float = 0.5):
    df = load_data(input_file)
    spectral_cols = get_spectral_columns(df)
    print(f"Found {len(spectral_cols)} spectral features")

    farms = sorted(df['FARM_ID'].unique())
    print(f"Detected farms: {farms}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    results_by_val: Dict[str, Dict[str, Dict[float, Dict]]] = {}
    detailed_records: List[Dict] = []
    distribution_records: List[Dict] = []
    loss_records: List[Dict] = []

    for val_farm in farms:
        print("\n" + "=" * 80)
        print(f"Validation farm: {val_farm}")
        print("=" * 80)
        results_by_val[val_farm] = {}
        val_df = df[df['FARM_ID'] == val_farm].copy()
        val_mastitis = val_df[val_df['group'] == 'mastitis'].copy()
        val_healthy = val_df[val_df['group'] == 'healthy'].copy()

        train_pool = [farm for farm in farms if farm != val_farm]
        if not train_pool:
            continue

        for train_size in range(1, len(train_pool) + 1):
            for combo in itertools.combinations(train_pool, train_size):
                combo_key = '_'.join(map(str, combo))
                print(f"\n--- Training farms: {combo} (count={len(combo)}), validation: {val_farm} ---")
                train_df = df[df['FARM_ID'].isin(combo)].copy()
                if train_df.empty:
                    print("  Skip: empty training data")
                    continue
                if len(train_df[train_df['group'] == 'mastitis']) < 5:
                    print("  Skip: insufficient mastitis samples in training data")
                    continue
                if len(val_mastitis) < 3:
                    print("  Skip: insufficient mastitis samples in validation farm")
                    continue

                variance_df = analyze_farm_specific_variance(train_df, spectral_cols)
                selected_features = select_farm_specific_features(variance_df, keep_percentile)
                if not selected_features:
                    print("  Skip: no selected features")
                    continue

                train_mastitis = train_df[train_df['group'] == 'mastitis'].copy()
                train_healthy = train_df[train_df['group'] == 'healthy'].copy()

                windows = []
                for start in range(30 - time_window + 1, -1, -step):
                    end = start + time_window - 1
                    if end <= 30:
                        windows.append((start, end))

                combo_results = {}
                for window_start, window_end in windows:
                    window_center = (window_start + window_end) / 2
                    print(f"  Window {window_start}-{window_end} (center={window_center})")

                    train_window_mastitis = train_mastitis[(train_mastitis['interval'] >= window_start) &
                                                           (train_mastitis['interval'] <= window_end)].copy()
                    val_window_mastitis = val_mastitis[(val_mastitis['interval'] >= window_start) &
                                                       (val_mastitis['interval'] <= window_end)].copy()
                    if len(train_window_mastitis) < 5 or len(val_window_mastitis) < 3:
                        print("    Skip window: insufficient mastitis samples")
                        continue

                    train_matched_healthy = match_dim_distribution(
                        train_window_mastitis, train_healthy,
                        random_state=42,
                        window_start=window_start,
                        window_end=window_end
                    )
                    val_matched_healthy = match_dim_distribution(
                        val_window_mastitis, val_healthy,
                        random_state=42,
                        window_start=window_start,
                        window_end=window_end
                    )

                    train_combined = pd.concat([train_window_mastitis, train_matched_healthy], ignore_index=True)
                    val_combined = pd.concat([val_window_mastitis, val_matched_healthy], ignore_index=True)

                    train_dist = analyze_data_distribution(train_combined, window_center, True, window_start, window_end)
                    train_record = {
                        'validation_farm': val_farm,
                        'train_farms': combo_key,
                        'train_farm_count': len(combo),
                        'set_type': 'train',
                        'time_window_center': window_center,
                        'mastitis_count': train_dist['mastitis_dim_stats']['count'],
                        'healthy_count': train_dist['healthy_dim_stats']['count'],
                        'mastitis_dim_mean': train_dist['mastitis_dim_stats']['mean'],
                        'mastitis_dim_std': train_dist['mastitis_dim_stats']['std'],
                        'healthy_dim_mean': train_dist['healthy_dim_stats']['mean'],
                        'healthy_dim_std': train_dist['healthy_dim_stats']['std'],
                        'time_window_start': window_start,
                        'time_window_end': window_end
                    }
                    for interval_key, missing_value in train_dist['missing_info'].items():
                        train_record[f'missing_{interval_key}'] = missing_value
                    distribution_records.append(train_record)

                    val_dist = analyze_data_distribution(val_combined, window_center, True, window_start, window_end)
                    val_record = {
                        'validation_farm': val_farm,
                        'train_farms': combo_key,
                        'train_farm_count': len(combo),
                        'set_type': 'validation',
                        'time_window_center': window_center,
                        'mastitis_count': val_dist['mastitis_dim_stats']['count'],
                        'healthy_count': val_dist['healthy_dim_stats']['count'],
                        'mastitis_dim_mean': val_dist['mastitis_dim_stats']['mean'],
                        'mastitis_dim_std': val_dist['mastitis_dim_stats']['std'],
                        'healthy_dim_mean': val_dist['healthy_dim_stats']['mean'],
                        'healthy_dim_std': val_dist['healthy_dim_stats']['std'],
                        'time_window_start': window_start,
                        'time_window_end': window_end
                    }
                    for interval_key, missing_value in val_dist['missing_info'].items():
                        val_record[f'missing_{interval_key}'] = missing_value
                    distribution_records.append(val_record)

                    train_sequences, content_scaler, metadata_scaler = create_enhanced_cow_sequences_for_two_stage(
                        train_window_mastitis, train_healthy, selected_features,
                        max_interval=window_end, min_interval=window_start
                    )
                    val_sequences, _, _ = create_enhanced_cow_sequences_for_two_stage(
                        val_window_mastitis, val_healthy, selected_features,
                        max_interval=window_end, min_interval=window_start,
                        content_scaler=content_scaler,
                        metadata_scaler=metadata_scaler
                    )

                    if not train_sequences or not val_sequences:
                        print("    Skip window: empty sequence set")
                        continue

                    window_size = window_end - window_start + 1
                    min_real_data_points = max(1, int(window_size * 0.25))

                    train_dataset = EnhancedTimeSeriesDataset(train_sequences, min_real_data_points)
                    val_dataset = EnhancedTimeSeriesDataset(val_sequences, min_real_data_points)
                    if len(train_dataset) == 0 or len(val_dataset) == 0:
                        print("    Skip window: empty dataset after filtering")
                        continue

                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=16,
                        shuffle=True,
                        collate_fn=enhanced_collate_fn
                    )
                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=16,
                        shuffle=False,
                        collate_fn=enhanced_collate_fn
                    )

                    content_dim = train_dataset.content_sequences[0].shape[1]
                    metadata_dim = train_dataset.metadata_sequences[0].shape[1]

                    # Calculate class weights for weighted loss
                    train_labels = train_dataset.labels
                    class_counts = np.bincount(train_labels)
                    total_samples = len(train_labels)
                    class_weights = [total_samples / (2 * count) if count > 0 else 0.0 for count in class_counts]

                    model = EnhancedTwoStageModel(
                        spectral_dim=content_dim,
                        metadata_dim=metadata_dim,
                        spectral_embedding_dim=64,
                        nhead=8
                    ).to(device)

                    accuracy, precision, recall, auc, epoch_losses, val_losses = train_two_stage_model(
                        model, train_loader, val_loader, device, num_epochs=num_epochs, class_weights=class_weights
                    )

                    detailed_records.append({
                        'validation_farm': val_farm,
                        'train_farms': combo_key,
                        'train_farm_count': len(combo),
                        'time_window_center': window_center,
                        'time_window_start': window_start,
                        'time_window_end': window_end,
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'auc': auc,
                        'iteration': 0
                    })

                    loss_records.extend([{
                        'validation_farm': val_farm,
                        'train_farms': combo_key,
                        'train_farm_count': len(combo),
                        'time_window_center': window_center,
                        'epoch': idx + 1,
                        'train_loss': train_loss,
                        'val_loss': val_loss
                    } for idx, (train_loss, val_loss) in enumerate(zip(epoch_losses, val_losses))])

                    combo_results.setdefault(combo_key, {})[window_center] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'auc': auc
                    }

                results_by_val[val_farm].update(combo_results)

    summary_records = []
    for val_farm, combos in results_by_val.items():
        for combo_key, windows in combos.items():
            if not windows:
                continue
            accuracy_mean = np.mean([metrics['accuracy'] for metrics in windows.values()])
            precision_mean = np.mean([metrics['precision'] for metrics in windows.values()])
            recall_mean = np.mean([metrics['recall'] for metrics in windows.values()])
            auc_mean = np.mean([metrics['auc'] for metrics in windows.values()])
            summary_records.append({
                'validation_farm': val_farm,
                'train_farms': combo_key,
                'train_farm_count': combo_key.count('_') + 1 if combo_key else 0,
                'accuracy_mean': accuracy_mean,
                'precision_mean': precision_mean,
                'recall_mean': recall_mean,
                'auc_mean': auc_mean
            })

    save_validation_results(summary_records, detailed_records, distribution_records, loss_records, time_window, step)
    print_validation_summary(results_by_val)
    return results_by_val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Leave-one-study validation for two-stage transformer-transformer model')
    parser.add_argument('--time_window', type=int, default=7, help='Time window length for two-stage modeling (default: 7)')
    parser.add_argument('--step', type=int, default=1, help='Sliding step size (default: 1)')
    parser.add_argument('--num_epochs', type=int, default=100, help='Training epochs (default: 100)')
    parser.add_argument('--input_file', type=str, default='Data/ft-nir.csv', help='Input CSV file path')
    parser.add_argument('--keep_percentile', type=float, default=0.5, help='Variance feature retention percentile (default: 0.5)')

    args = parser.parse_args()

    print("Running leave-one-study validation with parameters:")
    print(f"  Input file: {args.input_file}")
    print(f"  Time window: {args.time_window}")
    print(f"  Step: {args.step}")
    print(f"  Num epochs: {args.num_epochs}")
    print(f"  Keep percentile: {args.keep_percentile}")

    modeling_leave_one_study(
        time_window=args.time_window,
        step=args.step,
        num_epochs=args.num_epochs,
        input_file=args.input_file,
        keep_percentile=args.keep_percentile
    )
