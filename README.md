### Project structure

	Main/
	├── Codes/                        # Folder for codes for modeling
	│   ├── mir_modeling.py           # codes for mir modeling
	│   ├── nir_modeling.py           # codes for nir modeling
	│   ├── lofo_modeling.py          # codes for nir LOFO validation
	├── README.md                     # Readme file
	└── requirements.txt              # Dependencies

  ### Usage

    # FT-MIR modeling
    python mir_modeling.py --input_file INPUT_FILE [options]

    Input Parameters:
    --input_file INPUT_FILE: Path to the input CSV file containing FT-MIR spectral data
    --time_window TIME_WINDOW: Time window size for two-stage modeling in days (default: 7)
    --step STEP: Sliding window step size in days (default: 10)
    --num_epochs NUM_EPOCHS: Number of training epochs for deep learning models (default: 100)

    Output files:
    enhanced_data_distributions_w{time_window}_s{step}.csv: Data distribution and sample statistics for each strategy and time point
    enhanced_detailed_metrics_w{time_window}_s{step}.csv: Detailed performance metrics (accuracy, precision, recall, AUC) for each iteration
    enhanced_summary_results_w{time_window}_s{step}.csv: Summary results with mean and standard deviation for each strategy and time point
    enhanced_training_losses_w{time_window}_s{step}.csv: Training and validation loss records for two-stage strategies (epoch-by-epoch)

    # NIR modeling
    python nir_modeling.py --input_file INPUT_FILE [options]

    Input Parameters:
    --input_file INPUT_FILE: Path to the input CSV file containing NIR spectral data
    --time_window TIME_WINDOW: Time window size for two-stage modeling in days (default: 7)
    --step STEP: Sliding window step size in days (default: 1)
    --num_epochs NUM_EPOCHS: Number of training epochs for deep learning models (default: 100)

    Output files:
    enhanced_data_distributions_w{time_window}_s{step}.csv: Data distribution and sample statistics for each farm, strategy, and time point (includes Farm column)
    enhanced_detailed_metrics_w{time_window}_s{step}.csv: Detailed performance metrics for each farm, strategy, time point, and iteration (includes Farm column)
    enhanced_summary_results_w{time_window}_s{step}.csv: Summary results with mean and standard deviation for each farm, strategy, and time point (includes Farm column)
    enhanced_training_losses_w{time_window}_s{step}.csv: Training and validation loss records for two-stage strategies (includes Farm column)

    # NIR LOFO validation
    python lofo_modeling.py --input_file INPUT_FILE [options]

    Input Parameters:
    --input_file INPUT_FILE: Path to the input CSV file containing NIR spectral data
    --time_window TIME_WINDOW: Time window length for two-stage modeling in days (default: 7)
    --step STEP: Sliding window step size in days (default: 1)
    --num_epochs NUM_EPOCHS: Number of training epochs (default: 100)

    Output files:
    Summary CSV: - validationfarm, trainingfarms, trainingfarmcount, timewindowcenter - accuracymean, accuracystd, precisionmean, recallmean, auc_mean
    Detailed Metrics CSV: - validationfarm, trainingfarms, trainingfarmcount, timewindowcenter, iteration - accuracy, precision, recall, auc
    Distributions CSV: - validationfarm, trainingfarms, trainingfarmcount, timewindowcenter - trainmastitiscount, trainhealthycount, valmastitiscount, valhealthycount - trainmastitisdimmean, trainmastitisdimstd, etc.
    Losses CSV: - validationfarm, trainingfarms, trainingfarmcount, timewindowcenter, iteration, epoch - trainloss, valloss

  ### Pseudocode for mir-modeling

    Initialization:
    Load data from input_file
    Extract spectral columns (1000-3000 cm⁻¹, excluding 1800-2800 and 1585-1700)
    Define milk composition columns: ["totalfa", "protein", "lactose", ...]
    Separate data into mastitis and healthy groups
    Initialize device (CUDA if available, else CPU)
    
    Define experiment strategies:
        Single-day strategies:
            - single_day_spectral_plsda
            - single_day_spectral_rf
            - single_day_spectral_lstm
            - single_day_spectral_transformer
        Two-stage strategies:
            - two_stage_spectral_transformer
            - two_stage_composition_transformer
            - two_stage_composition_transformer_3
            - two_stage_spectral_transformer_lstm
            - two_stage_spectral_lstm_transformer
    
    Initialize result storage:
        all_results: Performance metrics
        data_distributions: Data distribution statistics
        detailed_metrics: Detailed metrics for each iteration
        training_losses: Training loss records
        model_info: Model size and running time

    Main Loop - For each strategy:
        If strategy type == 'single_day':
            For each time_point from 30 to 0 days ago:
                Filter mastitis data for current time_point
                If sample count < 5:
                    Skip to next time_point
            
            For each iteration (1 to 20):
                Match healthy samples by DIM distribution
                Combine mastitis and matched healthy data
                
                If iteration == 0:
                    Analyze and record data distribution
                
                Select feature columns based on strategy:
                    - spectral: Use spectral columns
                    - composition: Use milk composition columns
                    - composition_3: Use 3 main components only
                
                Perform 5-fold cross-validation:
                    For each fold:
                        Split data into train and test sets
                        
                        If model is deep learning (transformer/lstm):
                            Create EnhancedSingleDayDataset for train and test
                            Calculate class weights for weighted loss
                            Create DataLoader with batch_size=16
                            Initialize model (SingleDayTransformer or SingleDayLSTM)
                            Train and evaluate using train_and_evaluate_single_day
                            Calculate model size and running time
                        Else (traditional ML):
                            Create EnhancedSingleDayDataset for train and test
                            Extract features and labels
                            Train model (RandomForest/PLS-DA/LDA)
                            Predict on test set
                            Calculate metrics (accuracy, precision, recall, AUC)
                        
                        Store fold metrics
                    
                    Average metrics across folds
                    Store iteration results
            
            Calculate mean and std across iterations
            Store results for current time_point
    
    Else (strategy type == 'two_stage'):
        Generate sliding windows:
            For start from (30 - time_window + 1) to 0, step = -step:
                end = start + time_window - 1
                If end <= 30:
                    Add window (start, end)
        
        For each window (window_start, window_end):
            Filter mastitis data within window
            If sample count < 10:
                Skip to next window
            
            For each iteration (1 to 20):
                Match healthy samples by DIM distribution (consecutive DIM matching)
                Combine mastitis and matched healthy data
                
                If iteration == 0:
                    Analyze and record data distribution
                
                Select feature columns based on strategy
                
                Perform 5-fold cross-validation on cow IDs:
                    Split all cow IDs into train and test sets
                    
                    For each fold:
                        Separate train/test data by cow ID
                        Create scaler using only training data
                        Create enhanced cow sequences for two-stage modeling:
                            - Process mastitis cow time series
                            - Process healthy cow time series (DIM matching)
                            - Create complete timeline sequences
                            - Handle missing values
                        Apply training scaler to test data
                        Merge train and test sequences
                        Create EnhancedTimeSeriesDataset
                        Re-split dataset by cow ID
                        
                        Calculate class weights for weighted loss
                        Create DataLoader with batch_size=16
                        Initialize EnhancedTwoStageModel:
                            - transformer_transformer
                            - transformer_lstm_temporal
                            - lstm_transformer
                        Train and evaluate using train_and_evaluate_two_stage
                        Calculate model size and running time
                        Store fold metrics
                    
                    Average metrics across folds
                    Store iteration results
            
            Calculate mean and std across iterations
            Store results for current window

    Save Results:
        Save to CSV files:
        - enhanced_data_distributions_w{time_window}_s{step}.csv
        - enhanced_detailed_metrics_w{time_window}_s{step}.csv
        - enhanced_summary_results_w{time_window}_s{step}.csv
        - enhanced_training_losses_w{time_window}_s{step}.csv

    Print Summary:
        Print enhanced experiment summary:
        - Spectral vs milk composition comparison
        - Single-day modeling architecture comparison
        - Temporal information value validation
        - Two-stage model architecture comparison
        - Overall best method

  ### Pseudocode for nir-modeling

    Initialization:
    Load data from input_file
    Extract spectral columns (2121-2339 cm⁻¹)
    Define milk composition columns: ["FAT", "PROTEIN", "LACTOSE"]
    Get all unique FARM_IDs
    Initialize device (CUDA if available, else CPU)
    
    Define experiment strategies:
        Single-day strategies:
            - single_day_spectral_plsda
            - single_day_spectral_rf
            - single_day_spectral_lstm
            - single_day_spectral_transformer
        Two-stage strategies:
            - two_stage_spectral_transformer
            - two_stage_composition_transformer_3
            - two_stage_spectral_transformer_lstm
            - two_stage_spectral_lstm_transformer
    
    Initialize farm-level result storage:
        all_farm_results: Results for each farm
        all_farm_data_distributions: Data distributions for each farm
        all_farm_detailed_metrics: Detailed metrics for each farm
        all_farm_training_losses: Training losses for each farm
        all_farm_model_info: Model info for each farm

    Main Loop - For each farm:
        Filter data for current farm (FARM_ID)
        If sample count < 5:
            Skip to next farm
    
    Print farm statistics:
        - Total samples
        - Mastitis samples
        - Healthy samples
    
    Farm-specific feature selection:
        Analyze spectral feature variance for current farm
        Select features based on variance percentile (keep_percentile=1.0)
        If no valid features:
            Skip to next farm
    
    Separate data into mastitis and healthy groups
    
    Initialize farm-level result storage
    
    For each strategy:
        If strategy type == 'single_day':
            For each time_point from 30 to 0 days ago:
                Filter mastitis data for current time_point
                If sample count < 2:
                    Skip to next time_point
                
                For each iteration (1 to 2):
                    Match healthy samples by DIM distribution
                    Combine mastitis and matched healthy data
                    
                    If iteration == 0:
                        Analyze and record data distribution
                    
                    Select feature columns:
                        - spectral: Use farm-specific features
                        - composition_3: Use 3 main components
                    
                    Perform 5-fold cross-validation:
                        For each fold:
                            Split data into train and test sets
                            
                            If model is deep learning (transformer/lstm):
                                Create EnhancedSingleDayDataset
                                Calculate class weights for weighted loss
                                Create DataLoader with batch_size=16
                                Initialize model
                                Train and evaluate
                                Store model info (first iteration, first fold only)
                            Else (traditional ML):
                                Create EnhancedSingleDayDataset
                                Extract features and labels
                                Train model (RandomForest/PLS-DA/LDA)
                                Predict and calculate metrics
                            
                            Store fold metrics
                        
                        Average metrics across folds
                        Store iteration results
                
                Calculate mean and std across iterations
                Store results for current time_point
        
        Else (strategy type == 'two_stage'):
            Generate sliding windows:
                For start from (30 - time_window + 1) to 0, step = -step:
                    end = start + time_window - 1
                    If end <= 30:
                        Add window (start, end)
            
            For each window (window_start, window_end):
                Filter mastitis data within window
                If sample count < 10:
                    Skip to next window
                
                For each iteration (1 to 2):
                    Match healthy samples by DIM distribution
                    Combine mastitis and matched healthy data
                    
                    If iteration == 0:
                        Analyze and record data distribution
                    
                    Select feature columns:
                        - spectral: Use farm-specific features
                        - composition_3: Use 3 main components
                    
                    Perform 5-fold cross-validation on cow IDs:
                        Split all cow IDs into train and test sets
                        
                        For each fold:
                            Separate train/test data by cow ID
                            Create scaler using only training data
                            Create enhanced cow sequences
                            Apply training scaler to test data
                            Create EnhancedTimeSeriesDataset
                            Re-split dataset by cow ID
                            
                            Calculate class weights for weighted loss
                            Create DataLoader with batch_size=16
                            Initialize EnhancedTwoStageModel
                            Train and evaluate
                            Store model info (first iteration, first fold only)
                            Store fold metrics
                        
                        Average metrics across folds
                        Store iteration results
                
                Calculate mean and std across iterations
                Store results for current window
        
        Store results for current farm

    Save Results:
        Save to CSV files (with Farm column):
        - enhanced_data_distributions_w{time_window}_s{step}.csv
        - enhanced_detailed_metrics_w{time_window}_s{step}.csv
        - enhanced_summary_results_w{time_window}_s{step}.csv
        - enhanced_training_losses_w{time_window}_s{step}.csv

    Print Summary:
        Print enhanced experiment summary by farm:
        - For each farm:
            - Spectral vs milk composition comparison
            - Single-day modeling architecture comparison
            - Temporal information value validation
            - Two-stage model architecture comparison
            - Overall best method

  ### Pseudocode for nir-LOFO validation

    Initialization:
    Load data from input_file
    Extract spectral columns (2121-2339 cm⁻¹)
    Get all unique FARM_IDs (farms)
    Initialize device (CUDA if available, else CPU)
    
    Initialize result storage:
        results_by_val: Results organized by validation farm
        summary_records: Summary performance metrics
        detailed_records: Detailed metrics for each iteration
        distribution_records: Data distribution statistics
        loss_records: Training and validation loss records

    Main Loop - Leave-One-Study Validation:
        For each validation farm (val_farm):
          Print validation farm information
        
        Filter validation data:
            val_df = data where FARM_ID == val_farm
            val_mastitis = mastitis samples from val_df
            val_healthy = healthy samples from val_df
        
        If validation farm has insufficient samples (< 3 mastitis):
            Skip to next validation farm
        
        Create training pool:
            train_pool = all farms except val_farm
        
        For each training set size (from 1 to len(train_pool)):
            For each combination of farms in train_pool (size = training set size):
                combo_key = combination identifier
                
                Filter training data:
                    train_df = data where FARM_ID in combination
                    train_mastitis = mastitis samples from train_df
                    train_healthy = healthy samples from train_df
                
                If training data is insufficient:
                    Skip to next combination
                
                Farm-specific feature selection:
                    Analyze spectral feature variance from training data
                    Select features based on variance percentile (keep_percentile)
                    If no features selected:
                        Skip to next combination
                
                Generate sliding windows:
                    For start from (30 - time_window + 1) to 0, step = -step:
                        end = start + time_window - 1
                        If end <= 30:
                            Add window (start, end)
                
                Initialize combo_results
                
                For each window (window_start, window_end):
                    Filter training mastitis data within window
                    If sample count < 10:
                        Skip to next window
                    
                    Match healthy samples by DIM distribution:
                        Use consecutive DIM matching for time window
                        Match healthy cows with consecutive DIM values
                    
                    Analyze data distribution (first iteration only)
                    
                    Create enhanced cow sequences:
                        Process training mastitis cow time series
                        Process training healthy cow time series (DIM matching)
                        Create complete timeline sequences
                        Handle missing values
                        Create scaler from training data only
                    
                    Process validation data:
                        Filter validation mastitis data within window
                        Match validation healthy samples by DIM
                        Create validation cow sequences
                        Apply training scaler to validation data
                    
                    Merge training and validation sequences
                    Create EnhancedTimeSeriesDataset
                    Split dataset by cow ID (train vs validation)
                    
                    If insufficient samples:
                        Skip to next window
                    
                    Calculate class weights for weighted loss
                    Create DataLoader:
                        train_loader: batch_size=16, shuffle=True
                        val_loader: batch_size=16, shuffle=False
                    
                    Initialize EnhancedTwoStageModel:
                        spectral_encoder_type='transformer'
                        temporal_model_type='transformer'
                    
                    Train and evaluate model:
                        Train for num_epochs epochs
                        Use weighted loss for data imbalance
                        Apply gradient clipping (max_norm=1.0)
                        Evaluate on validation set
                        Calculate metrics: accuracy, precision, recall, AUC
                        Record training and validation losses
                    
                    Store window results:
                        - Performance metrics
                        - Training/validation losses
                        - Data distribution info
                
                Store combo results
                Store results in results_by_val[val_farm][combo_key]
        
        Store all results for current validation farm

    Save Results:
        Save to CSV files:
        - validation_summary_w{time_window}_s{step}.csv
        - validation_detailed_w{time_window}_s{step}.csv
        - validation_distributions_w{time_window}_s{step}.csv
        - validation_losses_w{time_window}_s{step}.csv

    Print Summary:
        Print validation summary:
        - For each validation farm:
            - Training farm combinations tested
            - Performance metrics for each combination
            - Best performing combination
