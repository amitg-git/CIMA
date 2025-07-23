import warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold
from scipy.stats import randint as sp_randint, uniform as sp_uniform, loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Added for evaluation
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex, rgb_to_hsv, hsv_to_rgb, Normalize
import seaborn as sns
import traceback
import os
import glob
import re
from typing import List, Union

from helper.file import FileLogger


class loguniform_int:
    """Integer valued version of the log-uniform distribution"""

    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)


class ConvergenceModel:
    def __init__(self, *,
                 feature_cols: Union[str, List[str]] = '',
                 valid_col='valid[]',
                 target_col='d[mm]',
                 noise_std: float = 0.0,
                 reduce_fps_flag: bool = False,
                 include_general_files=True,
                 include_divergence_files=False,
                 history_depth=5,
                 ext_feature_func=None,
                 random_state=42,
                 model_dir='convergence_model_final_v8_20250607_m56',
                 n_splits_cv=5,
                 n_iter_search=20,
                 n_jobs=-1,
                 only_print=True):
        """ Initializes the Convergence model """
        self.include_general_files = include_general_files
        self.include_divergence_files = include_divergence_files

        self.model_feature_cols = self._clean_col_names_list(list(feature_cols))
        self.valid_col = self._clean_col_name(valid_col)
        self.target_col = self._clean_col_name(target_col)
        self.history_depth = history_depth
        self.history_cols_added = False
        self.ext_feature_func = {} if ext_feature_func is None else ext_feature_func

        self.reduce_fps_flag = reduce_fps_flag
        self.noise_std = noise_std

        self.random_state = random_state
        self.model_dir = model_dir
        print(f"Model directory: {self.model_dir}")
        self.n_splits_cv = n_splits_cv
        self.n_iter_search = n_iter_search
        self.n_jobs = n_jobs
        self.model = None

        os.makedirs(self.model_dir, exist_ok=True)
        self.model_path = os.path.join(self.model_dir, 'model.joblib')
        self.features_path = os.path.join(self.model_dir, 'features.joblib')
        self.log_path = os.path.join(self.model_dir, 'trained_files.txt')
        self.logger = FileLogger(os.path.join(self.model_dir, 'train_log.txt'), only_print=only_print)

        self.dataset_dict = {
            'x_train': pd.DataFrame(), 'y_train': pd.Series(dtype=float),
            'x_val': pd.DataFrame(), 'y_val': pd.Series(dtype=float),
            'x_test': pd.DataFrame(), 'y_test': pd.Series(dtype=float),
        }

    @staticmethod
    def _clean_col_name(col_name):
        """ Remove the unit from one label. """
        return re.sub(r'\[.*?\]', '', str(col_name)).strip()

    def _clean_col_names_list(self, col_list):
        """ Remove the units from a label list. """
        return [self._clean_col_name(col) for col in col_list]

    def _clean_dataframe_cols(self, df):
        """ Remove the units from a dataframe. """
        df.columns = self._clean_col_names_list(df.columns)
        return df

    def _check_file_columns(self, file_path):
        """ Check for a valid file by checking its columns """
        try:
            header_df = pd.read_csv(file_path, nrows=0)
            header_df = self._clean_dataframe_cols(header_df)
            cols = set(header_df.columns)

            # common to CI files and general files
            required_cols = set(self.model_feature_cols + [self.valid_col])

            ci_file_types = ['convergence', 'divergence']
            is_ci_file = any(name in file_path for name in ci_file_types)
            if is_ci_file:
                required_cols.add(self.target_col)  # add d[mm] to the requirements

            missing_cols = required_cols - cols
            # if missing_cols: self.logger.log(f"missing cols in file {file_path}: {missing_cols}")
            return not missing_cols
        except Exception as e:
            self.logger.log(f"Error checking header for {file_path}: {e}")
            return False

    def find_categorized_files(self, data_folder_path):
        """ Finds valid files and categorizes based on name and column requirements. """
        convergence_files = []
        divergence_files = []
        general_files = []

        search_patterns = [os.path.join(data_folder_path, f) for f in ['*.txt', '*.csv']]

        all_files = []
        [all_files.extend(glob.glob(p)) for p in search_patterns]
        if not all_files:
            self.logger.log(f"Warn: No files in {data_folder_path}")
            return [], [], []

        self.logger.log(f"Found {len(all_files)} files. Filtering/Checking...")
        checked = 0

        for f_path in all_files:
            try:
                # Check columns *before* categorizing based on name
                if self._check_file_columns(f_path):
                    fname_lower = os.path.basename(f_path).lower()
                    if "convergence" in fname_lower:
                        convergence_files.append(f_path)
                    elif "divergence" in fname_lower:
                        if self.include_divergence_files:
                            divergence_files.append(f_path)
                    elif self.include_general_files:
                        # If it has all cols but isn't conv/div, treat as general
                        general_files.append(f_path)
            except Exception as e:
                self.logger.log(f"Warn: Could not check header for {f_path}: {e}")

            checked += 1
            if checked % 500 == 0:
                self.logger.log(f"Checked {checked}/{len(all_files)}...")

        self.logger.log(f"Found valid files ->\n"
                        f" - Conv: {len(convergence_files)}\n"
                        f" - Div: {len(divergence_files)}\n"
                        f" - Gen: {len(general_files)}")
        return convergence_files, divergence_files, general_files

    def _get_valid_df(self, df):
        valid_mask = (df[self.valid_col] == 1)
        filtered_df = df[valid_mask].copy()

        # Remove rows with any infinite values
        inf_mask = filtered_df.isin([float('inf'), float('-inf')]).any(axis=1)
        filtered_df = filtered_df[~inf_mask].copy()
        return filtered_df

    def _add_history_to_df(self, df: pd.DataFrame):
        """
        Adds lagged features for columns specified in self.model_feature_cols.
        Lagged columns have suffixes _n1, _n2, ..., inserted after each original column.
        All columns lose (history_depth-1) rows to maintain alignment.
        Returns:
            df_new: DataFrame with lagged columns for model features.
            new_cols: List of new lagged columns added.
        """
        if self.history_depth <= 1:
            return df.copy(), []

        cols = list(df.columns)
        result_cols, lagged_dfs, new_cols = [], [], []

        for col in cols:
            # Always keep original column
            result_cols.append(col)

            # Only add lagged features for model_feature_cols
            if col in self.model_feature_cols:
                for n in range(1, self.history_depth):
                    lagged_col = f"{col}_n{n}"
                    result_cols.append(lagged_col)
                    lagged_dfs.append(df[col].shift(n).rename(lagged_col))
                    new_cols.append(lagged_col)

        # Concatenate while preserving original column order + new lagged features
        df_new = pd.concat([df] + lagged_dfs, axis=1)[result_cols]
        df_new = df_new.iloc[self.history_depth - 1:].reset_index(drop=True)
        return df_new, new_cols

    @staticmethod
    def _reduce_df_fps(df, *, base=30, target=12):
        n_frames = len(df)
        target_frames = int(n_frames * target / base)
        indices = np.linspace(0, n_frames - 1, target_frames, dtype=int)
        df_reduced = df.iloc[indices].reset_index(drop=True)
        return df_reduced

    def create_dataset(self, convergence_files, divergence_files, general_files):
        """
        Create dataset for the model base on [convergence, divergence, general] file lists
        """
        self.logger.log("\nCreating dataset...")
        self.logger.log(f"--> history_depth = {self.history_depth}")

        full_dataset = pd.DataFrame()

        all_valid_files = {'general': [], 'convergence': [], 'divergence': []}

        file_lists_map = {'convergence': convergence_files, 'divergence': divergence_files}
        if self.include_general_files:
            file_lists_map['general'] = general_files
        n_file_lists = sum(len(lst) for lst in file_lists_map.values())

        self.logger.log(f"Processing {file_lists_map.keys()} with {n_file_lists} of files")

        # unique file index for future grouping
        unique_file_index = 0

        for source_type, file_list in file_lists_map.items():
            self.logger.log(f"Processing {len(file_list)} {source_type} files...")
            for i, fp in enumerate(file_list):
                try:
                    df = pd.read_csv(fp)
                    df = self._clean_dataframe_cols(df)

                    # Filter for valid rows and remove rows with infinity values
                    valid_df = self._get_valid_df(df)
                    if not valid_df.empty:
                        # Append used file (a valid and non-empty files)
                        all_valid_files[source_type].append(fp)

                        # preprocess target column for general and divergence source types
                        if source_type == 'general':
                            valid_df[self.target_col] = 0
                        elif source_type == 'divergence':
                            valid_df[self.target_col] *= -1

                        if self.reduce_fps_flag:
                            # Reduce FPS from 30 (recorded) to 10 (actual running)
                            base_fps, target_fps = 30, 10
                            self.logger.log(f"Reduce FPS from {base_fps} to {target_fps}")
                            valid_df = self._reduce_df_fps(valid_df, base=base_fps, target=target_fps)

                        # Add extended features to x
                        if self.ext_feature_func is not None:
                            for name, func in self.ext_feature_func.items():
                                new_cols = func(valid_df)
                                if len(new_cols) > 0 and new_cols[0] not in self.model_feature_cols:
                                    self.model_feature_cols += new_cols

                        # Add history to x
                        history_df, history_cols = self._add_history_to_df(valid_df)
                        if not self.history_cols_added:
                            self.model_feature_cols += history_cols
                            self.history_cols_added = True

                        # Select only necessary columns
                        required_cols = set(self.model_feature_cols + [self.valid_col, self.target_col])
                        history_df = history_df[list(required_cols)]

                        # add unique file index and source type for later grouping
                        history_df.insert(0, 'file_index', unique_file_index)
                        history_df.insert(1, 'source', source_type)
                        unique_file_index += 1

                        full_dataset = pd.concat([full_dataset, history_df], ignore_index=True, sort=False)
                    else:
                        self.logger.log(f'  Warning: {os.path.basename(fp)} is empty')

                except pd.errors.EmptyDataError:
                    self.logger.log(f"  Warn: Skipping empty file {os.path.basename(fp)}")
                except Exception as e:
                    self.logger.log(f"  Err loading {os.path.basename(fp)}: {e}")

        if full_dataset.empty:
            raise ValueError("No data loaded. Dataset is empty!")

        # log the used files in the full dataset
        self._log_used_files(all_valid_files['convergence'],
                             all_valid_files['divergence'],
                             all_valid_files['general'])
        self.logger.log("\nDone creating dataset...")

        return full_dataset

    def _log_used_files(self, convergence_files, divergence_files, general_files):
        """ Log used files into a text file """
        try:  # Log files
            with open(self.log_path, 'w') as f:
                f.write("#Conv:\n")
                [f.write(f"{fp}\n") for fp in convergence_files]
                f.write("\n#Div:\n")
                [f.write(f"{fp}\n") for fp in divergence_files]
                f.write("\n#Gen:\n")
                [f.write(f"{fp}\n") for fp in general_files]
            self.logger.log(f"Files logged: {self.log_path}")
        except Exception as e:
            self.logger.log(f"Warn: Could not write log: {e}")

    def _save_dataset(self):
        self.logger.log("\nSaving Datasets...")

        # Save Train, Val and Test sets into new CSV files
        for key, data in self.dataset_dict.items():
            self.logger.log(f"saving {key} ...")
            filepath = os.path.join(self.model_dir, f"{key}.csv")
            data.to_csv(filepath, index=False, header=True if 'x_' in key else False)

        self.logger.log("\nDone saving Datasets...")

    @staticmethod
    def _create_noise(std: float, size: int):
        noise = np.random.normal(loc=0, scale=std, size=size)
        return noise

    def create_dataset_and_split(self, data_folder, *, val_size=0.15, test_size=0.15):
        """ Loads data, generates labels, splits"""
        convergence_files, divergence_files, general_files = self.find_categorized_files(data_folder)
        if not convergence_files and not divergence_files and not general_files:
            raise ValueError("No valid files found.")

        # Create dataset and save it
        full_dataset = self.create_dataset(convergence_files, divergence_files, general_files)

        # Add noise to the target coloumn:
        if self.noise_std > 0:
            self.logger.log(f"Adding noise with std: {self.noise_std}")
            full_dataset[self.target_col] += self._create_noise(std=self.noise_std, size=len(full_dataset))

        # Split dataset depend on groups (unique files)
        self.split_dataset(full_dataset, val_size=val_size, test_size=test_size)

        # Save the Datasets
        # self._save_dataset()

        self.logger.log(f"\n--- Training using {full_dataset['file_index'].nunique()} files) ---")

    def split_dataset(self, df, *, val_size=0.15, test_size=0.15):
        self.logger.log("--- Split Percentages ---")
        self.logger.log(f"Validation: {val_size:.0%}")
        self.logger.log(f"Test: {test_size:.0%}")

        # Get unique file_index with their corresponding 'source'
        file_index_df = df[['file_index', 'source']].drop_duplicates()

        # First, split off the test set
        if test_size > 0:
            file_indices, test_indices = train_test_split(
                file_index_df,
                test_size=test_size,
                stratify=file_index_df['source'],
                random_state=self.random_state
            )
        else:
            file_indices = file_index_df

        # Next, split the remaining into train and val
        val_relative_size = val_size / (1 - test_size)  # Adjust val_size relative to remaining data
        train_indices, val_indices = train_test_split(
            file_indices,
            test_size=val_relative_size,
            stratify=file_indices['source'],
            random_state=self.random_state
        )

        # # First split: train and temp (val+test)
        # train, temp_df = train_test_split(
        #     df,
        #     test_size=(test_size + val_size),
        #     random_state=self.random_state
        # )
        #
        # # Second split: val and test
        # if test_size > 0:
        #     val, test = train_test_split(
        #         temp_df,
        #         test_size=(test_size / (val_size + test_size)),
        #         random_state=self.random_state
        #     )
        # else:
        #     val = temp_df

        # Helper to get all rows for a set of file_indices
        def get_rows(indices_df):
            return df[df['file_index'].isin(indices_df['file_index'])]

        train = get_rows(train_indices)
        x_train = train[self.model_feature_cols + ['file_index']]
        y_train = train[self.target_col]

        val = get_rows(val_indices)
        x_val = val[self.model_feature_cols + ['file_index']]
        y_val = val[self.target_col]

        if test_size > 0:
            test = get_rows(test_indices)
            x_test = test[self.model_feature_cols + ['file_index']]
            y_test = test[self.target_col]
        else:
            x_test = pd.DataFrame()
            y_test = pd.Series()

        self.logger.log(f"\nDatasets Loading Summary:")

        self.logger.log(f"Train files:\n{train.groupby('source')['file_index'].nunique()}")
        self.logger.log(f"Train shapes - x: {x_train.shape}, y: {y_train.shape}")
        # self.logger.log(f"Train {self.target_col} histogram:\n{y_train.value_counts()}")

        self.logger.log(f"Val files:\n{val.groupby('source')['file_index'].nunique()}")
        self.logger.log(f"Val shapes - x: {x_val.shape}, y: {y_val.shape}")
        # self.logger.log(f"Val {self.target_col} histogram:\n{y_val.value_counts()}")

        if test_size > 0:
            self.logger.log(f"Test files:\n{test.groupby('source')['file_index'].nunique()}")
            self.logger.log(f"Test shapes - x: {x_test.shape}, y: {y_test.shape}")
        # self.logger.log(f"Test {self.target_col} histogram:\n{y_test.value_counts()}")

        # Store in self.dataset_dict
        self.dataset_dict = {
            'x_train': x_train.reset_index(drop=True),
            'y_train': y_train.reset_index(drop=True),
            'x_val': x_val.reset_index(drop=True),
            'y_val': y_val.reset_index(drop=True),
            'x_test': x_test.reset_index(drop=True),
            'y_test': y_test.reset_index(drop=True),
        }
        self.logger.log("\nDone splitting dataset...")

    def load_ex_dataset(self, dir_path=None):
        try:
            if dir_path is None:
                dir_path = self.model_dir

            self.dataset_dict['x_train'] = pd.read_csv(os.path.join(dir_path, "x_train.csv"))
            self.dataset_dict['y_train'] = pd.read_csv(os.path.join(dir_path, "y_train.csv"), header=None, squeeze=True)
            self.dataset_dict['x_val'] = pd.read_csv(os.path.join(dir_path, "x_val.csv"))
            self.dataset_dict['y_val'] = pd.read_csv(os.path.join(dir_path, "y_val.csv"), header=None, squeeze=True)
            self.dataset_dict['x_test'] = pd.read_csv(os.path.join(dir_path, "x_test.csv"))
            self.dataset_dict['y_test'] = pd.read_csv(os.path.join(dir_path, "y_test.csv"), header=None, squeeze=True)

            self.model_feature_cols = self.dataset_dict['x_train'].columns
        except Exception as e:
            self.logger.log(f"Error loading build-int dataset: {e}")

    def train(self, data_folder, *,
              val_size=0.15, test_size=0.15, load_ex_dataset_dir=None, use_hptuning=False):
        """ trains, evaluates, and saves. """
        if load_ex_dataset_dir:
            self.load_ex_dataset(load_ex_dataset_dir)

        is_any_empty = any(value.empty for value in self.dataset_dict.values())
        if is_any_empty:
            self.logger.log("** loading new dataset and split it **")
            self.create_dataset_and_split(data_folder, val_size=val_size, test_size=test_size)
        else:
            self.logger.log("** Using external dataset **")

        self._train_model(use_hptuning=use_hptuning)
        self.evaluate_model()
        self.save()

    def _train_model(self, use_hptuning=False):
        x_train = self.dataset_dict['x_train']
        y_train = self.dataset_dict['y_train']
        x_val = self.dataset_dict['x_val'].drop(columns=['file_index'])
        y_val = self.dataset_dict['y_val']

        # Create groups for CV and remove 'file_index from features
        groups = x_train['file_index']
        x_train = x_train.drop(columns=['file_index'])

        if x_train.empty or len(y_train.values) != len(x_train.values):
            raise ValueError("Training data/labels mismatch.")

        self.logger.log(f"\n--- Training Model (HPTuning={use_hptuning}) ---")

        # Extra weight for d = 0
        sample_weight = None
        sample_weight = np.where((np.abs(y_train) < 5), 3.0, 1.0)
        self.logger.log(f"Use sample_weight: {(sample_weight is not None)}")

        if use_hptuning:
            self.logger.log(f"\nRunning Randomized Search CV ({self.n_iter_search} iterations)...")
            param_dist = {
                # 'num_leaves': sp_randint(30, 120),        # Control tree complexity
                'max_depth': sp_randint(6, 20),  # Explicitly limit tree depth
                'learning_rate': loguniform(0.002, 0.02),  # Smaller learning rates
                'min_child_samples': sp_randint(50, 300),  # Min samples in a leaf
                'reg_alpha': loguniform(0.1, 1000.0),  # L1 regularization
                'reg_lambda': loguniform(0.01, 10.0),  # L2 regularization
                'max_bin': sp_randint(32, 64),
                'min_split_gain': sp_uniform(loc=0, scale=0.3),  # Min gain to make a split
                'colsample_bytree': sp_uniform(loc=0.5, scale=0.5),  # Feature fraction (0.5 to 1.0)
                'subsample': sp_uniform(loc=0.5, scale=0.5),  # Bagging fraction (0.5 to 1.0)
                'subsample_freq': sp_randint(1, 5),  # Bagging frequency
                # if subsample < 1.0
            }

            model = lgb.LGBMRegressor(
                n_estimators=1000,
                random_state=self.random_state, n_jobs=self.n_jobs, verbosity=-1
            )

            # Define cross-validator
            cv = GroupKFold(n_splits=self.n_splits_cv)

            scoring = 'neg_mean_absolute_error'  # Use MAE as it aligns with eval_metric='l1' later
            search_clf = RandomizedSearchCV(estimator=model, param_distributions=param_dist,
                                            n_iter=self.n_iter_search,
                                            scoring=scoring,
                                            cv=cv, random_state=self.random_state,
                                            n_jobs=self.n_jobs, verbose=5, return_train_score=True)

            search_clf.fit(x_train, y_train, sample_weight=sample_weight, groups=groups)
            self._log_fit_process(search_clf)
            best_params = search_clf.best_params_
            self.model = lgb.LGBMRegressor(
                **best_params,
                random_state=self.random_state, n_jobs=self.n_jobs
            )
        else:
            self.logger.log("\nTraining model with default parameters...")
            self.model = lgb.LGBMRegressor(
                colsample_bytree=0.861027799735174,
                learning_rate=0.014703215466415787,
                max_bin=63,
                max_depth=15,
                min_child_samples=188,
                min_split_gain=0.0692227006220387,
                reg_alpha=114.72202280931874,
                reg_lambda=6.797834050129515,
                subsample=0.8749996243850502,
                subsample_freq=3,
                n_estimators=5000,
                random_state=self.random_state, n_jobs=self.n_jobs)

        # --- Fit the selected model (either best from HPT or default) ---
        def learning_rate_decay(current_iter):
            initial_lr = 0.014703215466415787
            decay_rate = 0.8
            th, step = 50, 5
            if (current_iter >= th) and (current_iter % step == 0):
                lr = initial_lr * (decay_rate ** ((current_iter - th) / step + 1))
            else:
                lr = initial_lr
            return lr

        evals_result = {}
        callbacks = [
            lgb.reset_parameter(learning_rate=learning_rate_decay),
            lgb.record_evaluation(evals_result),
            # Add early stopping based on validation set performance
            lgb.early_stopping(stopping_rounds=50,  # Number of rounds with no improvement to stop
                               verbose=True,  # Print messages when stopping happens
                               min_delta=0.01)  # Minimum change considered an improvement
        ]
        self.model.fit(
            x_train, y_train,
            sample_weight=sample_weight,
            eval_set=[(x_train, y_train), (x_val, y_val)],
            eval_metric='l1',  # MAE
            eval_names=['train', 'val'],
            callbacks=callbacks
        )
        self.logger.log("Training complete.")
        if self.model.best_iteration_:
            self.logger.log(f"Best iteration found by early stopping: {self.model.best_iteration_}")

        depths = self._get_model_depths()
        max_depth = max(depths)
        self.logger.log(f"Maximum depth of any tree: {max_depth}")
        self.logger.log(f"Average tree depth: {sum(depths) / len(depths):.2f}")

        self.logger.log('-' * 50)
        self.logger.log("**** Model Parameters: ****")
        self.logger.log(self.model.get_params())
        self.logger.log('-' * 50)

        # Plot MAE curves
        plt.figure(figsize=(8, 5))
        plt.plot(evals_result['train']['l1'], label='Train MAE')
        plt.plot(evals_result['val']['l1'], label='Val MAE')

        # Add vertical line for best iteration if available
        if hasattr(self.model, 'best_iteration_') and self.model.best_iteration_:
            plt.axvline(x=self.model.best_iteration_, color='r', linestyle='--',
                        label=f'Best Iteration ({self.model.best_iteration_})')

        plt.title('Training and Validation MAE per Iteration')
        plt.xlabel('Boosting Iteration')
        plt.ylabel('Mean Absolute Error (l1)')
        plt.legend(loc='best')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, "training_mae_curve.png"))
        plt.show()

    def _get_model_depths(self):
        # After fitting your model:
        booster = self.model.booster_  # or self.model if using native API

        # Get the model dump as a list of trees
        model_dump = booster.dump_model()
        tree_infos = model_dump['tree_info']

        # Compute depths for all trees
        def get_tree_depth(tree):
            def _depth(node):
                if 'left_child' not in node and 'right_child' not in node:
                    return 1
                left = _depth(node['left_child']) if 'left_child' in node else 0
                right = _depth(node['right_child']) if 'right_child' in node else 0
                return 1 + max(left, right)

            return _depth(tree['tree_structure'])

        depths = [get_tree_depth(tree) for tree in tree_infos]
        return depths

    def _log_fit_process(self, search_clf):
        """
        Logs the detailed results and summary of the RandomizedSearchCV process.
        Adapted from the provided reference code.
        """
        self.logger.log("\n" + "=" * 10 + " RandomizedSearchCV Detailed Log " + "=" * 10)

        try:
            cv_results = search_clf.cv_results_
            n_splits = search_clf.n_splits_  # Get actual number of splits used
            n_candidates = len(cv_results['params'])
            scoring_metric = search_clf.scoring if isinstance(search_clf.scoring, str) else 'score'

            # --- Header ---
            self.logger.log(
                f"Fitting {n_splits} folds for each of {n_candidates} candidates, "
                f"totalling {n_splits * n_candidates} fits."
            )
            self.logger.log(f"Scoring metric: {scoring_metric}")

            # --- Per-Fold Logging (if results available) ---
            # Check if detailed split scores are present
            fold_keys_present = all(f'split{k}_test_score' in cv_results for k in range(n_splits))

            if fold_keys_present:
                self.logger.log("\n--- Per-Fold Results ---")
                for candidate_idx in range(n_candidates):
                    params = cv_results['params'][candidate_idx]
                    self.logger.log(f"\nCandidate {candidate_idx + 1} Params: {params}")

                    # Format parameters nicely - include only those actually in params dict
                    # Adapt this list based on your actual param_dist keys
                    param_list_str = []
                    param_keys_in_dist = [
                        'num_leaves', 'max_depth', 'learning_rate', 'n_estimators',
                        'min_child_samples', 'reg_alpha', 'reg_lambda', 'subsample',
                        'colsample_bytree'
                    ]  # Add/remove keys based on your param_dist
                    for p_key in param_keys_in_dist:
                        if p_key in params:
                            p_val = params[p_key]
                            # Format floats precisely, keep ints as ints
                            if isinstance(p_val, float):
                                param_list_str.append(f"{p_key}={p_val:.6f}")  # Adjust precision if needed
                            else:
                                param_list_str.append(f"{p_key}={p_val}")

                    for fold in range(n_splits):
                        # Handle potential absence of train scores
                        train_score = cv_results.get(f'split{fold}_train_score', [np.nan] * n_candidates)[candidate_idx]
                        test_score = cv_results[f'split{fold}_test_score'][candidate_idx]

                        # Handle negative scoring metrics for display
                        display_train_score = -train_score if scoring_metric.startswith('neg_') else train_score
                        display_test_score = -test_score if scoring_metric.startswith('neg_') else test_score
                        score_name = scoring_metric[4:] if scoring_metric.startswith('neg_') else scoring_metric

                        self.logger.log(
                            f"  [CV Fold {fold + 1}/{n_splits}] {score_name}: "
                            f"(train={display_train_score:.5f}, test={display_test_score:.5f})"
                        )
            else:
                self.logger.log("\nNote: Detailed per-fold scores not found in cv_results_. Skipping per-fold log.")
                self.logger.log("Consider increasing verbose level during RandomizedSearchCV fit if needed.")

            # --- Summary Section ---
            self.logger.log("\n" + "=" * 10 + " SUMMARY OF CANDIDATES " + "=" * 10)
            summary_data = {
                'rank': cv_results['rank_test_score'],
                'mean_test_score': cv_results['mean_test_score'],
                'std_test_score': cv_results['std_test_score'],
                'mean_fit_time': cv_results['mean_fit_time']
            }
            # Add train scores if available
            if 'mean_train_score' in cv_results:
                summary_data['mean_train_score'] = cv_results['mean_train_score']
                summary_data['std_train_score'] = cv_results['std_train_score']

            summary_df = pd.DataFrame(summary_data)
            summary_df['params'] = cv_results['params']  # Add params as the last column for readability

            # Handle negative scoring metrics for display in summary
            display_score_name = scoring_metric[4:] if scoring_metric.startswith('neg_') else scoring_metric
            if scoring_metric.startswith('neg_'):
                summary_df[f'mean_test_{display_score_name}'] = -summary_df['mean_test_score']
                if 'mean_train_score' in summary_df.columns:
                    summary_df[f'mean_train_{display_score_name}'] = -summary_df['mean_train_score']
                # Select columns for display, showing the positive scores
                display_cols = ['rank', f'mean_test_{display_score_name}', 'std_test_score']
                if f'mean_train_{display_score_name}' in summary_df.columns:
                    display_cols.extend([f'mean_train_{display_score_name}', 'std_train_score'])
                display_cols.extend(['mean_fit_time', 'params'])
                summary_df_display = summary_df[display_cols].sort_values(by='rank')
            else:
                # Select columns for display (already positive scores)
                display_cols = ['rank', 'mean_test_score', 'std_test_score']
                if 'mean_train_score' in summary_df.columns:
                    display_cols.extend(['mean_train_score', 'std_train_score'])
                display_cols.extend(['mean_fit_time', 'params'])
                summary_df_display = summary_df[display_cols].sort_values(by='rank')

            self.logger.log(f"Scores based on '{display_score_name}', lower rank is better.")
            # Use to_string for potentially wide parameter display
            self.logger.log(summary_df_display.to_string(index=False))

            # --- Best Model Section ---
            self.logger.log("\n" + "=" * 10 + " BEST MODEL FOUND " + "=" * 10)
            best_score = search_clf.best_score_
            display_best_score = -best_score if scoring_metric.startswith('neg_') else best_score

            self.logger.log(f"Best Mean Cross-Validation Score ({display_score_name}): {display_best_score:.5f}")
            # Add interpretation if using common negative metrics
            if scoring_metric == 'neg_mean_squared_error':
                self.logger.log(f"  (Equivalent Mean Validation MSE: {-best_score:.5f})")
                self.logger.log(f"  (Equivalent Mean Validation RMSE: {np.sqrt(-best_score):.5f})")
            elif scoring_metric == 'neg_mean_absolute_error':
                self.logger.log(f"  (Equivalent Mean Validation MAE: {-best_score:.5f})")

            self.logger.log(f"Best Parameters Found:")
            # Log best parameters dictionary formatted nicely
            for key, value in search_clf.best_params_.items():
                self.logger.log(f"  - {key}: {value}")

            # --- Full Results (Condensed) ---
            # This section is somewhat redundant with the summary table but follows the reference format
            self.logger.log("\n" + "=" * 10 + " ALL CANDIDATE RESULTS (Mean +/- Std Dev) " + "=" * 10)
            for i in range(n_candidates):
                params = cv_results['params'][i]
                mean_test = cv_results['mean_test_score'][i]
                std_test = cv_results['std_test_score'][i]

                display_mean_test = -mean_test if scoring_metric.startswith('neg_') else mean_test

                log_line = f"Config {i + 1}: Test {display_score_name} = {display_mean_test:.5f} (±{std_test:.5f})"

                # Include train scores if available
                if 'mean_train_score' in cv_results:
                    mean_train = cv_results['mean_train_score'][i]
                    std_train = cv_results['std_train_score'][i]
                    display_mean_train = -mean_train if scoring_metric.startswith('neg_') else mean_train
                    log_line += f" | Train {display_score_name} = {display_mean_train:.5f} (±{std_train:.5f})"

                log_line += f" | Params: {params}"
                self.logger.log(log_line)

            self.logger.log("\n" + "=" * 10 + " End HPTuning Log " + "=" * 10)

        except KeyError as e:
            self.logger.log(f"ERROR logging HPT results: Missing expected key in cv_results_ - {e}.")
            self.logger.log("CV results dictionary keys:")
            self.logger.log(str(search_clf.cv_results_.keys()))
        except Exception as e:
            self.logger.log(f"ERROR occurred during HPT logging: {e}")
            import traceback
            self.logger.log(traceback.format_exc())

    def _plot_predicted_vs_actual(self, x_input, y_true, y_pred, set_name, split_by_focal=False):
        """
        Plots predicted vs actual values.
        Each unique (focal_x, focal_y) gets a unique base color.
        Brightness of each point is modulated by y_true (darker for lower values, brighter for higher).
        If split_by_focal is True, creates a grid of subplots, one for each unique focal point,
        and colors each point in each subplot according to its y_true value.
        """
        self.logger.log(f"Generating Predicted vs Actual plot for {set_name} set...")

        try:
            focal_pairs = list(zip(x_input['focal_x'], x_input['focal_y']))
            unique_focals = sorted(list(set(focal_pairs)))
            palette = sns.color_palette("hsv", len(unique_focals))
            focal_to_color = {fp: palette[i] for i, fp in enumerate(unique_focals)}

            # Normalize y_true for brightness (0 = darkest, 1 = brightest)
            y_norm = (y_true - np.min(y_true)) / (np.max(y_true) - np.min(y_true) + 1e-8)

            if not split_by_focal:
                # Assign colors with brightness modulation
                point_colors = []
                for i, fp in enumerate(focal_pairs):
                    base_rgb = np.array(focal_to_color[fp])
                    base_hsv = rgb_to_hsv(base_rgb.reshape(1, -1))[0]
                    base_hsv[2] = 0.3 + 0.7 * y_norm[i]  # 0.3=min brightness, 1.0=max
                    bright_rgb = hsv_to_rgb(base_hsv.reshape(1, -1))[0]
                    point_colors.append(to_hex(bright_rgb))

                plt.figure(figsize=(8, 8))
                plt.scatter(y_true, y_pred, c=point_colors, alpha=0.8, s=20, label=f'{set_name} Data', edgecolor='k',
                            linewidth=0.2)

                # Add y=x line
                min_val = min(y_true.min(), y_pred.min())
                max_val = max(y_true.max(), y_pred.max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.75, label='y=x (Perfect Prediction)')

                plt.xlim(min_val, max_val)
                plt.ylim(min_val, max_val)
                plt.title(f'Predicted vs Actual ({self.target_col}) on {set_name} Set')
                plt.xlabel(f'Actual {self.target_col}')
                plt.ylabel(f'Predicted {self.target_col}')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()
                plot_path = os.path.join(self.model_dir, f"predicted_vs_actual_{set_name.lower()}_set.png")
                plt.savefig(plot_path)
                self.logger.log(f"  Plot saved to: {os.path.basename(plot_path)}")
                plt.close()

            else:
                n_focals = len(unique_focals)
                n_cols = min(4, n_focals)
                n_rows = int(np.ceil(n_focals / n_cols))
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
                fig.suptitle(f'Predicted vs Actual ({self.target_col}) by Focal Point on {set_name} Set', fontsize=18)

                # Set up colormap for y_true
                cmap = plt.get_cmap("viridis")
                norm = Normalize(vmin=y_true.min(), vmax=y_true.max())

                for idx, fp in enumerate(unique_focals):
                    row, col = divmod(idx, n_cols)
                    ax = axes[row][col]
                    idxs = [i for i, f in enumerate(focal_pairs) if f == fp]
                    if not idxs:
                        ax.axis('off')
                        continue

                    # Color by y_true value
                    y_true_vals = y_true[idxs]
                    y_pred_vals = y_pred[idxs]
                    colors = cmap(norm(y_true_vals))

                    sc = ax.scatter(y_true_vals, y_pred_vals, c=colors, alpha=0.05, s=40, edgecolor='k', linewidth=0.3)

                    min_val = min(y_true_vals.min(), y_pred_vals.min(), -10)
                    max_val = max(y_true_vals.max(), y_pred_vals.max(), 90)
                    ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.75, label='y=x (Perfect Prediction)')

                    ax.set_xlim(min_val, max_val)
                    ax.set_ylim(min_val, max_val)
                    ax.set_title(f'Focal_x={fp[0]}, Focal_y={fp[1]}')
                    ax.set_xlabel(f'Actual {self.target_col}')
                    ax.set_ylabel(f'Predicted {self.target_col}')
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.6)

                # Hide any unused subplots
                for idx in range(n_focals, n_rows * n_cols):
                    row, col = divmod(idx, n_cols)
                    axes[row][col].axis('off')

                # Add a colorbar for y_true (only once for all subplots)
                fig.subplots_adjust(right=0.92)
                cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                fig.colorbar(sm, cax=cbar_ax, label='y_true value')

                plt.tight_layout(rect=[0, 0.03, 0.93, 0.95])
                plot_path = os.path.join(self.model_dir, f"predicted_vs_actual_{set_name.lower()}_by_focal.png")
                plt.savefig(plot_path)
                self.logger.log(f"  Subplots saved to: {os.path.basename(plot_path)}")
                plt.close()
        except Exception as e:
            self.logger.log(f"  Warning: Failed to generate Predicted vs Actual plot: {e}")
            plt.close()  # Ensure plot is closed even on error

    def _plot_residuals_histogram(self, residuals, set_name):
        """Plots the distribution of residuals."""
        self.logger.log(f"Generating Residuals Histogram for {set_name} set...")
        try:
            plt.figure(figsize=(10, 6))
            sns.histplot(residuals, kde=True, bins=50)  # Use more bins for potentially large data
            plt.title(f'Distribution of Residuals (Actual - Predicted) on {set_name} Set')
            plt.xlabel(f'Residual Value ({self.target_col})')
            plt.ylabel('Frequency')
            plt.axvline(0, color='red', linestyle='--', label='Zero Residual')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plot_path = os.path.join(self.model_dir, f"residuals_distribution_{set_name.lower()}_set.png")
            plt.savefig(plot_path)
            self.logger.log(f"  Plot saved to: {os.path.basename(plot_path)}")
            plt.close()
        except Exception as e:
            self.logger.log(f"  Warning: Failed to generate Residuals Histogram plot: {e}")
            plt.close()

    def _plot_residuals_vs_feature(self, residuals, feature_values, feature_name, set_name):
        """Plots residuals against a specific feature."""
        self.logger.log(f"Generating Residuals vs {feature_name} plot for {set_name} set...")
        try:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=feature_values, y=residuals, alpha=0.4, s=15, label=f'Residuals')
            plt.axhline(0, color='red', linestyle='--', label='Zero Residual')
            plt.title(f'Residuals vs {feature_name} on {set_name} Set')
            plt.xlabel(f'{feature_name}')
            plt.ylabel(f'Residual Value ({self.target_col})')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            # Sanitize feature name for filename
            safe_feature_name = feature_name.replace('[', '').replace(']', '').replace('/', '_').replace('\\', '_')
            plot_path = os.path.join(self.model_dir, f"residuals_vs_{safe_feature_name}_{set_name.lower()}_set.png")
            plt.savefig(plot_path)
            self.logger.log(f"  Plot saved to: {os.path.basename(plot_path)}")
            plt.close()
        except Exception as e:
            self.logger.log(f"  Warning: Failed to generate Residuals vs {feature_name} plot: {e}")
            plt.close()

    def _plot_mae_heatmap_scr_dist_vs_d_actual(self, y_true, y_pred, scr_dist_values, set_name):
        """
        Plots a heatmap of the Mean Absolute Error (MAE) based on binned screen distance
        (Y-axis) and actual target value d[mm] (X-axis).

        Args:
            y_true (pd.Series): The actual target values (d[mm]).
            y_pred (np.ndarray): The predicted target values.
            scr_dist_values (pd.Series): The screen distance values corresponding to y_true/y_pred.
            set_name (str): Name of the data set (e.g., "Test", "Validation").
        """
        self.logger.log(f"Generating MAE Heatmap (Screen Distance vs Actual {self.target_col}) for {set_name} set...")

        # Define bins
        # Screen Distance Bins (Y-axis)
        scr_dist_bins = [-np.inf, 50, 55, 60, 65, np.inf]
        scr_dist_labels = ['<50', '50-55', '55-60', '60-65', '>65']

        # Actual d[mm] Bins (X-axis)
        d_actual_bins = np.arange(-100, 101, 20)  # Edges: -100, -80, ..., 80, 100
        # Create labels like '(-100, -80]', '(-80, -60]', etc.
        d_actual_labels = [f'({d_actual_bins[i]}, {d_actual_bins[i + 1]}]' for i in range(len(d_actual_bins) - 1)]

        try:
            abs_error = np.abs(y_true - y_pred)

            # Create temporary DataFrame
            df = pd.DataFrame({
                'scr_dist': scr_dist_values,
                'd_actual': y_true,
                'abs_error': abs_error
            })

            # Apply binning using pd.cut
            # Use right=True (default) includes the right edge: (left, right]
            df['scr_dist_bin'] = pd.cut(df['scr_dist'], bins=scr_dist_bins, labels=scr_dist_labels, right=True,
                                        ordered=True)
            df['d_actual_bin'] = pd.cut(df['d_actual'], bins=d_actual_bins, labels=d_actual_labels, right=True,
                                        ordered=True)

            # Handle cases where data might fall outside explicit d_actual bins if any
            df.dropna(subset=['scr_dist_bin', 'd_actual_bin'], inplace=True)

            if df.empty:
                self.logger.log("  Warning: No data points fall within the specified bins. Skipping heatmap.")
                return

            # Create pivot table for mean absolute error
            pivot_table = pd.pivot_table(df, values='abs_error',
                                         index='scr_dist_bin',  # Y-axis
                                         columns='d_actual_bin',  # X-axis
                                         aggfunc=np.mean,  # Aggregate using mean
                                         observed=False)  # Show all defined bins even if empty

            if pivot_table.empty:
                self.logger.log("  Warning: Pivot table for heatmap is empty after aggregation. Skipping plot.")
                return

            plt.figure(figsize=(12, 7))  # Adjust size as needed
            sns.heatmap(pivot_table,
                        annot=True,  # Show MAE values in cells
                        fmt=".3f",  # Format values to 3 decimal places
                        cmap="viridis",  # Colormap (viridis, magma, plasma, coolwarm)
                        linewidths=.5,  # Add lines between cells
                        cbar_kws={'label': f'Mean Absolute Error ({self.target_col})'})  # Color bar label

            plt.xlabel(f'Actual {self.target_col} Range (binned)')
            plt.ylabel('Screen Distance [cm] Range (binned)')
            plt.title(f'Mean Absolute Error Heatmap ({set_name} Set)')
            plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels if they overlap
            plt.yticks(rotation=0)
            plt.tight_layout()

            plot_path = os.path.join(self.model_dir, f"mae_heatmap_scr_dist_vs_d_actual_{set_name.lower()}_set.png")
            plt.savefig(plot_path)
            self.logger.log(f"  Plot saved to: {os.path.basename(plot_path)}")
            plt.close()

        except Exception as e:
            self.logger.log(
                f"  Warning: Failed to generate MAE Heatmap (Screen Distance vs Actual {self.target_col}): {e}")
            self.logger.log(traceback.format_exc())  # Log detailed traceback for debugging
            plt.close()  # Ensure plot is closed even on error

    def _plot_feature_importance(self, max_features=20):
        """Plots the feature importance from the trained LightGBM model."""
        if not hasattr(self.model, 'feature_importances_'):
            self.logger.log("  Warning: Model does not have 'feature_importances_'. Skipping plot.")
            return

        self.logger.log(f"Generating Feature Importance plot (Top {max_features})...")
        try:
            importances = self.model.feature_importances_
            # Use stored cleaned feature names
            feature_names = self.model_feature_cols

            if len(importances) != len(feature_names):
                self.logger.log(
                    f"  Warning: Mismatch between importance count ({len(importances)}) and feature name count ({len(feature_names)}). Skipping plot.")
                return

            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

            # Select top N features
            top_features_df = feature_importance_df.head(max_features)

            plt.figure(figsize=(10, max(6, int(0.3 * len(top_features_df)))))  # Adjust height based on num features
            sns.barplot(x='importance', y='feature', data=top_features_df, palette='viridis')
            plt.title(f'Top {len(top_features_df)} Feature Importances (LightGBM)')
            plt.xlabel('Importance Score')
            plt.ylabel('Feature')
            plt.tight_layout()
            plot_path = os.path.join(self.model_dir, "feature_importance.png")
            plt.savefig(plot_path)
            self.logger.log(f"  Plot saved to: {os.path.basename(plot_path)}")
            plt.close()
        except Exception as e:
            self.logger.log(f"  Warning: Failed to generate Feature Importance plot: {e}")
            plt.close()

    def evaluate_model(self):
        """
        Evaluates the trained model on the test set (or validation set if test is empty)
        and generates various performance visualization plots.
        """
        self.logger.log("\n" + "=" * 10 + " Evaluating Model Performance " + "=" * 10)

        if self.model is None:
            self.logger.log("Model is not trained or loaded. Cannot evaluate.")
            return

        eval_set_name = "Test"
        x_eval = self.dataset_dict.get('x_test')
        y_eval = self.dataset_dict.get('y_test')

        # Ensure data exists and is not empty
        if x_eval is None or y_eval is None or x_eval.empty or y_eval.empty:
            self.logger.log("Test set data is missing or empty.")
            # Fallback to validation set
            x_eval = self.dataset_dict.get('x_val')
            y_eval = self.dataset_dict.get('y_val')
            if x_eval is None or y_eval is None or x_eval.empty or y_eval.empty:
                self.logger.log("Validation set data also missing or empty. Skipping evaluation.")
                return
            else:
                self.logger.log("Evaluating on Validation Set instead.")
                eval_set_name = "Validation"

        # Check for length mismatch
        if len(x_eval) != len(y_eval):
            self.logger.log(
                f"ERROR: {eval_set_name} data/label length mismatch: X={len(x_eval)}, y={len(y_eval)}. Skipping evaluation.")
            return

        x_eval = x_eval.drop(columns=['file_index'])
        self.logger.log(f"Evaluating on {eval_set_name} Set ({len(x_eval)} samples)...")

        try:
            # --- Prediction ---
            y_pred = self.model.predict(x_eval)
            residuals = y_eval - y_pred

            # --- Calculate Metrics ---
            mse = mean_squared_error(y_eval, y_pred)
            mae = mean_absolute_error(y_eval, y_pred)
            r2 = r2_score(y_eval, y_pred)
            rmse = np.sqrt(mse)

            self.logger.log(f"\n{eval_set_name} Set Performance Metrics:")
            self.logger.log(f"  Mean Squared Error (MSE):  {mse:.4f}")
            self.logger.log(f"  Root Mean Squared Error (RMSE):{rmse:.4f}")
            self.logger.log(f"  Mean Absolute Error (MAE): {mae:.4f}")
            self.logger.log(f"  R-squared (R2):            {r2:.4f}")

            # --- Generate Plots ---
            self.logger.log("\nGenerating evaluation plots...")

            # 1. Predicted vs Actual
            self._plot_predicted_vs_actual(x_eval, y_eval, y_pred, eval_set_name, split_by_focal=True)

            # 2. Residuals Histogram
            # self._plot_residuals_histogram(residuals, eval_set_name)

            # 3. Residuals vs Features (dGaze_x, scr_dist)
            # features_to_plot_residuals = ['dGaze_x[pixels]', 'scr_dist[cm]']
            # for original_feature_name in features_to_plot_residuals:
            #     cleaned_feature_name = self._clean_col_name(original_feature_name)
            #     if cleaned_feature_name in x_eval.columns:
            #         self._plot_residuals_vs_feature(residuals, x_eval[cleaned_feature_name],
            #                                         cleaned_feature_name, eval_set_name)
            #     else:
            #         self.logger.log(
            #             f"  Warning: Feature '{cleaned_feature_name}' not found in {eval_set_name} data.")

            # 4. Error Heatmap (scr_dist vs dGaze_x)
            scr_dist_col_orig = 'scr_dist[cm]'  # Original name
            scr_dist_col_clean = self._clean_col_name(scr_dist_col_orig)  # Cleaned name

            if scr_dist_col_clean in x_eval.columns:
                self._plot_mae_heatmap_scr_dist_vs_d_actual(y_eval, y_pred, x_eval[scr_dist_col_clean], eval_set_name)
            else:
                self.logger.log(
                    f"  Warning: Feature '{scr_dist_col_clean}' not found in {eval_set_name} data. Skipping MAE Heatmap (Scr Dist vs Actual d).")

            # 5. Feature Importance
            self._plot_feature_importance(max_features=25)  # Show top 25 features

            self.logger.log("\nEvaluation complete.")

        except Exception as e:
            self.logger.log(f"ERROR during model evaluation or plotting: {e}")
            self.logger.log(traceback.format_exc())  # Log detailed traceback

    # --- save/load Methods ---
    def save(self):
        if not all([self.model, self.model_feature_cols is not None]):
            raise RuntimeError("model info missing.")
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.model_feature_cols, self.features_path)

        self.logger.log(f"model saved: {self.model_path}")
        self.logger.log(f"Features saved: {self.features_path}")

    def load(self):
        if not os.path.exists(self.model_path) or not os.path.exists(self.features_path):
            raise FileNotFoundError(f"Required files not found in {self.model_dir}.")
        try:
            self.model = joblib.load(self.model_path)
            self.model_feature_cols = joblib.load(self.features_path)

            self.logger.log("Model and features loaded successfully.")
            self.logger.log(f"Features ({len(self.model_feature_cols)}): {self.model_feature_cols}")
        except Exception as e:  # Consider more specific exceptions like joblib errors if possible
            raise RuntimeError(f"Error loading components from {self.model_dir}: {e}") from e

    def predict(self, input_data, *, has_units: bool = True):
        """
        Predicts the target value (regression) for the given input data.

        Args:
            input_data (pd.DataFrame or dict): Input data containing features.
            has_units: Whether the input data has units or not.

        Returns:
            float or np.ndarray: array of predicted values (if input was DataFrame).

        Raises:
            RuntimeError: If the model is not loaded/trained or prediction fails.
            TypeError: If input_data is not a dictionary or DataFrame.
            ValueError: If input data is missing required features or has type issues.
        """
        if self.model is None:
            msg = "Model is not loaded or trained. Cannot predict."
            self.logger.log(f"Error: {msg}")
            raise RuntimeError(msg)
        if self.model_feature_cols is None:
            msg = "Model features are not loaded. Cannot validate input or predict."
            self.logger.log(f"Error: {msg}")
            raise RuntimeError(msg)

        # --- Input Validation and Preparation ---
        if isinstance(input_data, pd.DataFrame):
            return self.predict_df(input_data, has_units=has_units)
        elif isinstance(input_data, dict):
            return self.predict_dict(input_data, has_units=has_units)
        else:
            raise TypeError("Input data must be a pandas DataFrame or a dictionary.")

    def predict_df(self, input_df: pd.DataFrame, *, has_units: bool = True):
        if has_units:
            input_df = self._clean_dataframe_cols(input_df.copy())

        # --- Feature Validation ---
        missing_cols = set(self.model_feature_cols) - set(input_df.columns)
        if missing_cols:
            raise ValueError(f"Input data is missing required features (cleaned names): {list(missing_cols)}")

        # Select and order features exactly as the model expects
        input_features = input_df[self.model_feature_cols]

        # --- Prediction ---
        predictions = self.model.predict(input_features)
        return predictions

    def predict_dict(self, input_datadict: dict, *, has_units: bool = True):
        if has_units:
            cleaned_datadict = {}
            for key, value in input_datadict.items():
                cleaned_key = self._clean_col_name(key)
                cleaned_datadict[cleaned_key] = value
            # Use the clean datadict
            input_datadict = cleaned_datadict

        # --- Feature Validation ---
        missing_cols = set(self.model_feature_cols) - set(input_datadict.keys())
        if missing_cols:
            raise ValueError(f"Input data is missing required features (cleaned names): {list(missing_cols)}")

        # Convert datadict to a 2D NumPy array with features in the correct order
        input_features = np.array([[input_datadict[feat] for feat in self.model_feature_cols]])

        # --- Prediction ---
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            predictions = self.model.predict(input_features)
        return predictions


# --- Main Block for Testing ---
def add_focal_xy(df: pd.DataFrame):
    if 'x' in df.columns and 'y' in df.columns:
        df['focal_x'] = df['x']
        df['focal_y'] = df['y']
        return ['focal_x', 'focal_y']
    elif 'xLeft' in df.columns and 'yLeft' in df.columns and 'xRight' in df.columns and 'yRight' in df.columns:
        df['focal_x'] = np.ceil((df['xLeft'] + df['xRight']) / 2).astype(int)
        df['focal_y'] = df['yLeft']
        return ['focal_x', 'focal_y']
    else:
        print('Error: failed to add_focal_xy')
        return []


ext_feature_func_dict = {
    'focal_xy[pixels]': add_focal_xy,
}

# --- Configuration (Copied from request & adapted) ---
CI_PREDICTION_FEATURE_COLS = [
    "left_pupil_x[]", "left_pupil_y[]", "left_pupil_z[]",
    "left_proj_point_x[]", "left_proj_point_y[]", "left_proj_point_z[]",
    "left_gaze_vector_x[]", "left_gaze_vector_y[]", "left_gaze_vector_z[]",
    "left_gaze_vector_r[]", "left_gaze_vector_p[rad]", "left_gaze_vector_t[rad]",
    "left_gaze_vector_scr_r[pixels]", "left_gaze_vector_scr_t[rad]",
    "right_pupil_x[]", "right_pupil_y[]", "right_pupil_z[]",
    "right_proj_point_x[]", "right_proj_point_y[]", "right_proj_point_z[]",
    "right_gaze_vector_x[]", "right_gaze_vector_y[]", "right_gaze_vector_z[]",
    "right_gaze_vector_r[]", "right_gaze_vector_p[]", "right_gaze_vector_t[]",
    "right_gaze_vector_scr_r[pixels]", "right_gaze_vector_scr_t[rad]",
    "dGaze_x[pixels]",
    "scr_dist[cm]",
    "head_pose_x[]", "head_pose_y[]",
    "left_gaze_vector_scr_x[pixels]", "left_gaze_vector_scr_y[pixels]",
    "right_gaze_vector_scr_x[pixels]", "right_gaze_vector_scr_y[pixels]"
]

CI_VALID_COL = 'valid[]'
CI_TARGET_COL = 'd[mm]'
CI_MODEL_SAVE_DIR = "convergence_model_final_v8_20250607_m56"


# --- Main Execution Function (Modified as requested) ---
def ci_model_train():
    # --- Use configuration as provided ---
    DATA_FOLDER = "Statistics"  # Assuming this folder exists and has data

    VAL_SET_SIZE = 0.2
    TEST_SET_SIZE = 0.1
    HISTORY_DEPTH = 10
    RANDOM_SEED = 42
    USE_HPTUNING = False
    N_ITER_SEARCH = 40
    N_CV_SPLIT = 5
    N_JOBS = -1

    EXT_DATASET_DIR = None

    # --- Initialize and Train (using exact code from request) ---
    print(f"--- Initializing ConvergenceModel ---")
    print(f"Model directory: {CI_MODEL_SAVE_DIR}")
    print(f"Data folder: {DATA_FOLDER}")
    print(f"Using HPTuning: {USE_HPTUNING} (Iterations: {N_ITER_SEARCH})")
    print(f"Features ({len(CI_PREDICTION_FEATURE_COLS)}): {CI_PREDICTION_FEATURE_COLS}")
    print("-" * 30)

    model = ConvergenceModel(
        feature_cols=CI_PREDICTION_FEATURE_COLS,
        valid_col=CI_VALID_COL,
        target_col=CI_TARGET_COL,
        noise_std=3.0,
        reduce_fps_flag=True,
        include_general_files=True,
        include_divergence_files=False,
        history_depth=HISTORY_DEPTH,
        ext_feature_func=ext_feature_func_dict,
        random_state=RANDOM_SEED,
        model_dir=CI_MODEL_SAVE_DIR,
        n_iter_search=N_ITER_SEARCH,
        n_splits_cv=N_CV_SPLIT,
        n_jobs=N_JOBS,
        only_print=False  # Generate log files
    )

    try:
        print(f"\n--- Starting Model Training ---")
        model.train(
            data_folder=DATA_FOLDER,
            load_ex_dataset_dir=EXT_DATASET_DIR,
            val_size=VAL_SET_SIZE,
            test_size=TEST_SET_SIZE,
            use_hptuning=USE_HPTUNING
        )
        print(f"\n--- Model Training Completed Successfully ---")

        # --- Prediction Example (using loaded model - Adapted for ConvergenceModel) ---
        print("\n--- Prediction Example (using loaded model) ---")

        # Instantiate the SAME class used for training to load the model
        predictor = ConvergenceModel(
            feature_cols=CI_PREDICTION_FEATURE_COLS,  # Provide features for consistency check
            model_dir=CI_MODEL_SAVE_DIR,
            only_print=True  # Don't overwrite logs when just loading/predicting
        )
        print(f"Loading model from: {predictor.model_dir}")
        predictor.load()  # Loads model and features

        # Sample input data (dictionary format)
        sample_input_dict = {
            # Features from CI_PREDICTION_FEATURE_COLS
            'left_pupil_x[pixels]': 357.0,
            'left_pupil_y[pixels]': 256.0,
            'right_pupil_x[pixels]': 429.0,
            'right_pupil_y[pixels]': 258.0,
            'left_proj_point_x[pixels]': 316.0,  # Corrected key from sample
            'left_proj_point_y[pixels]': 266.0,  # Corrected key from sample
            'right_proj_point_x[pixels]': 407.0,  # Corrected key from sample
            'right_proj_point_y[pixels]': 270.0,  # Corrected key from sample
            'head_pose_x[]': 0.1,
            'head_pose_y[]': -0.2,
            'dGaze_x[pixels]': 50.2,
            'scr_dist[cm]': 58.0,
        }

        # Verify all expected features are in the sample dictionary
        # Use the feature list *loaded* by the predictor instance
        expected_features_cleaned = predictor.model_feature_cols  # These are cleaned names
        provided_keys_cleaned = {predictor._clean_col_name(k): v for k, v in sample_input_dict.items()}

        missing_in_sample = []
        final_sample_data = {}

        for feature in expected_features_cleaned:
            if feature in provided_keys_cleaned:
                final_sample_data[feature] = provided_keys_cleaned[feature]
            else:
                # Try finding original feature name if available (requires storing original names)
                # For now, just report missing based on cleaned name and default to 0.0
                missing_in_sample.append(feature)  # Report missing cleaned feature name
                final_sample_data[feature] = 0.0  # Default value

        if missing_in_sample:
            print(f"\nWarning: Sample input was missing expected (cleaned) features: {missing_in_sample}")
            print("Using default value 0.0 for missing features.")

        # Convert the final dictionary (with potentially added defaults) to a DataFrame
        # The DataFrame must have column names matching the cleaned feature names
        sample_df = pd.DataFrame([final_sample_data])  # Create a single-row DataFrame
        # Ensure column order matches predictor.model_feature_cols before prediction
        sample_df = sample_df[predictor.model_feature_cols]

        print("\nInput DataFrame for prediction (1 sample):")
        print(sample_df.to_string())  # Use to_string to show all columns

        # Make the prediction
        prediction_result = predictor.predict(sample_df)

        print(
            f"\nPrediction Result ({predictor.target_col}): {prediction_result[0]:.4f}")  # Access first element for single prediction

    except ValueError as ve:
        print(f"\n--- TRAINING OR PREDICTION FAILED (ValueError) ---")
        print(f"Error: {ve}")
        print("Check data folder, file formats, column names, and configuration.")
        traceback.print_exc()
    except FileNotFoundError as fnf:
        print(f"\n--- TRAINING OR PREDICTION FAILED (FileNotFoundError) ---")
        print(f"Error: {fnf}")
        print(
            f"Ensure the data folder '{DATA_FOLDER}' exists or model files exist in '{CI_MODEL_SAVE_DIR}' if loading.")
        traceback.print_exc()
    except Exception as e:
        print(f"\n--- An Unexpected Error Occurred ---")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        print("-" * 30)
        traceback.print_exc()
        print("-" * 30)


if __name__ == "__main__":
    ci_model_train()
