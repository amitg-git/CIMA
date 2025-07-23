import pandas as pd
import numpy as np
from matplotlib import cm
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_score, KFold
from scipy.stats import randint as sp_randint, uniform as sp_uniform, loguniform
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import os
import glob
import re  # For cleaning column names
from typing import List, Union

from helper.file import FileLogger


class loguniform_int:
    """Integer valued version of the log-uniform distribution"""

    def __init__(self, a, b):
        self._distribution = loguniform(a, b)

    def rvs(self, *args, **kwargs):
        """Random variable sample"""
        return self._distribution.rvs(*args, **kwargs).astype(int)


class FocalEstimationModel:
    """
    A class to train, evaluate, save, load, and use a LightGBM-based
    model for estimating focal coordinates from eye-tracking data.
    """

    def __init__(self,
                 feature_cols: Union[str, List[str]] = '',
                 valid_col='valid[]',
                 target_cols=['x[pixels]', 'y[pixels]'],
                 include_ci_files=False,
                 ci_d_col='d[mm]',
                 ci_x_cols=['xLeft[pixels]', 'xRight[pixels]'],
                 ci_y_cols=['yLeft[pixels]', 'yRight[pixels]'],
                 random_state=42,
                 model_dir='focal_model',
                 n_splits_cv=5,
                 n_iter_search=20,
                 n_jobs=-1,
                 only_print=True):
        """ Initializes the Focal Estimation model. """
        self.include_ci_files = include_ci_files

        self.model_feature_cols = self._clean_col_names_list(list(feature_cols))
        self.valid_col = self._clean_col_name(valid_col)
        self.target_cols = self._clean_col_names_list(list(target_cols))

        self.ci_d_col = self._clean_col_name(ci_d_col)
        self.ci_x_cols = self._clean_col_names_list(list(ci_x_cols))
        self.ci_y_cols = self._clean_col_names_list(list(ci_y_cols))
        self.ci_cols = self.ci_x_cols + self.ci_y_cols + [self.ci_d_col]

        self.random_state = random_state
        self.model_dir = model_dir
        print(f"Model directory: {self.model_dir}")
        self.n_splits_cv = n_splits_cv
        self.n_iter_search = n_iter_search
        self.n_jobs = n_jobs

        self.model_x = None
        self.model_y = None

        # Define file paths
        os.makedirs(self.model_dir, exist_ok=True)
        self.model_x_path = os.path.join(self.model_dir, 'model_focal_x.joblib')
        self.model_y_path = os.path.join(self.model_dir, 'model_focal_y.joblib')
        self.features_path = os.path.join(self.model_dir, 'features.joblib')
        self.log_path = os.path.join(self.model_dir, 'trained_files.txt')
        self.logger = FileLogger(os.path.join(self.model_dir, 'train_log.txt'), only_print=only_print)

        self.full_dataset = pd.DataFrame()
        self.dataset_dict = {
            'x_train': pd.DataFrame(), 'y_train': pd.DataFrame(),
            'x_val': pd.DataFrame(), 'y_val': pd.DataFrame(),
            'x_test': pd.DataFrame(), 'y_test': pd.DataFrame(),
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
            general_required_cols = set(self.model_feature_cols + [self.valid_col] + self.target_cols)
            ci_required_cols = set(self.model_feature_cols + [self.valid_col] + self.ci_cols)

            ci_file_types = ['convergence', 'divergence']
            is_ci_file = any(name in file_path for name in ci_file_types)
            required_cols = general_required_cols if not is_ci_file else ci_required_cols

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
        ci_file_types = ['convergence', 'divergence']

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
                    is_ci_file = any(name in fname_lower for name in ci_file_types)

                    if not is_ci_file:
                        general_files.append(f_path)
                    elif self.include_ci_files:
                        if "convergence" in fname_lower:
                            convergence_files.append(f_path)
                        elif "divergence" in fname_lower:
                            divergence_files.append(f_path)
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

    def create_dataset(self, convergence_files, divergence_files, general_files):
        """
        Create dataset for the model base on [convergence, divergence, general] file lists
        """
        self.logger.log("\nCreating dataset...")

        all_valid_files = {'general': [], 'convergence': [], 'divergence': []}

        file_lists_map = {'general': general_files, 'convergence': convergence_files, 'divergence': divergence_files}
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

                        # preprocess target column for non-general source types
                        if source_type != 'general':
                            valid_df = valid_df[valid_df[self.ci_d_col] == 0]
                            valid_df = valid_df.drop(columns=[self.ci_x_cols[1], self.ci_y_cols[1]])
                            valid_df = valid_df.rename(columns={
                                self.ci_x_cols[0]: self.target_cols[0], self.ci_y_cols[0]: self.target_cols[1]
                            })

                        # Select only necessary columns
                        required_cols = set(self.model_feature_cols + [self.valid_col] + self.target_cols)
                        valid_df = valid_df[list(required_cols)]

                        # add unique file index and source type for later grouping
                        valid_df.insert(0, 'file_index', unique_file_index)
                        unique_file_index += 1

                        self.full_dataset = pd.concat([self.full_dataset, valid_df], ignore_index=True, sort=False)
                    else:
                        self.logger.log(f'  Warning: {os.path.basename(fp)} is empty')

                except pd.errors.EmptyDataError:
                    self.logger.log(f"  Warn: Skipping empty file {os.path.basename(fp)}")
                except Exception as e:
                    self.logger.log(f"  Err loading {os.path.basename(fp)}: {e}")

        if self.full_dataset.empty:
            raise ValueError("No data loaded. Dataset is empty!")

        # log the used files in the full dataset
        self._log_used_files(all_valid_files['convergence'],
                             all_valid_files['divergence'],
                             all_valid_files['general'])
        self.logger.log("\nDone creating dataset...")

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
            filepath = os.path.join(self.model_dir, f"{key}.csv")
            data.to_csv(filepath, index=False, header=True if 'x_' in key else False)

        self.logger.log("\nDone saving Datasets...")

    def create_dataset_and_split(self, data_folder, *, val_size=0.15, test_size=0.15):
        """ Loads data, generates labels, splits"""
        convergence_files, divergence_files, general_files = self.find_categorized_files(data_folder)
        if not convergence_files and not divergence_files and not general_files:
            raise ValueError("No valid files found.")

        # Create dataset and save it
        self.create_dataset(convergence_files, divergence_files, general_files)

        # Split dataset depend on groups (unique files)
        self.split_dataset(val_size=val_size, test_size=test_size)

        # Save the Datasets
        self._save_dataset()

        self.logger.log(f"\n--- Training using {self.full_dataset['file_index'].nunique()} files) ---")

    def split_dataset(self, *, val_size=0.15, test_size=0.15):
        self.logger.log("--- Split Percentages ---")
        self.logger.log(f"Validation: {val_size:.0%}")
        self.logger.log(f"Test: {test_size:.0%}")

        df = self.full_dataset

        x_train, x_val, y_train, y_val = train_test_split(
            df[self.model_feature_cols + ['file_index']],
            df[self.target_cols],
            test_size=val_size, random_state=self.random_state
        )
        x_test = x_val
        y_test = y_val

        self.logger.log(f"\nDatasets Loading Summary:")
        self.logger.log(f"Train shapes - x: {x_train.shape}, y: {y_train.shape}")
        self.logger.log(f"Val shapes - x: {x_val.shape}, y: {y_val.shape}")
        self.logger.log(f"Test shapes - x: {x_test.shape}, y: {y_test.shape}")

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
        y_train_x = self.dataset_dict['y_train'].iloc[:, 0]
        y_train_y = self.dataset_dict['y_train'].iloc[:, 1]

        x_val = self.dataset_dict['x_val'].drop(columns=['file_index'])
        y_val_x = self.dataset_dict['y_val'].iloc[:, 0]
        y_val_y = self.dataset_dict['y_val'].iloc[:, 1]

        # Create groups for CV and remove 'file_index from features
        groups = x_train['file_index']
        x_train = x_train.drop(columns=['file_index'])

        if x_train.empty or len(y_train_x) != len(y_train_y) or len(y_train_x) != len(x_train):
            raise ValueError("Training data/labels mismatch.")

        self.logger.log(f"\n--- Training Model (HPTuning={use_hptuning}) ---")

        # --- Cross-Validation Step ---
        print(f"\nPerforming {self.n_splits_cv}-Fold Cross-Validation for performance estimation...")
        cv = GroupKFold(n_splits=self.n_splits_cv, shuffle=True, random_state=self.random_state)

        # Use default model parameters for CV estimate
        temp_model_x = lgb.LGBMRegressor(random_state=self.random_state, n_jobs=self.n_jobs, verbosity=-1)
        temp_model_y = lgb.LGBMRegressor(random_state=self.random_state, n_jobs=self.n_jobs, verbosity=-1)

        # Calculate CV scores (negative MAE)
        scores_x = cross_val_score(temp_model_x, x_train, y_train_x, groups=groups,
                                   cv=cv, scoring='neg_mean_absolute_error', n_jobs=self.n_jobs)
        scores_y = cross_val_score(temp_model_y, x_train, y_train_y, groups=groups,
                                   cv=cv, scoring='neg_mean_absolute_error', n_jobs=self.n_jobs)

        # Convert scores back to positive MAE for reporting
        mae_cv_x = -np.mean(scores_x)
        mae_cv_y = -np.mean(scores_y)
        std_cv_x = np.std(scores_x)
        std_cv_y = np.std(scores_y)

        print(f"\nCross-Validation Results ({self.n_splits_cv} folds):")
        print(f"  Average MAE X: {mae_cv_x:.2f} pixels (+/- {std_cv_x:.2f})")
        print(f"  Average MAE Y: {mae_cv_y:.2f} pixels (+/- {std_cv_y:.2f})")

        # --- Final Model Training ---
        print("\nTraining final models on the full training set...")

        self.model_x = lgb.LGBMRegressor(
            random_state=self.random_state, n_jobs=self.n_jobs
        )
        self.model_y = lgb.LGBMRegressor(
            random_state=self.random_state, n_jobs=self.n_jobs
        )

        evals_result_x = {}
        callbacks_x = [
            lgb.record_evaluation(evals_result_x),
            lgb.early_stopping(stopping_rounds=50, verbose=True, min_delta=0.01)
        ]
        self.model_x.fit(
            x_train, y_train_x,
            eval_set=[(x_train, y_train_x), (x_val, y_val_x)],
            eval_metric='l1',  # MAE,
            eval_names=['train', 'val'],
            callbacks=callbacks_x
        )

        evals_result_y = {}
        callbacks_y = [
            lgb.record_evaluation(evals_result_y),
            lgb.early_stopping(stopping_rounds=50, verbose=True, min_delta=0.01)
        ]

        self.model_y.fit(
            x_train, y_train_y,
            eval_set=[(x_train, y_train_y), (x_val, y_val_y)],
            eval_metric='l1',  # MAE,
            eval_names=['train', 'val'],
            callbacks=callbacks_y
        )

        self._plot_train_val_curves(evals_result_x, evals_result_y)

        self.logger.log("Training complete.")

    def _plot_train_val_curves(self, evals_result_x, evals_result_y):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))  # 2 rows, 1 column

        # Plot MAE curves for model_x
        ax1.plot(evals_result_x['train']['l1'], label='Train MAE (X)')
        ax1.plot(evals_result_x['val']['l1'], label='Val MAE (X)')
        if hasattr(self.model_x, 'best_iteration_') and self.model_x.best_iteration_:
            ax1.axvline(x=self.model_x.best_iteration_, color='r', linestyle='--',
                        label=f'Best Iteration (X: {self.model_x.best_iteration_})')
        ax1.set_title('Training and Validation MAE per Iteration (Model X)')
        ax1.set_xlabel('Boosting Iteration')
        ax1.set_ylabel('Mean Absolute Error (l1)')
        ax1.legend(loc='best')
        ax1.grid(True)

        # Plot MAE curves for model_y
        ax2.plot(evals_result_y['train']['l1'], label='Train MAE (Y)')
        ax2.plot(evals_result_y['val']['l1'], label='Val MAE (Y)')
        if hasattr(self.model_y, 'best_iteration_') and self.model_y.best_iteration_:
            ax2.axvline(x=self.model_y.best_iteration_, color='g', linestyle='--',
                        label=f'Best Iteration (Y: {self.model_y.best_iteration_})')
        ax2.set_title('Training and Validation MAE per Iteration (Model Y)')
        ax2.set_xlabel('Boosting Iteration')
        ax2.set_ylabel('Mean Absolute Error (l1)')
        ax2.legend(loc='best')
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, "training_mae_curve_xy.png"))
        plt.show()

    def evaluate_model(self):
        """
        Evaluates the trained model on the test set (or validation set if test is empty)
        and generates various performance visualization plots.
        """
        self.logger.log("\n" + "=" * 10 + " Evaluating Model Performance " + "=" * 10)

        if self.model_x is None or self.model_y is None:
            self.logger.log("Model is not trained or loaded. Cannot evaluate.")
            return

        eval_set_name = "Test"
        x_eval = self.dataset_dict.get('x_test')
        y_eval_x = self.dataset_dict.get('y_test').iloc[:, 0]
        y_eval_y = self.dataset_dict.get('y_test').iloc[:, 1]

        # Ensure data exists and is not empty
        if (x_eval is None or y_eval_x is None or y_eval_y is None or
                x_eval.empty or y_eval_x.empty or y_eval_y.empty):
            self.logger.log("Test set data is missing or empty.")
            # Fallback to validation set
            x_eval = self.dataset_dict.get('x_val')
            y_eval_x = self.dataset_dict.get('y_val').iloc[:, 0]
            y_eval_y = self.dataset_dict.get('y_val').iloc[:, 1]
            if (x_eval is None or y_eval_x is None or y_eval_y is None or
                    x_eval.empty or y_eval_x.empty or y_eval_y.empty):
                self.logger.log("Validation set data also missing or empty. Skipping evaluation.")
                return
            else:
                self.logger.log("Evaluating on Validation Set instead.")
                eval_set_name = "Validation"

        # Check for length mismatch
        if not (len(x_eval) == len(y_eval_x) == len(y_eval_y)):
            self.logger.log(
                f"ERROR: {eval_set_name} data/label length mismatch: "
                f"X={len(x_eval)}, y_x={len(y_eval_x)}, y_y={len(y_eval_y)}. Skipping evaluation."
            )
            return

        x_eval = x_eval.drop(columns=['file_index'])
        self.logger.log(f"Evaluating on {eval_set_name} Set ({len(x_eval)} samples)...")

        y_pred_x = self.model_x.predict(x_eval)
        y_pred_y = self.model_y.predict(x_eval)

        mae_x = mean_absolute_error(y_eval_x, y_pred_x)
        mse_x = mean_squared_error(y_eval_x, y_pred_x)
        rmse_x = np.sqrt(mse_x)
        mae_y = mean_absolute_error(y_eval_y, y_pred_y)
        mse_y = mean_squared_error(y_eval_y, y_pred_y)
        rmse_y = np.sqrt(mse_y)

        self.logger.log(f"MAE X: {mae_x:.2f} pixels")
        self.logger.log(f"RMSE X: {rmse_x:.2f} pixels")
        self.logger.log(f"MAE Y: {mae_y:.2f} pixels")
        self.logger.log(f"RMSE Y: {rmse_y:.2f} pixels")

        errors = np.sqrt((y_eval_x.values - y_pred_x) ** 2 + (y_eval_y.values - y_pred_y) ** 2)
        mean_euclidean_error = np.mean(errors)
        self.logger.log(f"\nMean Euclidean Error: {mean_euclidean_error:.2f} pixels")

        # --- Visualization ---
        self.logger.log("Generating evaluation plots for the final model...")
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(y_eval_x, y_pred_x, alpha=0.3, label='Pred vs Actual X', s=5)
        plt.plot([y_eval_x.min(), y_eval_x.max()], [y_eval_x.min(), y_eval_x.max()], 'r--', label='Ideal')
        plt.xlabel("Actual X (pixels)")
        plt.ylabel("Predicted X (pixels)")
        plt.title("X - Final Model Evaluation")
        plt.grid(True)
        plt.axis('equal')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.scatter(y_eval_y, y_pred_y, alpha=0.3, label='Pred vs Actual Y', s=5)
        plt.plot([y_eval_y.min(), y_eval_y.max()], [y_eval_y.min(), y_eval_y.max()], 'r--', label='Ideal')
        plt.xlabel("Actual Y (pixels)")
        plt.ylabel("Predicted Y (pixels)")
        plt.title("Y - Final Model Evaluation")
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(GAZE_MODEL_SAVE_DIR, "Final Model Evaluation.png"))
        plt.show()

        def group_points_by_actual_pairs(y_test_x, y_test_y, y_pred_x, y_pred_y):
            """
            Groups actual and predicted points by unique (x, y) pairs in actual points.

            Parameters:
                y_test_x (pd.Series or pd.DataFrame): Actual x-coordinates.
                y_test_y (pd.Series or pd.DataFrame): Actual y-coordinates.
                y_pred_x (np.ndarray or pd.Series): Predicted x-coordinates.
                y_pred_y (np.ndarray or pd.Series): Predicted y-coordinates.

            Returns:
                points_groups (list of pd.DataFrame): List of DataFrames of actual points grouped by unique pairs.
                pred_points_groups (list of pd.DataFrame): List of DataFrames of predicted points corresponding to each group.
            """
            # Combine actual x and y into a DataFrame
            points = pd.concat([y_test_x.reset_index(drop=True), y_test_y.reset_index(drop=True)], axis=1)
            points.columns = ['x', 'y']

            # Group by unique (x, y) pairs
            grouped = points.groupby(['x', 'y'])

            points_groups = []
            pred_points_groups = []

            for _, group_indices in grouped.groups.items():
                idx = list(group_indices)  # Indices of samples in this group

                # Actual points group
                actual_group = points.iloc[idx]
                points_groups.append(actual_group)

                # Predicted points group
                pred_group = pd.DataFrame({
                    'x': y_pred_x[idx],
                    'y': y_pred_y[idx]
                })
                pred_points_groups.append(pred_group)

            return points_groups, pred_points_groups

        points_groups, pred_points_groups = group_points_by_actual_pairs(y_eval_x, y_eval_y, y_pred_x, y_pred_y)
        n_groups = len(points_groups)
        colors = cm.get_cmap('tab20', n_groups)

        plt.figure(figsize=(8, 8))
        for i, (actual_group, pred_group) in enumerate(zip(points_groups, pred_points_groups)):
            color = colors(i)
            plt.scatter(actual_group['x'], actual_group['y'], color=color, label=f'Actual Group {i + 1}', s=30,
                        edgecolor='k')
            plt.scatter(pred_group['x'], pred_group['y'], color=color, marker='x', s=50)

        plt.xlabel("X (pixels)")
        plt.ylabel("Y (pixels)")
        plt.title("Actual vs. Predicted Locations Grouped by Unique Pairs")
        plt.legend(loc='best', fontsize='small', ncol=2)
        plt.grid(True)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(os.path.join(GAZE_MODEL_SAVE_DIR, "Actual vs. Predicted Locations.png"))
        plt.show()

    def save(self):
        """Saves the trained models and feature list to disk."""
        if self.model_x is None or self.model_y is None or self.model_feature_cols is None:
            raise RuntimeError("Model has not been trained yet. Cannot save.")
        joblib.dump(self.model_x, self.model_x_path)
        joblib.dump(self.model_y, self.model_y_path)
        joblib.dump(self.model_feature_cols, self.features_path)
        self.logger.log(f"Models saved to {self.model_x_path}, {self.model_y_path}")
        self.logger.log(f"Feature list saved to {self.features_path}")

    def load(self):
        """Loads trained models and feature list from disk."""
        if not all(os.path.exists(p) for p in [self.model_x_path, self.model_y_path, self.features_path]):
            raise FileNotFoundError(f"Model files not found in {self.model_dir}.")
        self.model_x = joblib.load(self.model_x_path)
        self.model_y = joblib.load(self.model_y_path)
        self.model_feature_cols = joblib.load(self.features_path)
        self.logger.log("Models and feature list loaded successfully.")
        self.logger.log(f"Model trained with features: {self.model_feature_cols}")

    def predict(self, input_data):
        """Predicts gaze coordinates (x, y) for new input data."""
        if self.model_x is None or self.model_y is None:
            msg = "Model is not loaded or trained. Cannot predict."
            self.logger.log(f"Error: {msg}")
            raise RuntimeError(msg)
        if self.model_feature_cols is None:
            msg = "Model features are not loaded. Cannot validate input or predict."
            self.logger.log(f"Error: {msg}")
            raise RuntimeError(msg)

        is_dict_input = isinstance(input_data, dict)

        # --- Input Validation and Preparation ---
        if is_dict_input:
            # Convert single dict to a single-row DataFrame
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data.copy()  # Work on a copy
        else:
            raise TypeError("Input data must be a pandas DataFrame or a dictionary.")

        if input_df.empty:
            raise ValueError("Input data cannot be empty.")

        # Remove units from column names of the input data
        try:
            input_df_cleaned = self._clean_dataframe_cols(input_df.copy())
        except Exception as e:
            self.logger.log(f"Error cleaning input data columns: {e}")
            raise ValueError("Failed to clean input data column names.") from e

        # --- Feature Validation ---
        # Check for missing features based on *cleaned* names stored in self.model_feature_cols
        missing_cols = set(self.model_feature_cols) - set(input_df_cleaned.columns)
        if missing_cols:
            raise ValueError(f"Input data is missing required features (cleaned names): {list(missing_cols)}")

        # Select and order features exactly as the model expects
        try:
            # Use the cleaned feature list
            input_features = input_df_cleaned[self.model_feature_cols]
        except KeyError as e:
            # This check is somewhat redundant due to missing_cols check, but acts as a safeguard
            raise ValueError(f"Column mismatch error during feature selection using cleaned names: {e}") from e

        # --- Prediction ---
        try:
            pred_x = self.model_x.predict(input_features)
            pred_y = self.model_y.predict(input_features)
        except Exception as e:
            self.logger.log(f"Prediction error: {e}")
            return None

        return np.vstack((pred_x, pred_y)).T


# --- Configuration ---
GAZE_MODEL_FEATURE_COLS = [
    'left_pupil_x[]', 'left_pupil_y[]',
    'right_pupil_x[]', 'right_pupil_y[]',
    'left_gaze_vector_x[]', 'left_gaze_vector_y[]',
    'right_gaze_vector_x[]', 'right_gaze_vector_y[]',
    'head_pose_x[]', 'head_pose_y[]',
    'scr_dist[cm]'
]
GAZE_MODEL_TARGET_COLS = ['x[pixels]', 'y[pixels]']
GAZE_MODEL_VALID_COL = 'valid[]'
GAZE_MODEL_SAVE_DIR = "focal_model"


# --- Main Execution Example ---
def focal_model_train():
    DATA_FOLDER = "./Statistics"

    VAL_SET_SIZE = 0.2
    TEST_SET_SIZE = 0
    RANDOM_SEED = 42
    USE_HPTUNING = False
    N_ITER_SEARCH = 20
    N_CV_SPLIT = 5
    N_JOBS = -1

    CI_MODEL_D_COL = 'd[mm]'
    CI_MODEL_X_COLS = ['xLeft[pixels]', 'xRight[pixels]']
    CI_MODEL_Y_COLS = ['yLeft[pixels]', 'yRight[pixels]']

    EXT_DATASET_DIR = None

    # --- Step 1: Initialize and Load ALL available valid data ---
    print("--- Initializing Model (for data loading) ---")
    # Create a temporary instance just to use its loading methods
    model = FocalEstimationModel(
        feature_cols=GAZE_MODEL_FEATURE_COLS,
        valid_col=GAZE_MODEL_VALID_COL,
        target_cols=GAZE_MODEL_TARGET_COLS,
        include_ci_files=False,
        ci_d_col=CI_MODEL_D_COL,
        ci_x_cols=CI_MODEL_X_COLS,
        ci_y_cols=CI_MODEL_Y_COLS,
        random_state=RANDOM_SEED,
        model_dir=GAZE_MODEL_SAVE_DIR,
        n_splits_cv=N_CV_SPLIT,
        n_iter_search=N_ITER_SEARCH,
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
        predictor = FocalEstimationModel()

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
        predicted_coords = predictor.predict(sample_df)
        if predicted_coords is not None:
            print(f"\nPredicted Coords [x, y]: {predicted_coords[0]}")

    except (FileNotFoundError, ValueError, RuntimeError, TypeError) as e:
        print(f"\nERROR: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    focal_model_train()
