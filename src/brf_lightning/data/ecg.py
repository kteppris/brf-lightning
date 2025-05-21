import lightning as L
from torch.utils.data import DataLoader, random_split, TensorDataset
import scipy.io
from pathlib import Path
import torch
import numpy as np

class ECGDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for handling ECG data.

    This module loads ECG data from .mat files, preprocesses it,
    and sets up DataLoaders for training, validation, and testing.

    Attributes:
        data_dir (Path): Directory where the data files are stored.
        batch_size (int): Batch size for the training DataLoader.
        num_workers (int): Number of workers for data loading.
        validation_split (float): Fraction of the training data to use for validation.
        train_dataset (TensorDataset | None): Training dataset.
        val_dataset (TensorDataset | None): Validation dataset.
        test_dataset (TensorDataset | None): Test dataset.
    """
    def __init__(
            self, 
            data_dir: str, 
            batch_size: int = 16, 
            num_workers: int = 4, 
            validation_split: float = 0.1
        ):
        """
        Args:
            data_dir (str): Directory where the data files are stored.
            batch_size (int): Batch size for the training DataLoader. Defaults to 16.
            num_workers (int): Number of workers for data loading. Defaults to 4.
            validation_split (float): Fraction of the training data to use for validation.
                                      Must be between 0 and 1. Defaults to 0.1.
        """
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        if not 0.0 <= validation_split < 1.0:
            raise ValueError("validation_split must be between 0.0 and 1.0")
        self.validation_split = validation_split

        # Datasets will be initialized in setup()
        self.train_dataset: TensorDataset | None = None
        self.val_dataset: TensorDataset | None = None
        self.test_dataset: TensorDataset | None = None

    def prepare_data(self):
        """
        In this case, data is assumed to be already present locally.
        """
        # Example: check if data files exist
        if not (self.data_dir / 'QTDB_train.mat').exists():
            raise FileNotFoundError(f"Training data QTDB_train.mat not found in {self.data_dir}")
        if not (self.data_dir / 'QTDB_test.mat').exists():
            raise FileNotFoundError(f"Test data QTDB_test.mat not found in {self.data_dir}")

    def setup(self, stage: str | None = None):
        """
        Load data and split into train, validation, and test sets.
        This hook is called on every GPU.

        Args:
            stage (str | None): Current stage ('fit', 'validate', 'test', 'predict').
                                Defaults to None.
        """
        # Load full training data and test data
        # The _load_data method handles loading and initial conversion
        full_train_dataset = self._load_data(self.data_dir / 'QTDB_train.mat')
        self.test_dataset = self._load_data(self.data_dir / 'QTDB_test.mat')

        # Split full_train_dataset into training and validation sets
        if self.validation_split > 0:
            val_size = int(self.validation_split * len(full_train_dataset))
            train_size = len(full_train_dataset) - val_size
            self.train_dataset, self.val_dataset = random_split(full_train_dataset, [train_size, val_size])
        else:
            self.train_dataset = full_train_dataset
            # If no validation split, val_dataset can be None or a small subset if needed for sanity checks
            self.val_dataset = None 

    def _load_data(self, file_path: Path) -> TensorDataset:
        """
        Loads data from a .mat file and converts it to a TensorDataset.

        Args:
            file_path (Path): Path to the .mat file.

        Returns:
            TensorDataset: Dataset containing processed inputs and targets.
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        mat_contents = scipy.io.loadmat(file_path)
        # The convert_data_format function handles the specific transformations
        # required for the ECG data.
        return convert_data_format(mat_contents)

    def _create_dataloader(self, dataset: TensorDataset | None, shuffle: bool, use_full_batch: bool = False) -> DataLoader | None:
        """
        Helper function to create a DataLoader.

        Args:
            dataset (TensorDataset | None): The dataset to load. If None, returns None.
            shuffle (bool): Whether to shuffle the data.
            use_full_batch (bool): If True, sets batch size to the entire dataset length.
                                   Typically used for validation/test.

        Returns:
            DataLoader | None: The configured DataLoader, or None if dataset is None.
        """
        if dataset is None:
            return None
            
        batch_size = len(dataset) if use_full_batch else self.batch_size
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(), # Automatically uses GPU if available
            shuffle=shuffle,
            persistent_workers=True if self.num_workers > 0 else False # Can speed up training
        )

    def train_dataloader(self) -> DataLoader | None:
        """Returns the DataLoader for the training set."""
        return self._create_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader | None:
        """
        Returns the DataLoader for the validation set.
        Data is not shuffled, and the entire validation set is typically loaded as one batch.
        """
        # For validation, we usually want to evaluate on the entire dataset in one go if memory permits.
        return self._create_dataloader(self.val_dataset, shuffle=False, use_full_batch=True)

    def test_dataloader(self) -> DataLoader | None:
        """
        Returns the DataLoader for the test set.
        Data is not shuffled, and the entire test set is typically loaded as one batch.
        """
        # For testing, similar to validation, evaluate on the entire dataset.
        return self._create_dataloader(self.test_dataset, shuffle=False, use_full_batch=True)

def convert_data_format(data: dict[str, np.ndarray]) -> TensorDataset:
    """
    Converts raw data loaded from .mat file into a TensorDataset.

    The function expects specific keys ('x' for inputs, 'y' for targets)
    in the input dictionary. It performs preprocessing steps such as:
    1. Converting NumPy arrays to PyTorch tensors.
    2. Changing data type to float.
    3. Removing the last time step from sequences (assumed to be non-informative or padding).

    Args:
        data (dict[str, np.ndarray]): Dictionary loaded from a .mat file,
                                      expected to contain 'x' and 'y' keys
                                      with NumPy arrays as values.
                                      - 'x' shape: (num_samples, sequence_length, num_input_features)
                                      - 'y' shape: (num_samples, sequence_length, num_target_features)

    Returns:
        TensorDataset: A dataset containing the processed input and target tensors.
                       - Processed inputs shape: (num_samples, sequence_length - 1, num_input_features)
                       - Processed targets shape: (num_samples, sequence_length - 1, num_target_features)
    
    Raises:
        KeyError: If 'x' or 'y' keys are not found in the input data dictionary.
    """
    if 'x' not in data or 'y' not in data:
        raise KeyError("Input data dictionary must contain 'x' and 'y' keys.")

    # --- Input Data Processing ---
    # Original input shape example: (num_samples, 1301, 4)
    inputs_np = data['x']
    inputs_torch = torch.from_numpy(inputs_np).to(torch.float32) # Explicitly use float32

    # Remove the last time step from the sequence.
    # This is a common preprocessing step if the last step is padding or irrelevant.
    # New shape example: (num_samples, 1300, 4)
    processed_inputs = inputs_torch[:, :-1, :]

    # --- Target Data Processing ---
    # Original target shape example: (num_samples, 1301, 6)
    # Targets are expected to be one-hot encoded for each time step.
    targets_np = data['y']
    targets_torch = torch.from_numpy(targets_np).to(torch.float32) # Explicitly use float32

    # Remove the last time step from the sequence, consistent with inputs.
    # New shape example: (num_samples, 1300, 6)
    processed_targets = targets_torch[:, :-1, :]

    # Create a TensorDataset from the processed inputs and targets.
    # TensorDataset wraps tensors; each sample will be retrieved by indexing tensors along the first dimension.
    dataset = TensorDataset(processed_inputs, processed_targets)

    return dataset