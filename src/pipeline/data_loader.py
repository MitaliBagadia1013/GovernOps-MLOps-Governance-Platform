import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class FraudDatasetLoader:

    def __init__(self, n_samples=100000, random_state=42):
        self.n_samples = n_samples
        self.random_state = random_state
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.data_dir.mkdir(exist_ok=True)

    def load_or_generate_data(self) -> pd.DataFrame:
        dataset_path = self.data_dir / "fraud_detection_dataset.csv"
        if dataset_path.exists():
            logger.info(f"Loading existing dataset from {dataset_path}")
            df = pd.read_csv(dataset_path)
            logger.info(f"Loaded {len(df)} transactions from disk")
            return df
        logger.info(
            f"Generating new fraud detection dataset ({self.n_samples} samples)"
        )
        df = self._generate_realistic_fraud_data()
        df.to_csv(dataset_path, index=False)
        logger.info(f"Saved dataset to {dataset_path}")
        return df

    def _generate_realistic_fraud_data(self) -> pd.DataFrame:
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=28,
            n_informative=20,
            n_redundant=5,
            n_classes=2,
            weights=[0.9983, 0.0017],
            flip_y=0.01,
            random_state=self.random_state,
        )
        feature_columns = [f"V{i}" for i in range(1, 29)]
        df = pd.DataFrame(X, columns=feature_columns)
        np.random.seed(self.random_state)
        amounts = np.random.lognormal(mean=4.5, sigma=1.5, size=self.n_samples)
        amounts = np.clip(amounts, 0.01, 25000)
        df["Amount"] = amounts
        time = np.random.uniform(0, 172800, self.n_samples)
        df["Time"] = time
        df["Class"] = y
        fraud_count = y.sum()
        fraud_pct = fraud_count / self.n_samples * 100
        logger.info(f"Generated dataset statistics:")
        logger.info(f"  Total transactions: {self.n_samples:,}")
        logger.info(f"  Fraudulent: {fraud_count:,} ({fraud_pct:.4f}%)")
        logger.info(f"  Legitimate: {self.n_samples - fraud_count:,}")
        logger.info(
            f"  Features: {len(feature_columns) + 2} (28 V-features + Amount + Time)"
        )
        return df

    def prepare_train_test_split(
        self, df: pd.DataFrame, test_size: float = 0.2
    ) -> tuple:
        X = df.drop("Class", axis=1)
        y = df["Class"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        logger.info(f"Train/Test Split:")
        logger.info(f"  Training samples: {len(X_train):,}")
        logger.info(f"  Test samples: {len(X_test):,}")
        logger.info(f"  Training fraud rate: {y_train.mean():.4f}")
        logger.info(f"  Test fraud rate: {y_test.mean():.4f}")
        return (X_train, X_test, y_train, y_test)


def load_fraud_detection_data(n_samples: int = 100000) -> tuple:
    loader = FraudDatasetLoader(n_samples=n_samples)
    df = loader.load_or_generate_data()
    X_train, X_test, y_train, y_test = loader.prepare_train_test_split(df)
    feature_names = X_train.columns.tolist()
    return (X_train, X_test, y_train, y_test, feature_names)


def download_kaggle_fraud_dataset() -> pd.DataFrame:
    try:
        import kaggle
    except ImportError:
        raise ImportError(
            "Kaggle API not installed. Install with: pip install kaggle\nThen setup credentials: https://www.kaggle.com/docs/api"
        )
    data_dir = Path(__file__).parent.parent.parent / "data" / "kaggle"
    data_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Kaggle credit card fraud dataset...")
    kaggle.api.dataset_download_files(
        "mlg-ulb/creditcardfraud", path=data_dir, unzip=True
    )
    csv_path = data_dir / "creditcard.csv"
    df = pd.read_csv(csv_path)
    logger.info(f"Downloaded {len(df):,} real transactions from Kaggle")
    logger.info(f"Fraud rate: {df['Class'].mean():.4f}")
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 80)
    print("FRAUD DETECTION DATASET LOADER - TEST")
    print("=" * 80)
    X_train, X_test, y_train, y_test, features = load_fraud_detection_data(
        n_samples=50000
    )
    print(f"\nDataset loaded successfully!")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Test samples: {len(X_test):,}")
    print(f"   Features: {len(features)}")
    print(f"   Feature names: {features[:5]}... (showing first 5)")
    print(f"   Fraud rate (train): {y_train.mean():.4f}")
    print(f"   Fraud rate (test): {y_test.mean():.4f}")
