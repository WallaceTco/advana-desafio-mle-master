import pandas as pd
import numpy as np
from typing import Tuple, Union, List, Optional
from sklearn.linear_model import LogisticRegression

class DelayModel:
    """
    DelayModel
    -----------
    Predicts flight delays based on basic flight information.
    Uses a Logistic Regression model with One-Hot Encoded (OHE) categorical features.

    This class reproduces the logic from the Data Scientist's notebook (`exploration.ipynb`)
    and is designed for both training and inference.
    """

    def __init__(self) -> None:
        """
        Initialize the DelayModel.

        Attributes:
            _model (LogisticRegression | None): trained logistic regression model.
            _feature_columns (List[str] | None): feature columns used during training,
                                                 required for alignment at inference time.
        """
        self._model: Optional[LogisticRegression] = None
        self._feature_columns: Optional[List[str]] = None

    # -----------------------------------------
    # ---------- Internal Utilities -----------
    # -----------------------------------------
    @staticmethod
    def _ensure_delay_column(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Ensures that the target column (`delay`) exists.

        If the column is missing but both 'Fecha-I' (scheduled) and 'Fecha-O' (operated)
        are available, it computes the delay in minutes and assigns:
            - 1 if delay > 15 minutes
            - 0 otherwise

        Args:
            df (pd.DataFrame): raw flight data.
            target_column (str): name of the target column ('delay').

        Returns:
            pd.DataFrame: DataFrame that includes the target column.
        """
        if target_column in df.columns:
            return df

        if {"Fecha-I", "Fecha-O"}.issubset(df.columns):
            dfx = df.copy()
            fi = pd.to_datetime(dfx["Fecha-I"], errors="coerce")
            fo = pd.to_datetime(dfx["Fecha-O"], errors="coerce")
            min_diff = (fo - fi).dt.total_seconds() / 60.0
            dfx[target_column] = (min_diff > 15).astype(int)
            return dfx

        raise ValueError(
            f"Target '{target_column}' does not exist and cannot be derived without 'Fecha-I' and 'Fecha-O'."
        )

    @staticmethod
    def _build_ohe_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Builds input features using One-Hot Encoding.

        Based on the Data Scientist’s final setup, the relevant variables are:
        - OPERA: airline
        - TIPOVUELO: flight type (I/N)
        - MES: month number

        Args:
            df (pd.DataFrame): raw flight data containing the required columns.

        Returns:
            pd.DataFrame: numeric DataFrame ready for model training or prediction.
        """
        required = ["OPERA", "TIPOVUELO", "MES"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for feature generation: {missing}")

        feats = pd.concat(
            [
                pd.get_dummies(df["OPERA"], prefix="OPERA"),
                pd.get_dummies(df["TIPOVUELO"], prefix="TIPOVUELO"),
                pd.get_dummies(df["MES"], prefix="MES"),
            ],
            axis=1,
        ).astype(np.float32)

        return feats

    # -----------------------------------------
    # ---------- API pública requerida --------
    # -----------------------------------------
    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
        """
        Preprocess raw flight data for training or prediction.

        If a target column is provided, returns a tuple (X, y),
        where X are the encoded features and y is the target series.
        If not provided, returns only X.

        Args:
            data (pd.DataFrame): raw flight data.
            target_column (str, optional): name of the target column ('delay').

        Returns:
            Union[Tuple[pd.DataFrame, pd.Series], pd.DataFrame]:
                - (X, y) if target_column is provided
                - X only if not provided

        Example:
            X, y = model.preprocess(df, target_column="delay")
            X_new = model.preprocess(df_new)
        """
        df = data.copy()

        if target_column:
            df = self._ensure_delay_column(df, target_column)
            y = df[target_column].astype(int)

        X = self._build_ohe_features(df)

        if target_column:
            return X, y

        return X

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Train the logistic regression model using preprocessed data.

        Args:
            features (pd.DataFrame): input variables (X).
            target (pd.Series | pd.DataFrame): target variable (y).

        Notes:
            Stores the feature column names internally to ensure correct alignment
            when predicting on new data.
        """
        if not isinstance(target, (pd.Series, pd.DataFrame)):
            raise ValueError("target must be a pandas Series or DataFrame")

        y = target.squeeze().astype(int)
        X = features.copy()

        self._feature_columns = X.columns.tolist()

        self._model = LogisticRegression(
            max_iter=200,
            class_weight="balanced",
            random_state=42,
        )
        self._model.fit(X, y)

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predicts flight delay labels (0 or 1) for new input data.

        The method automatically aligns input columns with those used during training,
        filling missing ones with zeros and discarding unknown columns.

        Args:
            features (pd.DataFrame): preprocessed features for prediction.

        Returns:
            List[int]: list of predicted delay labels
                       (0 = on time, 1 = delayed).

        Example:
            preds = model.predict(X_new)
        """
        if self._model is None or self._feature_columns is None:
            raise RuntimeError("Model has not been trained. Call fit() before predict().")

        X = features.copy()

        for col in self._feature_columns:
            if col not in X.columns:
                X[col] = 0.0

        X = X[self._feature_columns]

        preds = self._model.predict(X)
        return preds.astype(int).tolist()