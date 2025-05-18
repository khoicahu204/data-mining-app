
import pandas as pd
import math

class NaiveBayesClassifier:
    def __init__(self):
        self.class_probs = {}  # P(class)
        self.feature_probs = {}  # P(feature=value | class)
        self.classes = []

    def fit(self, df: pd.DataFrame, target_col: str):
        self.classes = df[target_col].unique()
        total_rows = len(df)

        for c in self.classes:
            df_c = df[df[target_col] == c]
            self.class_probs[c] = len(df_c) / total_rows
            self.feature_probs[c] = {}

            for col in df.columns:
                if col == target_col:
                    continue
                self.feature_probs[c][col] = {}
                value_counts = df_c[col].value_counts()
                total = len(df_c)

                for val in df[col].unique():
                    # Laplace smoothing
                    count = value_counts.get(val, 0)
                    prob = (count + 1) / (total + len(df[col].unique()))
                    self.feature_probs[c][col][val] = prob

    def predict(self, sample: dict):
        probs = {}
        for c in self.classes:
            prob_c = math.log(self.class_probs[c])
            for feature, value in sample.items():
                prob = self.feature_probs[c].get(feature, {}).get(value, 1e-6)
                prob_c += math.log(prob)
            probs[c] = prob_c

        # Trả về class có log xác suất lớn nhất
        return max(probs, key=probs.get), probs
