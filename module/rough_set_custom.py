# module/rough_set_custom.py – Tập thô tự cài đặt
import pandas as pd
from itertools import combinations

class RoughSet:
    def __init__(self, df, condition_cols, decision_col):
        self.df = df.copy()
        self.C = condition_cols
        self.D = decision_col

    def indiscernibility(self, cols):
        # Phân lớp tương đương dựa trên tập thuộc tính
        groups = self.df.groupby(cols)
        return [set(idx) for _, idx in groups.groups.items()]

    def lower_approx(self, target_val):
        # Tập các dòng mà class = target_val
        decision_block = set(self.df[self.df[self.D] == target_val].index)
        condition_blocks = self.indiscernibility(self.C)
        return set.union(*[block for block in condition_blocks if block <= decision_block])

    def upper_approx(self, target_val):
        decision_block = set(self.df[self.df[self.D] == target_val].index)
        condition_blocks = self.indiscernibility(self.C)
        return set.union(*[block for block in condition_blocks if block & decision_block])

    def positive_region(self):
        pos_region = set()
        decisions = self.df[self.D].unique()
        for d in decisions:
            pos_region |= self.lower_approx(d)
        return pos_region

    def dependency_degree(self):
        pos = self.positive_region()
        return len(pos) / len(self.df)

    def find_reduct(self):
        best_reduct = self.C.copy()
        for r in range(1, len(self.C) + 1):
            for subset in combinations(self.C, r):
                approx = RoughSet(self.df, list(subset), self.D)
                if approx.positive_region() == self.positive_region():
                    return list(subset)
        return best_reduct

    def generate_rules(self):
        rules = []
        condition_blocks = self.indiscernibility(self.C)
        for block in condition_blocks:
            decision_vals = self.df.loc[block, self.D].unique()
            if len(decision_vals) == 1:
                condition_values = self.df.loc[list(block)].iloc[0][self.C].to_dict()
                rules.append({
                    'conditions': condition_values,
                    'decision': decision_vals[0]
                })
        return rules
