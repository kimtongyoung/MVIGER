import pdb
from typing import Dict, List
import json
import random


class TreeNode:
    def __init__(self, idx):
        self.idx = idx
        self.children = {}


class SampleTree:
    def __init__(self):
        self.root = TreeNode(None)

    def insert(self, path):
        current = self.root
        for level in path:
            if level not in current.children:
                current.children[level] = TreeNode(level)
            current = current.children[level]

    def search_children(self, path):
        current = self.root
        for level in path:
            if level in current.children:
                current = current.children[level]
            else:
                return None
        return list(current.children.keys())

    def select_candidate(self, path, avoid=None):
        candidates = self.search_children(path).copy()
        if avoid:
            candidates.remove(avoid)
        if not candidates:
            return None
        selected_candidate = random.choice(candidates)

        return selected_candidate

    def sampling_negative(self, gt):
        assert len(gt) >= 2
        sample, avoid = gt[:-2], gt[-2]
        while True:
            selected = self.select_candidate(sample, avoid)
            if selected is None:
                if gt[0] == 'Beauty':
                    sample = ['Beauty']
                    avoid = gt[1]
                elif gt[0] == 'Sports_and_Outdoors':
                    sample = ['Sports_and_Outdoors']
                    avoid = gt[1]
                else:
                    sample = []
                    avoid = gt[0]
                continue
            else:
                sample.append(selected)
                avoid = None

            if selected[:4] == 'Leaf':
                break
        return sample
