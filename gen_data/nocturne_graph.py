import os
import torch
from typing import List
from torch_geometric.data import HeteroData, InMemoryDataset


class NocturneScenarioGraph(InMemoryDataset):
    def __init__(
        self,
        root,
        json_id: int, scene_id: int,
        data_list: List[HeteroData] = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.data_list = data_list
        self.json_id = json_id
        self.scene_id = scene_id
        super().__init__(root, transform, pre_filter=pre_filter,
                         pre_transform=pre_transform, log=False)

        if self.data_list is not None:
            self.process()
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f"{self.json_id}_{self.scene_id}.pt"]

    @property
    def processed_dir(self) -> str:
        return self.root

    def process(self):
        """ Save the [Data] into disk. """
        os.makedirs(self.processed_dir, exist_ok=True)
        data, slices = self.collate(self.data_list)
        torch.save((data, slices), self.processed_paths[0])