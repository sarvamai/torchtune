from typing import Dict, Optional
import torch
from torch.utils.data import Dataset
from torchtune.datasets._preference import PreferenceDataset
import pdb

class CustomPreferenceDataset(Dataset):
    """
    A dataset class for preference data that can store precomputed reference log probabilities.
    """
    def __init__(self, 
            base_dataset: PreferenceDataset,
        ) -> None:
        """
        Initialize the preference dataset.
        
        Args:
            base_dataset: The underlying dataset containing preference pairs
        """
        self.base_dataset = base_dataset
        self.ref_chosen_logps: Optional[torch.Tensor] = None
        self.ref_rejected_logps: Optional[torch.Tensor] = None

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> Dict:
        item = self.base_dataset[idx]

        if self.ref_chosen_logps is not None:
            item['ref_chosen_logps'] = self.ref_chosen_logps[idx]
        if self.ref_rejected_logps is not None:
            item['ref_rejected_logps'] = self.ref_rejected_logps[idx]
            
        return item

    def add_reference_log_probs(self, 
                              ref_chosen_logps: torch.Tensor, 
                              ref_rejected_logps: torch.Tensor) -> None:
        """
        Add precomputed reference log probabilities to the dataset.
        
        Args:
            ref_chosen_logps: Log probabilities for chosen responses
            ref_rejected_logps: Log probabilities for rejected responses
        """
        self.ref_chosen_logps = ref_chosen_logps
        self.ref_rejected_logps = ref_rejected_logps 