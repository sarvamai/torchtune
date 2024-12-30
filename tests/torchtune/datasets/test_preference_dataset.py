# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Mapping
from unittest import mock

import pytest
import torch
from tests.common import ASSETS
from tests.test_utils import DummyTokenizer
from torchtune.data import Message
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.datasets._preference import preference_dataset, PreferenceDataset, CustomPreferenceDataset
from torchtune.modules.transforms import Transform


class ToDummyPreferenceMessages(Transform):
    def __call__(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        chosen_messages = [
            Message.from_dict(sample["prompt"][0]),
            Message.from_dict(sample["chosen"][0]),
        ]

        rejected_messages = [
            Message.from_dict(sample["prompt"][0]),
            Message.from_dict(sample["rejected"][0]),
        ]

        return {"chosen": chosen_messages, "rejected": rejected_messages}


class TestPreferenceDataset:
    @pytest.fixture
    def dialogue(self):
        return [
            {
                "prompt": [
                    {
                        "role": "user",
                        "content": "What is 2+2?",
                        "masked": True,
                    },
                ],
                "chosen": [
                    {
                        "role": "assistant",
                        "content": "The answer is 4.",
                        "masked": False,
                    },
                ],
                "rejected": [
                    {
                        "role": "assistant",
                        "content": "The answer is 12.",
                        "masked": False,
                    },
                ],
            },
        ]

    @pytest.fixture
    def expected(self):
        return {
            "prompt": [
                0,
                4,
                2,
                4,
            ],
            "chosen": [
                3,
                6,
                2,
                2,
                -1,
            ],
            "rejected": [
                3,
                6,
                2,
                3,
                -1,
            ],
        }

    @mock.patch("torchtune.datasets._preference.load_dataset")
    def test_get_item(self, mock_load_dataset, dialogue, expected):
        mock_load_dataset.return_value = dialogue
        expected_chosen_tokens = expected["prompt"] + expected["chosen"]
        expected_chosen_labels = [CROSS_ENTROPY_IGNORE_IDX] * len(
            expected["prompt"]
        ) + expected["chosen"]
        expected_rejected_tokens = expected["prompt"] + expected["rejected"]
        expected_rejected_labels = [CROSS_ENTROPY_IGNORE_IDX] * len(
            expected["prompt"]
        ) + expected["rejected"]

        ds = PreferenceDataset(
            source="iam/agoofy/goober",
            message_transform=ToDummyPreferenceMessages(),
            tokenizer=DummyTokenizer(),
        )
        assert len(ds) == 1
        mock_load_dataset.assert_called_once()

        prompt, label = ds[0]["chosen_input_ids"], ds[0]["chosen_labels"]
        assert prompt == expected_chosen_tokens
        assert label == expected_chosen_labels

        prompt, label = ds[0]["rejected_input_ids"], ds[0]["rejected_labels"]
        assert prompt == expected_rejected_tokens
        assert label == expected_rejected_labels

    def test_load_local_json(self):
        expected_tokenized_chosen_prompts = [
            [0, 4, 2, 1, 2, 4, 1, 4, 1, 4, 2, 2, 9, 3, 3, 5, -1]
        ]
        expected_tokenized_rejected_prompts = [
            [0, 4, 2, 1, 2, 4, 1, 4, 1, 4, 2, 2, 9, 4, 4, 4, -1]
        ]

        # prompt length is number of tokens shared between
        # the tokenized rejected and chosen messages
        prompt_length = 13
        expected_chosen_labels = [
            [CROSS_ENTROPY_IGNORE_IDX] * prompt_length + [3, 3, 5, -1]
        ]
        expected_rejected_labels = [
            [CROSS_ENTROPY_IGNORE_IDX] * prompt_length + [4, 4, 4, -1]
        ]

        ds = preference_dataset(
            tokenizer=DummyTokenizer(),
            source="json",
            data_files=str(ASSETS / "hh_rlhf_tiny.json"),
            train_on_input=False,
            split="train",
        )

        assert len(ds) == 1

        expected_keys = [
            "chosen_input_ids",
            "chosen_labels",
            "rejected_input_ids",
            "rejected_labels",
        ]
        assert set(ds[0].keys()) == set(expected_keys)
        assert len(ds[0].keys()) == 4

        assert expected_tokenized_chosen_prompts[0] == ds[0]["chosen_input_ids"]
        assert expected_tokenized_rejected_prompts[0] == ds[0]["rejected_input_ids"]

        assert expected_chosen_labels[0] == ds[0]["chosen_labels"]
        assert expected_rejected_labels[0] == ds[0]["rejected_labels"]

    def test_dataset_fails_with_packed(self):
        with pytest.raises(
            ValueError,
            match="Packed is currently not supported for preference datasets.",
        ):
            preference_dataset(
                tokenizer=DummyTokenizer(),
                source="json",
                data_files=str(ASSETS / "hh_rlhf_tiny.json"),
                train_on_input=False,
                split="train",
                packed=True,
            )

class TestCustomPreferenceDataset:
    @pytest.fixture
    def base_dataset(self):
        # Mocking a base PreferenceDataset
        mock_dataset = mock.MagicMock(spec=PreferenceDataset)
        mock_dataset.__len__.return_value = 1
        mock_dataset.__getitem__.return_value = {
            "chosen_input_ids": [0, 1, 2],
            "chosen_labels": [0, 1, 2],
            "rejected_input_ids": [3, 4, 5],
            "rejected_labels": [3, 4, 5],
        }
        return mock_dataset

    def test_custom_preference_dataset(self, base_dataset):
        custom_dataset = CustomPreferenceDataset(base_dataset=base_dataset)

        assert len(custom_dataset) == 1
        expected_keys = [
            "chosen_input_ids",
            "chosen_labels",
            "rejected_input_ids",
            "rejected_labels",
        ]
        assert set(custom_dataset[0].keys()) == set(expected_keys)
        assert len(custom_dataset[0].keys()) == 4

    def test_add_reference_log_probs(self, base_dataset):
        custom_dataset = CustomPreferenceDataset(base_dataset=base_dataset)

        ref_chosen_logps = torch.tensor([0.1, 0.2, 0.3])
        ref_rejected_logps = torch.tensor([0.4, 0.5, 0.6])

        custom_dataset.add_reference_log_probs(ref_chosen_logps, ref_rejected_logps)

        expected_keys = [
            "chosen_input_ids",
            "chosen_labels",
            "rejected_input_ids",
            "rejected_labels",
            "ref_chosen_logps",
            "ref_rejected_logps",
        ]
        assert set(custom_dataset[0].keys()) == set(expected_keys)
        assert len(custom_dataset[0].keys()) == 6

        for i in range(len(custom_dataset)):
            item = custom_dataset[i]
            assert torch.equal(item['ref_chosen_logps'], ref_chosen_logps[i])
            assert torch.equal(item['ref_rejected_logps'], ref_rejected_logps[i]) 
