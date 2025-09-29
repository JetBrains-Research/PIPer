# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the EnvBench dataset to verl format
"""

import argparse
import os

import datasets

from verl.utils.hdfs_io import copy, makedirs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None)
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--config", type=str, default="default")

    args = parser.parse_args()

    data_source = "JetBrains-Research/envbench-zeroshot-rl"

    train_dataset = datasets.load_dataset(data_source, args.config, split="train")
    test_dataset = datasets.load_dataset(data_source, args.config, split="test")

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            prompt = example.pop("messages")
            repository = example.pop("repository")
            revision = example.pop("revision")
            data = {
                "data_source": data_source,
                "prompt": prompt,
                "ability": "code",
                "reward_model": {
                    "style": "rule",
                    # veRL code expects some ground truth
                    "ground_truth": None,
                },
                "extra_info": {"split": split, "index": idx, "repository": repository, "revision": revision, "initial_prompt": prompt, "tools_kwargs": {"repository": repository, "revision": revision}},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    makedirs(local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir, exist_ok=True)

        copy(src=local_dir, dst=hdfs_dir)
