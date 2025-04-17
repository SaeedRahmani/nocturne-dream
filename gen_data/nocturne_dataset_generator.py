"""
This script is to generate Nocture-based Waymo Motion Dataset.

@author: Zhenlin (Gavin) Xu
@date: March 31, 2025
"""

import os
import gc
import json
import zarr
import h5py
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from nocturne import Simulation
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch_geometric.data import HeteroData
from .nocturne_graph import NocturneScenarioGraph


class NocturneDatasetGenerator:
    def __init__(self, config: DictConfig):

        # load settings
        self.T = 90
        self.dt = 0.1
        self.image_dataset = config.image_dataset
        self.graph_dataset = config.graph_dataset
        self.scenario_config: dict = OmegaConf.to_container(
            config.load_scenario, resolve=True)
        self.load_directory = Path(config.load_directory)
        self.save_directory = Path(config.save_directory)
        self.show_progress_bar = not config.show_progress_bar

        self.mode: str = config.dataset_mode
        self.format: str = config.format
        assert self.format in [
            "h5", "npz", "zarr"], f"Invalid format name {self.format}"
        self.dataset_saved_name = f"{self.mode}.{self.format}"

        # concat paths
        self.train_dataset_path = self.load_directory / \
            Path("formatted_json_v2_no_tl_train")
        assert Path.exists(
            self.train_dataset_path), f"Raw dataset (train) path {self.train_dataset_path} is not valid"
        self.val_dataset_path = self.load_directory / \
            Path("formatted_json_v2_no_tl_valid")
        assert Path.exists(
            self.val_dataset_path), f"Raw dataset (validation) path {self.val_dataset_path} is not valid"

        # retrieve scenario filenames (json)
        self.train_jsonfiles = sorted([str(file) for file in self.train_dataset_path.rglob(
            'tfrecord*.json') if file.is_file()])
        self.val_jsonfiles = sorted([str(file) for file in self.val_dataset_path.rglob(
            'tfrecord*.json') if file.is_file()])

    def __len__(self):
        return len(self.train_jsonfiles) + len(self.val_jsonfiles)

    def gen_image_dataset(self, parallel: bool = False):
        """ Generate image dataset. """

        # Enable software rendering to avoid segmentation fault
        # Prevents the "cannot create thread specific key for __cxa_get_globals()" error
        os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'

        if parallel:
            self.gen_image_dataset_parallel()
        else:
            self.gen_image_dataset_sequential()

    def retrieve_image_sample(self, json_name):
        """
        Process one image-based sample, return obs and action arrays.
        """
        json_id, scenario_id = NocturneDatasetGenerator.parse_scenario_metadata(
            str(Path(json_name).name))
        sim = Simulation(scenario_path=json_name, config=self.scenario_config)
        scenario = sim.getScenario()
        ego_vehicle = scenario.getVehicles()[-1]

        for vehicle in scenario.getVehicles():
            vehicle.expert_control = True

        img_allframes, action_allframes = [], []
        for t in tqdm(range(self.T), desc="Frame", leave=False, disable=True):
            img = scenario.getImage(
                img_width=self.image_dataset.img_width,
                img_height=self.image_dataset.img_height,
                draw_target_positions=self.image_dataset.draw_target_positions,
                padding=50.0,
                source=ego_vehicle,
                view_width=120,
                view_height=120,
                rotate_with_source=self.image_dataset.rotate_with_source,
            )
            img_allframes.append(img)
            action = scenario.expert_action(ego_vehicle, t)
            action = np.array([action.acceleration, action.steering, action.head_angle])
            action_allframes.append(action)
            sim.step(self.dt)

        img_allframes = np.stack(img_allframes)
        action_allframes = np.stack(action_allframes)

        # Optional: return IDs as well for naming/indexing
        return img_allframes, action_allframes, json_id, scenario_id

    def gen_image_dataset_dask(self):
        import dask
        from dask import delayed, compute
        from dask.distributed import Client, progress, LocalCluster

        self.saved_path = self.save_directory / Path('image') / Path(self.format)
        self.saved_path.mkdir(parents=True, exist_ok=True)


        cluster = LocalCluster(
            n_workers=8,               # 启动的 worker 数
            threads_per_worker=1,      # 每个 worker 使用的线程数
            memory_limit='10GB'         # 每个 worker 限制的最大内存（字符串或 float）
        )
        client = Client(cluster)

        dataset_path = self.saved_path / Path(self.dataset_saved_name)
        zarr_root = zarr.open_group(str(dataset_path), mode='w')

        total_samples = len(self.train_jsonfiles) # + self.val_jsonfiles)
        obs_shape = (self.T, 256, 256, 4)
        action_shape = (self.T, 3)

        zarr_obs = zarr_root.create_dataset(
            "obs",
            shape=(total_samples, *obs_shape),
            chunks=(1, *obs_shape),
            dtype='uint8',  # 或 img_allframes.dtype
            compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=2)
        )
        zarr_action = zarr_root.create_dataset(
            "action",
            shape=(total_samples, *action_shape),
            chunks=(1, *action_shape),
            dtype='float32',
            compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=2)
        )

        def process_and_write(idx, json_name):
            zarr_root = zarr.open_group(str(dataset_path), mode='a')
            img_allframes, action_allframes, json_id, scenario_id = self.retrieve_image_sample(json_name)
            zarr_root["obs"][idx] = img_allframes
            zarr_root["action"][idx] = action_allframes
            return f"{json_id}_{scenario_id}"

        from dask.diagnostics import ProgressBar

        futures = [
            dask.delayed(process_and_write)(i, json_name)
            for i, json_name in enumerate(self.train_jsonfiles)
        ]

        future_result = client.compute(futures)
        progress(future_result)
        results = client.gather(future_result)

        # futures = [
        #     dask.delayed(process_and_write)(i, json_name)
        #     for i, json_name in enumerate(self.train_jsonfiles)
        # ]
        # progress(futures)  # 可视化进度条
        # dask.compute(*futures)

        # zarr.consolidate_metadata(str(self.saved_path / Path(self.dataset_saved_name)))
        # print(f"已使用 Dask+Zarr 并行完成保存，数据目录为：{self.saved_path}")

    def gen_image_dataset_parallel(self):
        """
        Generate image-based dataset in sequence.
        """
        self.saved_path = self.save_directory / \
            Path('image') / Path(self.format)
        self.saved_path.mkdir(parents=True, exist_ok=True)

        if self.format == "h5":
            raise RuntimeError(
                "h5py does not support multi-processing. Consider using Zarr instead.")
        self.show_progress_bar = True
        # Adjust max_workers as per your CPU cores
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(self.process_single_image_sample, json_name)
                for json_name in self.train_jsonfiles[:15] + self.val_jsonfiles[:15]
            ]

            # Show progress bar
            for future in tqdm(as_completed(futures), total=len(futures), desc="IMAGE:train"):
                future.result()  # This will raise exceptions if any occurred in the process
    
        if self.format == "zarr":
            zarr.consolidate_metadata(str(self.saved_path / Path(self.dataset_saved_name)))
 
        print(f"Save IMAGE dataset into {self.saved_path}")

    def gen_image_dataset_sequential(self):
        """
        Generate image-based dataset in sequence.
        """

        self.saved_path = self.save_directory / \
            Path('image') / Path(self.format)
        self.saved_path.mkdir(parents=True, exist_ok=True)

        for json_name in tqdm(self.train_jsonfiles[:15], desc="IMAGE:train", disable=self.show_progress_bar):
            self.process_single_image_sample(json_name)
        for json_name in tqdm(self.val_jsonfiles[:15], desc="IMAGE:valid", disable=self.show_progress_bar):
            self.process_single_image_sample(json_name)

        print(f"Save IMAGE dataset into {self.saved_path}")

    def process_single_image_sample(self, json_name):
        """
        Process one image-based sample.
        """
        # retrieve metadata: json id, scene id
        json_id, scenario_id = NocturneDatasetGenerator.parse_scenario_metadata(
            str(Path(json_name).name))
        sim = Simulation(scenario_path=json_name,
                         config=self.scenario_config)
        scenario = sim.getScenario()

        ego_vehicle = scenario.getVehicles()[-1]

        # make sure all vehicles replay the waymo dataset.
        for vehicle in scenario.getVehicles():
            vehicle.expert_control = True

        img_allframes, action_allframes = [], []
        for t in tqdm(range(self.T), desc="Frame", leave=False, disable=self.show_progress_bar):

            ego_vehicle = scenario.getVehicles()[-1]

            # retrieve obs
            img = scenario.getImage(
                img_width=self.image_dataset.img_width,
                img_height=self.image_dataset.img_height,
                draw_target_positions=self.image_dataset.draw_target_positions,
                padding=50.0,
                source=ego_vehicle,
                view_width=120,
                view_height=120,
                rotate_with_source=self.image_dataset.rotate_with_source,
            )
            img_allframes.append(img)

            # retrieve action
            action = scenario.expert_action(ego_vehicle, t)
            action = np.array([
                action.acceleration, action.steering, action.head_angle
            ])
            action_allframes.append(action)

            sim.step(self.dt)

        img_allframes = np.stack(img_allframes)
        action_allframes = np.stack(action_allframes)

        if self.format == "npz":
            np.savez(
                file=f"{str(self.saved_path / Path(f'{json_id}_{scenario_id}.npz'))}",
                # shape: (T=90, H, W, C=4)
                obs=torch.tensor(img_allframes),
                # shape: (T=90, A=3)
                action=torch.tensor(action_allframes),
            )
        elif self.format == "h5":
            with h5py.File(str(self.saved_path / Path(self.dataset_saved_name)), 'a') as hf:
                grp_name = f"{json_id}_{scenario_id}"

                if grp_name in hf:
                    # print(f"Warning: Overwriting existing dataset {grp_name}")
                    del hf[grp_name]

                grp = hf.create_group(grp_name)
                grp.create_dataset("obs", data=img_allframes,
                                   compression="gzip", chunks=True)
                grp.create_dataset("action", data=action_allframes,
                                   compression="gzip", chunks=True)
        elif self.format == "zarr":
           
            root = zarr.open_group(
                str(self.saved_path / Path(self.dataset_saved_name)), 
                mode='a',
                # consolidate_on_close=True,
            )

            grp_name = f"{json_id}_{scenario_id}"
            if grp_name in root:
                # print(f"Warning: Overwriting existing dataset {grp_name}")
                del root[grp_name]

            grp = root.create_group(grp_name)

            # 存储 obs (T=90, H, W, C=4)
            grp.create_dataset(
                "obs",
                data=img_allframes,
                chunks=(10, *img_allframes.shape[1:]),  # 以10帧为单位分块
                dtype=img_allframes.dtype,
                compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=2)
            )

            # 存储 action (T=90, A=3)
            grp.create_dataset(
                "action",
                data=action_allframes,
                chunks=(10, action_allframes.shape[1]),  # 以10帧为单位分块
                dtype=action_allframes.dtype,
                compressor=zarr.Blosc(cname="zstd", clevel=5, shuffle=2)
            )

            # 额外存储 json_id 和 scenario_id 作为属性
            grp.attrs["json_id"] = json_id
            grp.attrs["scenario_id"] = scenario_id
            # zarr.consolidate_metadata(str(self.saved_path / Path(self.dataset_saved_name)))

        del img
        del scenario
        del sim
        gc.collect()

    def gen_graph_dataset(self):
        """ Generate graph dataset. """

        saved_path = self.save_directory / Path('graph')
        saved_ids = list()

        # For GNN-based world model.
        for json_name in tqdm(self.train_jsonfiles[:6], desc="Graph:train", disable=self.show_progress_bar):

            scenario_graphs = []

            # retrieve metadata: json id, scene id
            json_id, scenario_id = NocturneDatasetGenerator.parse_scenario_metadata(
                str(Path(json_name).name))

            assert type(json_id) == str and len(
                json_id) == 5, f"{json_id} is not valid"
            assert type(scenario_id) == str
            saved_ids.append([json_id, scenario_id])

            sim = Simulation(scenario_path=json_name,
                             config=self.scenario_config)
            scenario = sim.getScenario()

            # make sure all vehicles replay the waymo dataset
            for vehicle in scenario.getVehicles():
                vehicle.expert_control = True

            # create a graph per frame
            for t in tqdm(range(self.T), desc="Frame", leave=False, disable=self.show_progress_bar):

                # get ego vehicle reference
                ego_vehicle = scenario.getVehicles()[-1]

                # create an heterogeneous graph object
                graph = HeteroData()

                # retrieve and build node: `vehicle`
                vehicle_node_vectors = []
                for veh in scenario.getVehicles():
                    vehicle_node_state = scenario.ego_state(veh)  # dim: (10,)
                    vehicle_node_vectors.append(vehicle_node_state)
                    # print(veh.id)
                graph["vehicle"].x = torch.Tensor(np.stack(
                    vehicle_node_vectors))  # dim: (N_Veh, 10)

                # retrieve and build node: `cyclist`
                cyclist_node_vectors = []
                for cyc in scenario.getCyclists():
                    cyclist_node_state = scenario.ego_state(cyc)
                    cyclist_node_vectors.append(cyclist_node_state)
                if len(cyclist_node_vectors) > 0:
                    cyclist_node_vectors = np.stack(cyclist_node_vectors)
                    graph["cyclist"].x = torch.Tensor(np.stack(
                        cyclist_node_vectors))

                # retrieve and build node: `pedestrian`
                pedestrian_node_vectors = []
                for ped in scenario.getCyclists():
                    pedestrian_node_state = scenario.ego_state(ped)
                    pedestrian_node_vectors.append(pedestrian_node_state)
                if len(pedestrian_node_vectors) > 0:
                    pedestrian_node_vectors = np.stack(pedestrian_node_vectors)
                    graph["pedestrian"].x = torch.Tensor(
                        np.stack(pedestrian_node_vectors))

                # # retrieve and build node: `roadline` # TODO: how to better represent the road network
                # roadlines = scenario.getRoadLines()

                # retrieve and build edge: `distance` # TODO: view angle and view region (partial observe or not?)
                src_indices, dst_indices = [], []
                edge_property_distance = []
                for idx, veh1 in enumerate(scenario.getVehicles()):
                    for jdx, veh2 in enumerate(scenario.getVehicles()):
                        if idx != jdx:
                            src_indices.append(idx)
                            dst_indices.append(jdx)
                            # TODO: calculate the distance between two veh
                            edge_property_distance.append([0, 1, 2])

                graph["vehicle", "to", "vehicle"].edge_index = torch.tensor(
                    [src_indices, dst_indices], dtype=torch.long)
                graph["vehicle", "to", "vehicle"].edge_attr = torch.tensor(
                    edge_property_distance)

                # retrieve and build label: `expert_action`
                expert_action = scenario.expert_action(ego_vehicle, t)
                graph.y = torch.tensor(np.array(
                    [expert_action.acceleration, expert_action.steering, expert_action.head_angle]))  # dim: (3,)
                assert graph.y.shape == (3,)
                # append to scenario graphs per frame
                scenario_graphs.append(graph)

                # simulation goes forward one step
                sim.step(self.dt)

            # save as a Dataset object
            dataset = NocturneScenarioGraph(
                root="./raw_data/graph", data_list=scenario_graphs, json_id=json_id, scene_id=scenario_id)
            # dataset.save(scenario_graphs, path="../raw_data/graph/")

            # clean unused objects
            del scenario
            del sim

        with open('./raw_data/graph/json_scene_id.json', 'w') as file:
            json.dump(saved_ids, file, indent=4)
        print(f"Save GRAPH dataset into {saved_path}")

    @classmethod
    def parse_scenario_metadata(self, json_name):
        # tfrecord-00136-of-00150_28.json
        parts = json_name.split('-')
        json_id = parts[1]
        scene_id = parts[-1].split('_')[1].split('.')[0]
        return (json_id, scene_id)
