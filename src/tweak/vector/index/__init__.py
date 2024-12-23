import faiss
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from tunip.file_utils import services as file_services
from tunip.np_op_utils import normalized_embeddings
from tunip.path.mart.vector import (
    MartVectorIndexDocumentDid2vidPath,
    MartVectorIndexDocumentSourceIndexTypeSnapshotPath,
    MartVectorIndexDocumentSourceIndexTypePath,
    MartVectorIndexDocumentVid2didPath
)
from tunip.path_utils import services as path_services
from tunip.snapshot_utils import SnapshotCleaner, snapshot_now

from tweak.vector import VectorLoader


class VectorIndexBuilder:

    def __init__(self, service_config, source_type, d_size, index_type="flat_inner_product", upload_service=True):
        self.service_config = service_config
        self.source_type = source_type
        self.d_size = d_size
        self.index_type = index_type
        self.upload_service = upload_service

    def __call__(self, latest_vectors_root_path: str, id_field_name: str):
        """ download vectors(parquet files), transform vectors to arrow files, build indexes with arrow files, and upload them
        /user/[username]/mart/vector_index/document/[source_type]/[index_type]/vector-[i].index
        /user/[username]/mart/vector_index/document/[source_type]/[index_type]/did2vid/item2vec_id.pkl
        /user/[username]/mart/vector_index/document/[source_type]/[index_type]/vid2did/vec2item_id.pkl

        Args:
            latest_vectors_root_path (str): /user/[username]/warehouse/vectors/[task_name]/[phase_type]
        """
        vectors_path = f"{latest_vectors_root_path}/vectors"

        if self.service_config.has_dfs:
            file_service = file_services(self.service_config)
            file_service.download(vectors_path)
        else:
            file_service = None

        local_file_service = file_services.get("LOCAL", config=self.service_config.config)
        local_path_service = path_services.get("LOCAL", config=self.service_config.config)

        # build vector arrow
        arrows_path = f"{latest_vectors_root_path}/arrow"
        local_file_service.mkdirs(arrows_path)

        expr_valid_user_account_sid = pc.field(id_field_name) != ""
        pq_filepaths = [fpath for fpath in local_file_service.list_dir(vectors_path) if fpath.endswith(".parquet")]
        for i, fpath in enumerate(pq_filepaths):
            arrow_table = pq.read_table(fpath).filter(expr_valid_user_account_sid)
            sink = pa.OSFile(local_path_service.build(f"{arrows_path}/vectors-{i}.arrow"), "wb")
            with pa.RecordBatchFileWriter(sink, arrow_table.schema) as writer:
                writer.write_table(arrow_table)

        # build vector-index and id-mapping
        raw_index = faiss.IndexFlatIP(self.d_size)
        index = faiss.IndexIDMap2(raw_index)
        i2v_dict = dict()
        v2i_dict = dict()

        start_vid = 0
        for fpath in local_file_service.list_dir(arrows_path):
            x = VectorLoader(f"{fpath}")()
            ids = x[0].to_pylist()

            vectors_stacked = np.vstack(x[1])  # vectors
            xb = normalized_embeddings(vectors_stacked)
            index.train(xb)
            # index.add(xb)

            end_vid = start_vid + len(xb)
            vids = list(range(start_vid, end_vid))

            assert len(ids) == len(vids)

            index.add_with_ids(xb, np.array(vids))

            start_vid = end_vid

            i2v_dict.update(zip(ids, vids))
            v2i_dict.update(zip(vids, ids))

        snapshot_dt = snapshot_now()
        vector_index_dir_path = MartVectorIndexDocumentSourceIndexTypeSnapshotPath(
            user_name=self.service_config.username,
            source_type=self.source_type,
            index_type=self.index_type,
            snapshot_dt=snapshot_dt
        )
        vector_index_root_dir_path = MartVectorIndexDocumentSourceIndexTypePath(
            user_name=self.service_config.username,
            source_type=self.source_type,
            index_type=self.index_type
        )

        local_file_service.mkdirs(vector_index_dir_path)

        # write down vector.index
        vector_index_filepath = f"{vector_index_dir_path}/vector.index"
        faiss.write_index(index, local_path_service.build(vector_index_filepath))

        local_file_service.mkdirs(f"{vector_index_dir_path}/did2vid")
        local_file_service.mkdirs(f"{vector_index_dir_path}/vid2did")

        # write down id-mappings
        local_file_service.dumps_pickle(f"{vector_index_dir_path}/did2vid/item2vec_id.pkl", i2v_dict)
        local_file_service.dumps_pickle(f"{vector_index_dir_path}/vid2did/vec2item_id.pkl", v2i_dict)

        local_snapshot_cleaner = SnapshotCleaner(self.service_config, paths_left=3)
        local_snapshot_cleaner.clean(str(vector_index_root_dir_path), force_fs="LOCAL")

        if upload_service is True:
            if self.service_config.has_dfs and file_service:
                file_service.mkdirs(vector_index_dir_path)
                payload = local_file_service.load_binary(vector_index_filepath)
                file_service.write_binary(vector_index_filepath, payload)
    
                snapshot_cleaner = SnapshotCleaner(self.service_config, paths_left=3)
                snapshot_cleaner.clean(str(vector_index_root_dir_path))
    
            mart_did2vid_path = MartVectorIndexDocumentDid2vidPath(
                user_name=self.service_config.username,
                source_type=self.source_type,
                index_type=self.index_type,
                snapshot_dt=snapshot_dt
            )
            mart_vid2did_path = MartVectorIndexDocumentVid2didPath(
                user_name=self.service_config.username,
                source_type=self.source_type,
                index_type=self.index_type,
                snapshot_dt=snapshot_dt
            )
    
            file_service.mkdirs(str(mart_did2vid_path))
            file_service.write_binary(f"{str(mart_did2vid_path)}/item2vec_id.pkl", local_file_service.load_binary(f"{str(mart_did2vid_path)}/item2vec_id.pkl"))
            # file_service.copy_files(warehouse_did2vid_path, str(mart_did2vid_path))
    
            file_service.mkdirs(str(mart_vid2did_path))
            file_service.write_binary(f"{str(mart_vid2did_path)}/vec2item_id.pkl", local_file_service.load_binary(f"{str(mart_vid2did_path)}/vec2item_id.pkl"))
            # file_service.copy_files(warehouse_vid2did_path, str(mart_vid2did_path))
