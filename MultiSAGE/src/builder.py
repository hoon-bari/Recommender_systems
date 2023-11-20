import pandas as pd
import dgl
import torch

class PandasGraphBuilder(object):
    def __init__(self):
        self.entity_pk_to_name = {}
        self.entity_pk = {}
        self.entity_key_map = {}
        self.num_nodes_per_type = {}
        self.edges_per_relation = {}
        self.relation_name_to_etype = {}
        self.relation_src_key = {}
        self.relation_dst_key = {}
        self.entity_features = {}

    def convert_to_tensor(self, dataframe):
        tensors = {}
        for column, dtype in dataframe.dtypes.items():
            if pd.api.types.is_integer_dtype(dtype):
                # 정수형 텐서를 1차원으로 유지
                tensors[column] = torch.tensor(dataframe[column].values, dtype=torch.long)
            elif pd.api.types.is_float_dtype(dtype):
                # 실수형 텐서는 2차원으로 유지
                tensors[column] = torch.tensor(dataframe[column].values, dtype=torch.float).unsqueeze(1)
        return tensors

    def add_entities(self, entity_table, primary_key, name, features=None):
        entities = entity_table[primary_key].astype('category')
        entities = entities.cat.reorder_categories(entity_table[primary_key].values)
        self.entity_pk_to_name[primary_key] = name
        self.entity_pk[name] = primary_key
        self.num_nodes_per_type[name] = entity_table.shape[0]
        self.entity_key_map[name] = entities
        if features is not None:
            tensor_features = self.convert_to_tensor(features)
            self.entity_features[name] = tensor_features

    def add_binary_relations(self, relation_table, source_key, destination_key, name):
        src = relation_table[source_key].astype('category')
        src = src.cat.set_categories(self.entity_key_map[self.entity_pk_to_name[source_key]].cat.categories)
        dst = relation_table[destination_key].astype('category')
        dst = dst.cat.set_categories(self.entity_key_map[self.entity_pk_to_name[destination_key]].cat.categories)
        srctype = self.entity_pk_to_name[source_key]
        dsttype = self.entity_pk_to_name[destination_key]
        etype = (srctype, name, dsttype)
        self.relation_name_to_etype[name] = etype
        self.edges_per_relation[etype] = (src.cat.codes.values.astype('int64'), dst.cat.codes.values.astype('int64'))
        self.relation_src_key[name] = source_key
        self.relation_dst_key[name] = destination_key

    def build(self):
        graph = dgl.heterograph(self.edges_per_relation, self.num_nodes_per_type)
        for entity_type, feature_tensors in self.entity_features.items():
            # 각 특성을 개별적으로 그래프에 추가
            for feature_name, tensor in feature_tensors.items():
                graph.nodes[entity_type].data[feature_name] = tensor
        return graph





