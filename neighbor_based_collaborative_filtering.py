import re
from itertools import chain

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from tqdm import tqdm

from Title_Based_Playlist_Generator import TagExtractor

class NeighborBasedCollaborativeFiltering:
    def __init__(self, train_path, val_path, limit=2):
        """
        Neighbor based CF and Discriminative Re-weighting/Re-ranking
        Paper : Automatic Music Playlist Continuation via Neighbor based CF and Discriminative Reweighting/Reranking
        """
        self.train = load_json(train_path)
        self.val = load_json(val_path)
        self.data = load_json(train_path) + load_json(val_path)

        print('Build Vocab ...')
        self.build_vocab(limit)

    def build_vocab(self, limit):
        self.filter = TagExtractor(limit)
        self.filter.build_by_vocab(set(chain.from_iterable([ply['tags'] for ply in self.data])))

        self.corpus = {}

        for ply in tqdm(self.data):
            raw_title = re.findall('[0-9a-zA-Z가-힣]+', ply['plylst_title'])
            extracted_tags = self.filter.convert(" ".join(raw_title + ply['tags']))
            ply['tags'] = list(set(extracted_tags))
            self.corpus[ply['id']] = ply['songs'] + ply['tags']

        self.songs = set(chain.from_iterable(ply['songs'] for ply in self.data))
        self.tags = set(chain.from_iterable(ply['tags'] for ply in self.data))
        self.num_songs = max(self.songs) + 1  # != len(self.songs)
        self.num_tags = len(self.tags)

        print("> Corpus :", len(self.corpus))
        print(f'> Songs + Tags = {len(self.songs)} + {len(self.tags)} = {len(self.songs) + len(self.tags)}')

    def _get_tag2id(self):
        self.tag2id = {tag: i + self.num_songs for i, tag in enumerate(self.tags)}
        self.id2tag = {i: tag for tag, i in self.tag2id.items()}

    def _get_inner_id(self, items):
        """
        id list for indexing item_user_matrix
        """
        return [item if isinstance(item, int) else self.tag2id[item] for item in items]

    def build_neighbor_based_embedding(self, alpha=2):
        self._get_tag2id()

        rows = []
        columns = []

        for uid, items in self.corpus.items():
            rows.extend([uid] * len(items))
            columns.extend(self._get_inner_id(items))

        scores = [1] * len(rows)

        print('Build user-item matrix ...')
        self.user_item_matrix = csr_matrix((scores, (rows, columns)),
                                           shape=(max(rows) + 1, self.num_songs + self.num_tags))
        self.item_user_matrix = self.user_item_matrix.T

        norm = np.array(self.user_item_matrix.sum(axis=1).T, dtype=np.float32)
        norm[norm == 0.] = np.inf
        self.norm_inverse = csr_matrix(1 / np.power(norm, alpha))

    def get_base_embedding(self, uid):
        return self.user_item_matrix.dot(self.user_item_matrix[uid].T).T

    def get_song_embedding(self, uid, alpha=2):
        base_embedding = self.get_base_embedding(uid)
        song_embeddings = base_embedding.multiply(self.item_user_matrix)  # elementwise
        return self.norm_inverse.multiply(song_embeddings)