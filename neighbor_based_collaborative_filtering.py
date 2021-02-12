import re
from tqdm import tqdm
from itertools import chain

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix

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

    def build_csr_matrix(self):
        """
        :param alpha: control the influence of long playlist , 0 <= alpha <= 1
        """

        print('Build user-item matrix ...')

        self._get_tag2id()

        rows = []
        columns = []

        for uid, items in self.corpus.items():
            rows.extend([uid] * len(items))
            columns.extend(self._get_inner_id(items))

        scores = [1] * len(rows)

        self.user_item_matrix = csr_matrix((scores, (rows, columns)),
                                           shape=(max(rows) + 1, self.num_songs + self.num_tags))
        self.item_user_matrix = self.user_item_matrix.T

        self.norm = np.array(self.item_user_matrix.sum(axis=0), dtype=np.float32)
        self.norm[self.norm == 0.] = np.inf

    def get_norm_inversed(self, power = 1):
        return csr_matrix(np.power(self.norm, -power))

    def get_base_embedding(self, uid):
        return self.user_item_matrix.dot(self.user_item_matrix[uid].T).T

    def get_item_embeddings(self, uid, alpha = 0.5):
        base_embedding = self.get_base_embedding(uid)
        item_embeddings = base_embedding.multiply(self.item_user_matrix)  # elementwise
        return self.get_norm_inversed(power = alpha).multiply(item_embeddings) # elementwise

    def build_rating_matrix(self, uid, alpha = 0.5, beta = 0.5):

        # 1 ) uid에 대한 item embedding 생성
        item_embeddings = self.get_item_embeddings(uid, alpha = alpha)


        # 2 ) uid가 소비한 item만 embedding 추출해놓기
        preferred_iids = self._get_inner_id(self.corpus[uid])
        preferred_item_embeddings = item_embeddings[preferred_iids]

        # 3 ) dot product로 <s{i},s{j}> 계산
        simliarity_nominator = item_embeddings.dot(preferred_item_embeddings.T)

        # 4 ) beta에 맞는 norm 계산
        norm_per_user = np.power(self.item_user_matrix.sum(axis = 1), beta)
        norm_preferred = np.power(self.item_user_matrix[preferred_iids].sum(axis = 1),1-beta)
        simliarity_denominator = norm_per_user*norm_preferred.T.astype('float')
        simliarity_denominator[simliarity_denominator == 0] = np.inf

        # 5 ) rating matrix 및 최종 r{j} 계산
        rating_matrix = simliarity_nominator/simliarity_denominator
        rating_per_item = rating_matrix.sum(axis = 1)

        return rating_per_item