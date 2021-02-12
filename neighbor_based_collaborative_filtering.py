import re
import time
from tqdm import tqdm
from itertools import chain, islice

import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.svm import LinearSVC

from concurrent.futures import ThreadPoolExecutor
import concurrent.futures

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
        self.svc = {}

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

    def get_raw_id(self, items):
        return [iid if iid < self.num_songs else self.id2tag[iid] for iid in items]

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

    def get_norm_inversed(self, power=1):
        return csr_matrix(np.power(self.norm, -power))

    def get_base_embedding(self, uid):
        return self.user_item_matrix.dot(self.user_item_matrix[uid].T).T

    def get_item_embeddings(self, uid, alpha=0.5):
        base_embedding = self.get_base_embedding(uid)
        item_embeddings = base_embedding.multiply(self.item_user_matrix)  # elementwise
        return self.get_norm_inversed(power=alpha).multiply(item_embeddings)  # elementwise

    def get_rating_matrix(self, uid, alpha=0.5, beta=0.5):

        # 1 ) uid에 대한 item embedding 생성
        item_embeddings = self.get_item_embeddings(uid, alpha=alpha)

        # 2 ) uid가 소비한 item만 embedding 추출해놓기
        preferred_iids = self._get_inner_id(self.corpus[uid])
        preferred_item_embeddings = item_embeddings[preferred_iids]

        # 3 ) dot product로 <s{i},s{j}> 계산
        simliarity_nominator = item_embeddings.dot(preferred_item_embeddings.T)

        # 4 ) beta에 맞는 norm 계산
        norm_per_user = np.power(self.item_user_matrix.sum(axis=1), beta)
        norm_preferred = np.power(self.item_user_matrix[preferred_iids].sum(axis=1), 1 - beta)
        simliarity_denominator = norm_per_user * norm_preferred.T.astype('float')
        simliarity_denominator[simliarity_denominator == 0] = np.inf

        # 5 ) rating matrix 및 최종 r{j} 계산
        rating_matrix = simliarity_nominator / simliarity_denominator
        rating_per_item = rating_matrix.sum(axis=1)

        # 6 ) SVC 적합을 위한 label
        labels = np.zeros(self.num_songs + self.num_tags)
        labels[preferred_iids] = 1

        return rating_per_item, labels

    def _train_given_user(self, uid, alpha=0.5, beta=0.5, regularization=0.5,
                          dual=True, tolerance=1e-6, class_weight={0: 1, 1: 1}, max_iter=360000):

        """
        Train Support Vector Classifier (single user)
        """

        rating, labels = self.get_rating_matrix(uid, alpha=alpha, beta=beta)
        classifier = LinearSVC(C=regularization, dual=dual, tol=tolerance, class_weight=class_weight, max_iter=max_iter)
        classifier.fit(rating, labels)

        predictions = (-1) * classifier.decision_function(rating)
        self.svc[uid] = {'model': classifier, 'predictions': predictions}

    def trainSVC(self, alpha=0.5, beta=0.5, regularization=0.5,
                 dual=True, tolerance=1e-6, class_weight={0: 1, 1: 1}, max_iter=360000):

        """
        Train Support Vector Classifier (all user)
        """

        print('Train SVC ...')
        start = time.time()
        with ThreadPoolExecutor() as exe:
            results = [exe.submit(self._train_given_user, uid
                                  , alpha, beta, regularization, dual, tolerance
                                  , class_weight, max_iter) for uid in self.corpus.keys()]

            for Future_obj in concurrent.futures.as_completed(results):
                Future_obj.result()
                if len(self.svc) % 50 == 0:
                    print(f'\tFit {len(self.svc)}-th model ... {(time.time() - start) // 60} min')

        print(f'Completed : {(time.time() - start) // 60} min')

    def recommend(self, uid):
        recommend = np.argpartition(self.svc[uid]['predictions'], 1000)
        already_seen = self.user_item_matrix[uid].toarray()[0]

        rec_songs = list(islice((iid for iid in recommend if not already_seen[iid] and iid < self.num_songs), 100))
        rec_tags = self.get_raw_id(
            islice((iid for iid in recommend if not already_seen[iid] and iid >= self.num_songs), 10))

        return rec_songs, rec_tags

if __name__ == '__main__':
    model = NeighborBasedCollaborativeFiltering(train_path, val_que_path)
    model.build_csr_matrix()

    model.trainSVC()
    songs, tags = model.recommend(uid = 147668)