import argparse
import glob
import faiss
import torch

emb_dimension = 123


def load_into_faiss(embedding_list, with_gpu):
    """
    Load embeddings into faiss index, handle GPU if needed
    :param embedding_list: list of embeddings files to load
    :param with_gpu:
    :return:
    """
    if with_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, emb_dimension)
    else:
        index = faiss.IndexFlatIP(emb_dimension)
    for embedding_file in embedding_list:
        try:
            embedding = torch.load(embedding_file)['mean_representations'][36]
            index.add(embedding)
        except:
            print(f'{embedding_file} pt files does not have mean_representations')
    return index


def search_against_index(query_embedding_list, index, k, with_gpu):
    """
    Search queries against index and return scores
    :param query_embedding_list: list of query embeddings
    :param index: faiss index
    :param k: number of nearest neighbors to return
    :param with_gpu:
    :return:
    """
    search_results = {}
    # TODO Search chuncks in a way that fits in GPU RAM
    for query_embedding_file in query_embedding_list:
        try:
            query_embedding = torch.load(query_embedding_file)['mean_representations'][36]
            if with_gpu:
                query_embedding = query_embedding.cuda()
            scores, matches = index.search(query_embedding, k)
            search_results[query_embedding_file] = zip(scores[0], matches[0])
        except:
            print(f'{query_embedding_file} pt files does not have mean_representations')

    return search_results


if __name__ == '__main__':

    # parse command line folder where embeddings are stored
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_embeddings_folder', type=str, default='embeddings')
    parser.add_argument('--query_embeddings_folder', type=str, default='embeddings')
    parser.add_argument('--with_gpu', type=bool, default=False)

    args = parser.parse_args()
    db_embeddings_folder = args.db_embeddings_folder
    query_embeddings_folder = args.query_embeddings_folder
    with_gpu = args.with_gpu

    embedding_list = glob.glob(db_embeddings_folder + '/*.pt')
    index = load_into_faiss(embedding_list, with_gpu)
    query_embedding_list = glob.glob(query_embeddings_folder + '/*.pt')
    search_results = search_against_index(query_embedding_list, index, with_gpu)
    for query_embedding_file, results in search_results.items():
        for score, match in results:
            print(query_embedding_file, score, match)
