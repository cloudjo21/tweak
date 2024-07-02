import argparse
import os
import pathlib
import time

from collections import defaultdict

import statistics

from tunip.constants import PRETRAIN_SENT_SEPARATOR
from tunip.corpus.corpus_path_provider import CorpusPathProviderFactory


def list_filepaths(input_dir):
    path_provider = CorpusPathProviderFactory.create(
        corpus_type="dir",
        input_dir=input_dir,
        extensions=[".txt"],
        split_path=""
    )
    return path_provider()


class RawCorpusSharder:

    def __init__(self, shard_size, output_dir, input_files):

        self.output_dir = output_dir

        self.input_files = input_files
        self.fraction_test_set = 0.1

        # self.n_shards = n_shards
        # os.path.getsize(path)
        total_raw_txt_size = 0
        for input_file in input_files:
            total_raw_txt_size += os.path.getsize(input_file)

        if total_raw_txt_size % shard_size > 0:
            self.n_shards = total_raw_txt_size // shard_size + 1
        else:
            self.n_shards = total_raw_txt_size // shard_size
        self.n_test_shards = max(self.n_shards * self.fraction_test_set, 1)
        
        self.articles = {}
        self.sentences = {}

        self.output_name_prefix = "ko_corpus"
        self.output_training_identifier = '_training'
        self.output_test_identifier = '_test'
        self.output_file_extension = '.txt'
        self.output_training_files = {}
        self.output_test_files = {}

        self.init_output_files()
        
        print(f"NUM OF SHARDS: {self.n_shards}")


    def init_output_files(self):
        print("START TO INIT  OUTPUT FILES")
        assert len(self.output_training_files) == 0, 'Internal storage self.output_files already contains data. This function is intended to be used by the constructor only.'
        assert len(self.output_test_files) == 0, 'Internal storage self.output_files already contains data. This function is intended to be used by the constructor only.'

        for i in range(self.n_shards):
            name = self.output_name_prefix + self.output_training_identifier + '_' + str(i) + self.output_file_extension
            self.output_training_files[name] = []

        for i in range(self.n_shards):
            name = self.output_name_prefix + self.output_test_identifier + '_' + str(i) + self.output_file_extension
            self.output_test_files[name] = []

        print("END TO INIT  OUTPUT FILES")

    def load_docs(self):
        print('START TO Loading Documents')

        global_article_count = 0
        for input_file in self.input_files:
            print('input file:', input_file)
            with open(input_file, mode='r', newline='\n') as f:
                for i, line in enumerate(f):
                    if line.strip():
                        self.articles[global_article_count] = line.rstrip()
                        global_article_count += 1

        print('END TO Loading Articles: There are', len(self.articles), 'articles.') 


    def segment_docs(self):
        print('START TO Segmentation', flush=True)
        if len(self.articles) < 1:
            self.load_docs()

        for i, article in enumerate(self.articles):
            self.sentences[i] = list(filter(lambda s: s and (len(s) > 1), [sent.strip() for sent in self.articles[article].split(PRETRAIN_SENT_SEPARATOR)]))

            if i % 5000 == 0:
                print(f'Segmenting document {i}')

        print('END TO Segmentation')


    def shard(self):
        print(f"{self.n_shards} number of shards would be built ...", flush=True)

        # Create dictionary with - key: sentence count per article, value: article id number
        sentence_counts = defaultdict(lambda: [])

        min_tokens_of_sent = 2

        max_sentences = 0
        total_sentences = 0  # total number of tokens for all the articles

        article_ids_remove = []
        for article_id in self.sentences:
            current_length = len(self.sentences[article_id])
            if current_length >= min_tokens_of_sent:
                sentence_counts[current_length].append(article_id)
                max_sentences = max(max_sentences, current_length)
                total_sentences += current_length
            else:
                article_ids_remove.append(article_id)
        
        for id_ in article_ids_remove:
            del self.sentences[id_]

        n_sentences_assigned_to_training = int((1 - self.fraction_test_set) * total_sentences)

        nominal_sentences_per_training_shard = n_sentences_assigned_to_training // self.n_shards
        nominal_sentences_per_test_shard = (total_sentences - n_sentences_assigned_to_training) // self.n_test_shards

        consumed_article_set = set({})
        list(map(self.articles.pop, article_ids_remove))
        unused_article_set = set(self.articles.keys())

        # Make first pass and add one article worth of lines per file
        for filename in self.output_training_files:
            print(f"Prepare to article id and num of sents for {filename}", flush=True)
            try:
                current_article_id = sentence_counts[max_sentences][-1]
            except:
                print(f"[EXCEPTION]: {filename}, max_sentences: {max_sentences}")
                exit(0)
            sentence_counts[max_sentences].pop(-1)
            self.output_training_files[filename].append(current_article_id)
            consumed_article_set.add(current_article_id)
            unused_article_set.remove(current_article_id)

            # Maintain the max sentence count
            while len(sentence_counts[max_sentences]) == 0 and max_sentences > min_tokens_of_sent:
                max_sentences -= 1

            if len(self.sentences[current_article_id]) > nominal_sentences_per_training_shard:
                nominal_sentences_per_training_shard = len(self.sentences[current_article_id])
                print('Warning: A single article contains more than the nominal number of sentences per training shard.')

        for filename in self.output_test_files:
            print(f"Prepare to article id and num of sents for {filename}", flush=True)
            try:
                current_article_id = sentence_counts[max_sentences][-1]
            except:
                print(f"[EXCEPTION]: {filename}, max_sentences: {max_sentences}")
                exit(0)
            sentence_counts[max_sentences].pop(-1)
            self.output_test_files[filename].append(current_article_id)
            consumed_article_set.add(current_article_id)
            unused_article_set.remove(current_article_id)

            # Maintain the max sentence count
            while len(sentence_counts[max_sentences]) == 0 and max_sentences > min_tokens_of_sent:
                max_sentences -= 1

            if len(self.sentences[current_article_id]) > nominal_sentences_per_test_shard:
                nominal_sentences_per_test_shard = len(self.sentences[current_article_id])
                print('Warning: A single article contains more than the nominal number of sentences per test shard.')

        
        training_counts = []
        test_counts = []
        for shard in self.output_training_files:
            training_counts.append(self._num_of_sentences_for(self.output_training_files[shard]))
        for shard in self.output_test_files:
            test_counts.append(self._num_of_sentences_for(self.output_test_files[shard]))
        training_median = statistics.median(training_counts)
        test_median = statistics.median(test_counts)

        # Make subsequent passes over files to find articles to add without going over limit
        history_remaining = []
        n_history_remaining = 4

        start_time = time.time()
        while len(consumed_article_set) < len(self.articles):
            if len(consumed_article_set) % int(len(self.articles) / 100) == 0:
                print(f"{len(consumed_article_set)} number of articles are consumed ...", flush=True)

            for fidx, file in enumerate(self.output_training_files):
                nominal_next_article_size = min(nominal_sentences_per_training_shard - training_counts[fidx], max_sentences)

                # Maintain the max sentence count
                while len(sentence_counts[max_sentences]) == 0 and max_sentences > min_tokens_of_sent:
                    max_sentences -= 1

                while len(sentence_counts[nominal_next_article_size]) == 0 and nominal_next_article_size > 0:
                    nominal_next_article_size -= 1

                if nominal_next_article_size not in sentence_counts or nominal_next_article_size == 0 or training_counts[fidx] > training_median:
                    continue    # skip adding to this file, will come back later if no file can accept unused articles

                current_article_id = sentence_counts[nominal_next_article_size][-1]
                sentence_counts[nominal_next_article_size].pop(-1)

                self.output_training_files[file].append(current_article_id)
                consumed_article_set.add(current_article_id)
                unused_article_set.remove(current_article_id)

            for fidx, file in enumerate(self.output_test_files):
                nominal_next_article_size = min(nominal_sentences_per_test_shard - test_counts[fidx], max_sentences)

                # Maintain the max sentence count
                while len(sentence_counts[max_sentences]) == 0 and max_sentences > min_tokens_of_sent:
                    max_sentences -= 1

                while len(sentence_counts[nominal_next_article_size]) == 0 and nominal_next_article_size > 0:
                    nominal_next_article_size -= 1

                if nominal_next_article_size not in sentence_counts or nominal_next_article_size == 0 or test_counts[fidx] > test_median:
                    continue    # skip adding to this file, will come back later if no file can accept unused articles

                current_article_id = sentence_counts[nominal_next_article_size][-1]
                sentence_counts[nominal_next_article_size].pop(-1)

                self.output_test_files[file].append(current_article_id)
                consumed_article_set.add(current_article_id)
                unused_article_set.remove(current_article_id)


            # If unable to place articles a few times, bump up nominal sizes by fraction until articles get placed
            if len(history_remaining) == n_history_remaining:
                history_remaining.pop(0)
            history_remaining.append(len(unused_article_set))

            history_same = True
            for i in range(1, len(history_remaining)):
                history_same = history_same and (history_remaining[i-1] == history_remaining[i])

            if history_same:
                nominal_sentences_per_training_shard += 1

            training_counts = []
            test_counts = []
            for shard in self.output_training_files:
                training_counts.append(self._num_of_sentences_for(self.output_training_files[shard]))
            for shard in self.output_test_files:
                test_counts.append(self._num_of_sentences_for(self.output_test_files[shard]))

            training_median = statistics.median(training_counts)
            test_median = statistics.median(test_counts)

            if (time.time() - start_time) > 100:
                start_time = time.time()
                print('Distributing data over shards:', len(unused_article_set), 'articles remaining.', flush=True)
        
        if len(unused_article_set) != 0:
            print('Warning: Some articles did not make it into output files.')

        for shard in self.output_training_files:
            print(f"Training shard: {self._num_of_sentences_for(self.output_training_files[shard])}")


    def write(self):
        print('START TO Write Shards to Disk', flush=True)

        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        for shard in self.output_training_files:
            self._write_single_shard(shard, self.output_training_files[shard])

        for shard in self.output_test_files:
            self._write_single_shard(shard, self.output_test_files[shard])

        print('END TO Write Shards to Disk')


    def _write_single_shard(self, shard_name, shard):
        with open(f"{self.output_dir}/{shard_name}", mode='w', newline='\n') as f:
            for article_id in shard:
                for line in self.sentences[article_id]:
                    f.write(line + '\n')

                f.write('\n')  # Line break between articles

    def _num_of_sentences_for(self, shard):
        result = 0
        for article_id in shard:
            result += len(self.sentences[article_id])

        return result


def main(args):

    if args.corpus_type == "file":
        filepaths = args.file_paths
    elif args.corpus_type == "dir":
        filepaths = []
        for p in args.file_paths:
            pathlib.Path(args.file_paths[0]).mkdir(parents=True, exist_ok=True)
            filepaths.extend(list(list_filepaths(input_dir=p)))
    else:
        raise RunTimeError("Not supported args.corpus_type!!")

    # sharder = RawCorpusSharder(256*1024*1024, ["kowiki.articles.txt"])
    sharder = RawCorpusSharder(args.file_block_size, args.output_dir, filepaths)
    sharder.segment_docs()

    t = time.process_time()

    sharder.shard()

    print(f"It takes about {(time.process_time() - t)/60.} minutes.")
    print("END TO build sharding corpus.")

    sharder.write()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="corpus shard")

    parser.add_argument(
        '-fp',
        '--file_paths',
        action='append',
        required=True
    )
    parser.add_argument(
        '-dir',
        '--output_dir',
        type=str,
        required=True
    )
    parser.add_argument(
        '-sz',
        '--file_block_size',
        type=int,
        required=True,
        default=256*1024*1024
    ) 
    parser.add_argument(
        '-ct',
        '--corpus_type',
        help='corpus path type (file/dir)',
        type=str,
        required=True,
        default='dir'
    )
    main(parser.parse_args())
