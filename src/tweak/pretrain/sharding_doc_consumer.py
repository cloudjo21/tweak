from abc import ABC


class ShardingArticleConsumer:
    self __init__(self, n_total_tokens, fraction_test_set, input_files):
        """
        n_total_tokens: total number of tokens for all of the articles
        """
        self.n_total_tokens = n_total_tokens

        # self.n_training_shards = 

    def _get_n_training_shards(self, total_size, shard_size):
        if total_size %
    def _get_sizeof_article_files(self):
        total_raw_txt_size = 0
        for input_file in input_files:
            total_raw_txt_size += os.path.getsize(input_file)
        return total_raw_txt_size


    def ready_to_shard(self):
        pass


class TrainingShardingArticleConsumer(ShardingArticleConsumer):
    def __init__(self, n_total_tokens, fraction_test_set, n_training_shards):
        super().__init__(self, n_total_tokens, fraction_test_set)

        self.n_training_shards = n_training_shards

        n_tokens_to_training = int((1 - self.fraction_test_set) * n_total_tokens)
        self.n_nominal_tokens = n_tokens_to_training // self.n_training_shards

    def 

class TestShardingArticleConsumer(ShardingArticleConsumer):
    def __init__(self, n_total_tokens, fraction_test_set, n_test_shards):
        super().__init__(self, n_total_tokens, fraction_test_set)

        self.n_test_shards = n_test_shards

        n_tokens_to_training = int((1 - self.fraction_test_set) * n_total_tokens)
        self.n_nominal_tokens = (n_total_tokens - n_tokens_to_training) // self.n_test_shards
