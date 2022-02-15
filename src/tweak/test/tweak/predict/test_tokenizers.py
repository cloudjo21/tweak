import pathlib
import unittest

from transformers import AutoConfig, AutoTokenizer
from transformers.file_utils import is_remote_url, get_from_cache

from tunip.env import NAUTS_LOCAL_ROOT
from tunip.path_utils import services as path_services
from tunip.service_config import get_service_config

from tweak.predict.config import TokenizerConfig
from tweak.predict.resource_materialize import ResourceMaterializer


class TokenizersTest(unittest.TestCase):
    def test_resource_materialize(self):
        service_config = get_service_config(force_service_level='dev')

        config_json = """
            {
                "model_path": "/user/nauts/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator",
                "path": "/user/nauts/mart/tokenizers/monologg%2Fkoelectra-small-v3-discriminator",
                "max_length": 128
            }
        """

        config = TokenizerConfig.parse_raw(config_json)
        assert config
        assert config.max_length == 128

        ResourceMaterializer.apply_for_tokenizer(config, service_config)
        vocab_path = f"{NAUTS_LOCAL_ROOT}/user/nauts/mart/tokenizers/monologg%2Fkoelectra-small-v3-discriminator/vocab.txt"
        assert pathlib.Path(vocab_path).exists()


    # def test_load_from_hdfs(self):
    #     hdfs_vocab_path = '/user/nauts/mart/tokenizers/monologg%2Fkoelectra-small-v3-discriminator/vocab.txt'
    #     hdfs_config_path = '/user/nauts/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator/config.json'

    #     service_config = get_service_config(force_service_level='dev')
    #     path_service = path_services.get('HDFS', config=service_config.config)
    #     vocab_path = path_service.build_http(hdfs_vocab_path)
    #     config_path = path_service.build_http(hdfs_config_path)

    #     import urllib.parse
    #     # config_path = 'http://dev01.ascent.com:50070/user/nauts/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator'
    #     # config_path = 'http://dev01.ascent.com:50070/user/nauts/mart/plm/models/monologg%252Fkoelectra-small-v3-discriminator'
    #     # config_path = 'http://dev01.ascent.com:50070/user/nauts/mart/plm/models/monologg%25252Fkoelectra-small-v3-discriminator'
    #     # config_path = 'http://dev01.ascent.com:50070/user/nauts/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator/config.json'

    #     # config_path = 'http://dev01.ascent.com:50070/webhdfs/v1/user/nauts/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator/config.json?op=GETFILESTATUS'
    #     config_path = 'http://dev01.ascent.com:50070/webhdfs/v1/user/nauts/mart/plm/models/monologg%252Fkoelectra-small-v3-discriminator/config.json?op=GETFILESTATUS'

    #     # config_path = 'http://dev01.ascent.com:50070/webhdfs/v1/user/nauts/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator/config.json?op=OPEN'
    #     # config_path = 'http://dev01.ascent.com:50070/webhdfs/v1/user/nauts/mart/plm/models/monologg%252Fkoelectra-small-v3-discriminator/config.json?op=OPEN'
    #     # config_path = 'http://dev01.ascent.com:50070/webhdfs/v1/user/nauts/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator/config.json?op=OPEN'
    #     # config_path = 'http://dev01.ascent.com:50070/webhdfs/v1/user/nauts/mart/plm/models/monologg%2Fkoelectra-small-v3-discriminator/config.json'
    #     # config_path = f"http://dev01.ascent.com:50070/webhdfs/v1/user/nauts/mart/plm/models/{urllib.parse.quote('monologg%252Fkoelectra-small-v3-discriminator')}/config.json?op=OPEN"
    #     # config_path = f"http://dev01.ascent.com:50070/webhdfs/v1/user/nauts/mart/plm/models/{urllib.parse.quote('monologg%2Fkoelectra-small-v3-discriminator')}"

    #     print(vocab_path)
    #     print(config_path)

    #     res = get_from_cache(config_path, force_download=True)
    #     print(res)

    #     # from urllib.parse import urlparse
    #     # print(urlparse(config_path))
    #     # assert is_remote_url(config_path)

    #     # auto_config = AutoConfig.from_pretrained(config_path)
    #     # assert auto_config
    #     # tokenizer = AutoTokenizer.from_pretrained(vocab_path, config=auto_config)
    #     # assert tokenizer
