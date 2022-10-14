import argparse
import json
import pathlib
import re
import sys
import time
import unicodedata

from abc import ABC
from io import BufferedWriter, FileIO
from itertools import chain

from Korpora import KowikiTextKorpus, NamuwikiTextKorpus, ModuNEKorpus, ModuMorphemeKorpus

from tunip.corpus.corpus_path_provider import CorpusPathProviderFactory
from tunip.constants import PRETRAIN_SENT_SEPARATOR


class KorporaBuilder(ABC):
    def use_korpora(self):
        return True


class DummyCorpus:
    pass


class NotSupportedCorpusTypeException(Exception):
    pass


class KorporaWikiBuilder(KorporaBuilder):

    def __init__(self, file_name, out_filepath, buffer_size=256*1024*1024):
        self.out_filepath = f"{out_filepath}/{file_name}"
        self.buffer_size = buffer_size
        self.ko_split_re = re.compile("\\n+|(?<=다\.)")

        all_chars = (chr(i) for i in range(sys.maxunicode))
        categories = {'Cf'}
        non_printable_chars = ''.join(c for c in all_chars if unicodedata.category(c) in categories)
        self.non_printable_char_re = re.compile('[%s]' % re.escape(non_printable_chars))

    def build(self, corpus):
        # titles = [
        #     " = 송찬호 =",
        #     " = = 약력  = =",
        #     " = = 수상  = =",
        #     " = = 저서  = =",
        #     " = = =시집 = = =",
        #     " = = =동시집 = = =",
        #     " = = =시집 = = =",
        #     " = = =동시집 = = =",
        #     " = 분류:나라별 기사 =",
        #     " = 분류:스리랑카의 언어 =",
        #     " = 분류:캘리포니아주 출신 배우 =",
        #     " = 베드로 십자 =",
        #     " = = 기원  = =",
        #     " = = 상징  = =",
        #     " = = 악마 숭배와의 관계  = =",
        #     " = = 관련 항목  = =",
        #     " = = 외부 링크  = =",
        #     " = 분류:1899년 건축 =",
        #     " = 틀:남아메리카의 행정 구역 =",
        #     " = 탐색이론 =",
        #     " = = 외부 링크  = =",
        # ]
        titles = corpus.train.get_all_pairs()
        texts = corpus.train.get_all_texts()
        #titles = corpus.train[:20].pair
        #texts = corpus.train[:20].text

        t = time.process_time()

        fio = FileIO(self.out_filepath, 'w')
        sout = BufferedWriter(fio, buffer_size=self.buffer_size)
        buffers = []
        line_count = 0
        for (title_txt, txt_) in zip(titles, texts):
            title_txt = title_txt.strip()
            if re.search(r"^(=\s*)[^=\s]", title_txt) or title_txt == '=':
                # get the start of article
                if buffers and title_txt != "\n":
                    # \u2028 line separator
                    line = PRETRAIN_SENT_SEPARATOR.join(chain(*[[sent.strip() for sent in self.ko_split_re.split(buf)] for buf in buffers])).strip()
                    if len(line) < 2 or not line:
                        continue

                    sout.write(line.encode("utf-8"))
                    sout.write("\n".encode("utf-8"))

                    txt_ = self.non_printable_char_re.sub('', txt_)
                    buffers = [txt_]
                    line_count += 1
                else:
                    txt_ = self.non_printable_char_re.sub('', txt_)
                    buffers.append(txt_)
                    line_count += 1
            elif re.search(r"^(=\s*){2,}", title_txt):
                txt_ = self.non_printable_char_re.sub('', txt_)
                buffers.append(txt_)
                line_count += 1
            else:
                print(f"[ERROR] Not supported section formats: {title_txt} : {txt_}")
                exit(0)
            if line_count % 1000 == 0:
                print(f"{line_count} number of lines are reading ...")
        
        if buffers:
            line = PRETRAIN_SENT_SEPARATOR.join(chain(*[[sent.strip() for sent in self.ko_split_re.split(buf)] for buf in buffers])).strip()

            sout.write(line.encode("utf-8"))
            sout.write("\n".encode("utf-8"))

            print(f"{line_count} number of lines are reading ...")
        
        sout.flush()
        sout.close()

        print(f"It takes about {(time.process_time() - t)/60.} minutes.")
        print("END TO build lined documents.")


# Document Line Builders
class KowikiArticleLineBuilder(KorporaWikiBuilder):

    def __init__(self, filename, out_filepath, buffer_size=256*1024*1024):
        super(KowikiArticleLineBuilder, self).__init__(filename, out_filepath, buffer_size)

    def __call__(self, corpus: KowikiTextKorpus):
        self.build(corpus)
        

class NamuwikiArticleLineBuilder(KorporaWikiBuilder):

    def __init__(self, filename, out_filepath, buffer_size=256*1024*1024):
        super(NamuwikiArticleLineBuilder, self).__init__(filename, out_filepath, buffer_size)

    def __call__(self, corpus: NamuwikiTextKorpus):
        self.build(corpus)


class KorporaModooCorpusBuilder(KorporaBuilder):

    def __init__(self, filename, out_filepath, buffer_size=256*1024*1024):
        self.filepath = filename
        self.out_filepath = out_filepath
        self.buffer_size = buffer_size
        self.ko_split_re = re.compile("\\n+|(?<=다\.)")

        all_chars = (chr(i) for i in range(sys.maxunicode))
        categories = {'Cf'}
        non_printable_chars = ''.join(c for c in all_chars if unicodedata.category(c) in categories)
        self.non_printable_char_re = re.compile('[%s]' % re.escape(non_printable_chars))

    def build(self, corpus, extension = ".json"):
        path_provider = CorpusPathProviderFactory.create(
            corpus_type="dir",
            input_dir=self.filepath,
            extensions=[extension],
            split_path=""
        )

        for filepath in path_provider():
            fio = FileIO(f"{pathlib.Path(self.out_filepath)}/{pathlib.Path(filepath).with_suffix('.txt').name}", 'w')
            sout = BufferedWriter(fio, buffer_size=self.buffer_size)

            with open(filepath, encoding='utf-8') as dsf:
                article_json = json.load(dsf, encoding='utf-8')
         
            for doc in article_json['document']:
                buffers = []
                for sent in doc['sentence']:
                    buffers.append(sent['form'])

                line = PRETRAIN_SENT_SEPARATOR.join([sent.strip() for sent in buffers])
                line = self.non_printable_char_re.sub('', line)
                sout.write(line.encode('utf-8'))
                sout.write('\n'.encode('utf-8'))

            sout.flush()
            sout.close()


class ModooCorpusArticleLineBuilder:

    def __init__(self, filepath, out_filepath, buffer_size=256*1024*1024):
        self.filepath = filepath
        self.out_filepath = out_filepath
        self.buffer_size = buffer_size

        all_chars = (chr(i) for i in range(sys.maxunicode))
        categories = {'Cf'}
        non_printable_chars = ''.join(c for c in all_chars if unicodedata.category(c) in categories)
        self.non_printable_char_re = re.compile('[%s]' % re.escape(non_printable_chars))
    
    def __call__(self, corpus):

        path_provider = CorpusPathProviderFactory.create(
            corpus_type="dir",
            input_dir=self.filepath,
            extensions=[".json"],
            split_path=""
        )

        for filepath in path_provider():
            fio = FileIO(f"{pathlib.Path(self.out_filepath)}/{pathlib.Path(filepath).with_suffix('.txt').name}", 'w')
            sout = BufferedWriter(fio, buffer_size=self.buffer_size)

            with open(filepath, encoding='utf-8') as dsf:
                article_json = json.load(dsf, encoding='utf-8')
            
            for doc in article_json['document']:
                buffers = []
                for sent in doc['paragraph']:
                    buffers.append(sent['form'])

                line = PRETRAIN_SENT_SEPARATOR.join([sent.strip() for sent in buffers])
                line = self.non_printable_char_re.sub('', line)
                sout.write(line.encode('utf-8'))
                sout.write('\n'.encode('utf-8'))

            sout.flush()
            sout.close()


class ModooCorpusNamedEntityBuilder(KorporaModooCorpusBuilder):

    def __init__(self, filename, out_filepath, buffer_size=256*1024*1024):
        super(ModooCorpusNamedEntityBuilder, self).__init__(filename, out_filepath, buffer_size)

    def __call__(self, corpus):
        self.build(corpus, extension = ".JSON")


class ModooCorpusMorphemeBuilder(KorporaModooCorpusBuilder):

    def __init__(self, filename, out_filepath, buffer_size=256*1024*1024):
        super(ModooCorpusMorphemeBuilder, self).__init__(filename, out_filepath, buffer_size)

    def __call__(self, corpus):
        fio = FileIO(f"{pathlib.Path(self.out_filepath)}/nikl_mp.txt", 'w')
        sout = BufferedWriter(fio, buffer_size=self.buffer_size)
        buffers = []
        doc_id = '.'.join(corpus.train[0].sentence_id.split('.')[:2])

        for morpheme in corpus.train:
            morpheme_doc_id = '.'.join(morpheme.sentence_id.split('.')[:2])

            if (doc_id == morpheme_doc_id):
                buffers.append(morpheme.sentence)
                
            elif(doc_id != morpheme_doc_id):
                doc_id = morpheme_doc_id

                if buffers:
                    line = PRETRAIN_SENT_SEPARATOR.join([sent.strip() for sent in buffers])
                    line = self.non_printable_char_re.sub('', line)

                    sout.write(line.encode('utf-8'))
                    sout.write('\n'.encode('utf-8'))
                buffers = []

        if buffers:
            line = PRETRAIN_SENT_SEPARATOR.join([sent.strip() for sent in buffers])
            line = self.non_printable_char_re.sub('', line)

            sout.write(line.encode('utf-8'))
            sout.write('\n'.encode('utf-8'))


class ModooCorpusDependentBuilder(KorporaModooCorpusBuilder):

    def __init__(self, filename, out_filepath, buffer_size=256*1024*1024):
        super(ModooCorpusDependentBuilder, self).__init__(filename, out_filepath, buffer_size)

    def __call__(self, corpus):
        self.build(corpus)


def main(args):

    pathlib.Path(args.output_file_path).mkdir(parents=True, exist_ok=True)

    if args.corpus_name == 'kowiki':

        file_name = "kowiki.articles.txt" if not args.file_path else args.file_path
        corpus = KowikiTextKorpus()
        article_builder = KowikiArticleLineBuilder(file_name, args.output_file_path)

    elif args.corpus_name == 'namuwiki':

        file_name = "namuwiki.articles.txt"
        corpus = NamuwikiTextKorpus()
        article_builder = NamuwikiArticleLineBuilder(file_name, args.output_file_path)

    elif args.corpus_name == 'nikl_news':

        corpus = DummyCorpus()
        article_builder = ModooCorpusArticleLineBuilder(args.file_path, args.output_file_path)

    elif args.corpus_name == 'nikl_ne':

        corpus = DummyCorpus()
        article_builder = ModooCorpusNamedEntityBuilder(args.file_path, args.output_file_path)

    elif args.corpus_name == 'nikl_mp':

        corpus = ModuMorphemeKorpus() if not args.file_path else ModuMorphemeKorpus(root_dir = args.file_path)
        article_builder = ModooCorpusMorphemeBuilder(args.file_path, args.output_file_path)

    elif args.corpus_name == 'nikl_dp':

        corpus = DummyCorpus()
        article_builder = ModooCorpusDependentBuilder(args.file_path, args.output_file_path)

    else:
        raise NotSupportedCorpusTypeException()

    article_builder(corpus)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="build corpus for each document by line")

    parser.add_argument(
        '-cn',
        '--corpus_name',
        type=str,
        required=True
    )
    parser.add_argument(
        '-fp',
        '--file_path',
        type=str,
        required=False,
        default=None
    )
    parser.add_argument(
        '-of',
        '--output_file_path',
        type=str,
        required=False,
        default=None
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
