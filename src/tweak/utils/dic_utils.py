
def get_dic_entries(dic_filepath):

    dic_entries = []
    with open(dic_filepath, encoding='utf-8') as dic_f:
        dic_entries = dic_f.readlines()

    return dic_entries

