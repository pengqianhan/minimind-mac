import itertools
import re
import json
import jsonlines
import psutil
import ujson
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from datasets import load_dataset

bos_token = "<s>"
eos_token = "</s>"

# pretrain
def process_wiki_clean():
    with open('./dataset/clean-wikipedia-cn.json', 'r', encoding='utf-8') as f_read:
        data = [ujson.loads(line) for line in f_read]
    data_len = len(data)
    doc_ids = []
    for idx, line in enumerate(data):
        text = line['response']
        text_id = tokenizer(f'{bos_token}{text}{eos_token}').data['input_ids']
        if len(text_id) > 5:
            doc_ids += text_id
        if idx % (int(data_len / 20)) == 0:
            print(f"[{idx}/{data_len}] {text}")
    arr = np.array(doc_ids, dtype=np.uint16)
    with open('./dataset/clean-wikipedia-cn.bin', 'wb') as f:
        f.write(arr.tobytes())


# pretrain
def process_other():
    data = []

    with open('./dataset/alpaca_gpt4_data_zh.json', 'r', encoding='utf-8') as f:
        data_ = json.load(f)
        data += data_

    with open('./dataset/alpaca_data_zh_51k.json', 'r', encoding='utf-8') as f:
        data_ = json.load(f)
        data += data_

    doc_ids = []
    for idx, per in enumerate(data):
        q = per['instruction']
        i = per['input']
        a = per['output']
        q = q + i
        if len(q) < 10 or len(a) < 5:
            continue
        if len(q) > 256 or len(a) > 256:
            continue
        text_id = tokenizer(f'{bos_token}{q}，{a}{eos_token}').data['input_ids']
        if len(text_id) > 5:
            doc_ids += text_id
        if idx % 50000 == 0:
            print(idx, len(data))

    arr = np.array(doc_ids, dtype=np.uint16)
    with open('./dataset/clean_other.bin', 'wb') as f:
        f.write(arr.tobytes())


def process_seq_monkey(chunk_size=50000):
    doc_ids = []
    chunk_idx = 0

    with jsonlines.open('./dataset/mobvoi_seq_monkey_general_open_corpus.jsonl') as reader:
        while True:
            chunk = list(itertools.islice(reader, chunk_size))
            if not chunk:
                break
            # print the length of chunk 
            # print(len(chunk))## = chunk_size
            for idx, obj in enumerate(chunk):
                try:
                    content = obj.get('text', '')
                    if len(content) > 512:
                        continue
                    text_id = tokenizer(f'{bos_token}{content}{eos_token}').data['input_ids']
                    # print(text_id['input_ids']==text_id.data['input_ids'])## == True
                    
                    doc_ids += text_id
                except UnicodeDecodeError as e:
                    print(f"Skipping invalid line {chunk_idx * chunk_size + idx + 1}: {e}")
                    continue

            chunk_idx += 1
            print(f"Processed chunk {chunk_idx} with {chunk_size} lines")

            ### 为了训练速度，只取前 20 个chunk
            if chunk_idx > 20:
                break
            #####################

            if len(doc_ids) > 1000000:
                arr = np.array(doc_ids, dtype=np.uint16)
                with open(f'./dataset/clean_seq_monkey.bin', 'ab') as f:
                    f.write(arr.tobytes())
                doc_ids = []

    if doc_ids:
        arr = np.array(doc_ids, dtype=np.uint16)
        with open(f'./dataset/clean_seq_monkey.bin', 'ab') as f:
            f.write(arr.tobytes())


def pretrain_process():
    # process_wiki_clean()
    process_seq_monkey()

    data_path_list = [
        # './dataset/clean-wikipedia-cn.bin',
        './dataset/clean_seq_monkey.bin'
    ]
    data_lst = []
    for data_path in data_path_list:
        with open(data_path, 'rb') as f:
            data = np.fromfile(f, dtype=np.uint16)
            data_lst.append(data)
    # print('len(data_lst):',len(data_lst))##==data_path_list 里文件数量
    arr = np.concatenate(data_lst)
    print(arr.shape)
    with open('./dataset/pretrain_data.bin', 'wb') as f:
        f.write(arr.tobytes())


def sft_process(contain_history=False):
    file_name = 'sft_data.csv'
    if not contain_history:
        file_name = 'sft_data_single.csv'

    def chinese_ratio(text):
        # 匹配所有中文字符
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        # 中文字符数量占比
        return len(chinese_chars) / len(text) if text else 0

    def process_and_write_data(data):
        q_lst, a_lst, history_lst = [], [], []
        for per in data:
            history, q, a = per['history'], per['q'], per['a']

            if (contain_history and not history) or not q or not a:
                continue
            if len(q) < 10 or len(a) < 5:
                continue
            if len(q) > 256 or len(a) > 256:
                continue
            # 判断q和a中中文字符占比是否超过70%
            if not (chinese_ratio(q) > 0.9 and chinese_ratio(a) > 0.9):
                continue

            q_lst.append(q)
            a_lst.append(a)
            if contain_history:
                history_lst.append(history)
            else:
                history_lst.append([])

        # 创建DataFrame并追加到CSV文件
        df = pd.DataFrame({'history': history_lst, 'q': q_lst, 'a': a_lst})
        df.to_csv(f'./dataset/{file_name}', mode='a', header=False, index=False, lineterminator='\r\n')

    chunk_size = 1000  # 每次处理的记录数
    data = []

    with open(f'./dataset/{file_name}', 'w', encoding='utf-8') as f:
        f.write('history,q,a\n')

    sft_datasets = ['./dataset/sft_data_zh.jsonl']
    if not contain_history:
        sft_datasets = ['./dataset/sft_data_zh.jsonl']

    for path in sft_datasets:
        with jsonlines.open(path) as reader:
            for idx, obj in enumerate(reader):
                try:
                    data.append({
                        'history': obj.get('history', ''),
                        'q': obj.get('input', '') + obj.get('q', ''),
                        'a': obj.get('output', '') + obj.get('a', '')
                    })

                    if len(data) >= chunk_size:
                        process_and_write_data(data)
                        data = []
                except jsonlines.InvalidLineError as e:
                    print(f"Skipping invalid JSON line {idx + 1}: {e}")
                    continue

            if data:
                process_and_write_data(data)
                data = []

def rl_process():
    ################
    # Dataset
    ################

    dataset_path = ['./dataset/dpo/dpo_zh_demo.json',
                    './dataset/dpo/train_data.json',
                    './dataset/dpo/huozi_rlhf_data.json', ]

    train_dataset = load_dataset('json', data_files=dataset_path)

    def process(row):
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["reject"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    ds = train_dataset.map(
        process,
        load_from_cache_file=False,
    )

    output_dataset_path = './dataset/dpo/train_data.json'
    ds['train'].to_json(output_dataset_path, force_ascii=False, orient='records', lines=True)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer', use_fast=False)
    print('tokenizer词表大小：', len(tokenizer))##6400；minimind tokenizer 词表大小为6400

    ################
    # 1: pretrain
    # 2: sft
    # 3: RL
    ################
    process_type = 1

    if process_type == 1:
        pretrain_process()
        ###在dataset目录下生成了pretrain_data.bin和clean_seq_monkey.bin两个文件
    if process_type == 2:
        sft_process(contain_history=False)
    if process_type == 3:
        rl_process()
