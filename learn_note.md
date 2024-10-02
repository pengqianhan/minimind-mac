参考 官方readme 和 https://github.com/jingyaogong/minimind/issues/26

0. train tokenizer
   - 下载 | **【tokenizer训练集】** | [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main) / [百度网盘](https://pan.baidu.com/s/1yAw1LVTftuhQGAC1Y9RdYQ?pwd=6666) 文件为 tokenizer_train.jsonl 
   - [博客或视频讲解](https://www.bilibili.com/video/BV1KZ421M7di/?spm_id_from=333.880.my_history.page.click&vd_source=e587bac74600ca53ef886eea337fe87d)
1. data_process.py 处理数据，为pretrain 数据集做准备
   - 下载 | **【Pretrain数据】**   | [Seq-Monkey官方](http://share.mobvoi.com:5000/sharing/O91blwPkY)  / [百度网盘](https://pan.baidu.com/s/1-Z8Q37lJD4tOKhyBs1D_6Q?pwd=6666) / [HuggingFace](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main) | 此处是从hugging face下载 mobvoi_seq_monkey_general_open_corpus.jsonl 文件,大小为 14.5GB,解压后为33.39GB
   - 运行 data_process.py ，处理mobvoi_seq_monkey_general_open_corpus.jsonl，
       - if process_type == 1:在dataset目录下生成了pretrain_data.bin和clean_seq_monkey.bin两个文件
       - process_type == 2:
       - process_type == 3:
2. pretrain.py 预训练model
   -  使用 ./dataset/pretrain_data.bin 来预训练，直接运行1-pretrain.py即可
3. 