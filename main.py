# 功能说明：
# 1. 本脚本用于运行 HippoRAG 的检索与 QA 主流程。
# 2. 新增了只评测固定部分 query 的能力：--max_queries / --sample_seed / --subset_mode。
# 3. 为了快速试方法，本版本默认支持在完整 corpus 上建索引，但只对抽样后的 query 子集做评测。
# 4. 跑完后会直接打印 Recall@2 / Recall@5 / ExactMatch / F1，并保存本次子集下标以便复现实验。

import os
import random
import json
import argparse
import logging

from typing import List

from src.hipporag.HippoRAG import HippoRAG
from src.hipporag.utils.misc_utils import string_to_bool
from src.hipporag.utils.config_utils import BaseConfig

# os.environ["LOG_LEVEL"] = "DEBUG"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"


def get_gold_docs(samples: List, dataset_name: str = None) -> List:
    gold_docs = []
    for sample in samples:
        if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
            gold_title = set([item[0] for item in sample['supporting_facts']])
            gold_title_and_content_list = [item for item in sample['context'] if item[0] in gold_title]
            if dataset_name.startswith('hotpotqa'):
                gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
            else:
                gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
        elif 'contexts' in sample:
            gold_doc = [item['title'] + '\n' + item['text'] for item in sample['contexts'] if item['is_supporting']]
        else:
            assert 'paragraphs' in sample, "`paragraphs` should be in sample, or consider the setting not to evaluate retrieval"
            gold_paragraphs = []
            for item in sample['paragraphs']:
                if 'is_supporting' in item and item['is_supporting'] is False:
                    continue
                gold_paragraphs.append(item)
            gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]

        gold_doc = list(set(gold_doc))
        gold_docs.append(gold_doc)

    return gold_docs


def get_gold_answers(samples):
    gold_answers = []
    for sample_idx in range(len(samples)):
        gold_ans = None
        sample = samples[sample_idx]

        if 'answer' in sample or 'gold_ans' in sample:
            gold_ans = sample['answer'] if 'answer' in sample else sample['gold_ans']
        elif 'reference' in sample:
            gold_ans = sample['reference']
        elif 'obj' in sample:
            gold_ans = set(
                [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']]
            )
            gold_ans = list(gold_ans)

        assert gold_ans is not None

        if isinstance(gold_ans, str):
            gold_ans = [gold_ans]

        assert isinstance(gold_ans, list)
        gold_ans = set(gold_ans)

        if 'answer_aliases' in sample:
            gold_ans.update(sample['answer_aliases'])

        gold_answers.append(gold_ans)

    return gold_answers


def main():
    parser = argparse.ArgumentParser(description="HippoRAG retrieval and QA")
    parser.add_argument('--dataset', type=str, default='musique', help='Dataset name')
    parser.add_argument('--llm_base_url', type=str, default='https://api.key77qiqi.cn/v1', help='LLM base URL')
    parser.add_argument('--llm_name', type=str, default='gpt-4o-mini', help='LLM name')
    parser.add_argument('--embedding_name', type=str, default='nvidia/NV-Embed-v2', help='embedding model name')
    parser.add_argument(
        '--force_index_from_scratch',
        type=str,
        default='false',
        help='If set to True, will ignore all existing storage files and graph data and will rebuild from scratch.'
    )
    parser.add_argument(
        '--force_openie_from_scratch',
        type=str,
        default='false',
        help='If set to False, will try to first reuse openie results for the corpus if they exist.'
    )
    parser.add_argument(
        '--openie_mode',
        choices=['online', 'offline'],
        default='online',
        help='OpenIE mode, offline denotes using VLLM offline batch mode for indexing, while online denotes'
    )
    parser.add_argument('--save_dir', type=str, default='outputs', help='Save directory')

    # 新增：用于快速评测固定数量 query
    parser.add_argument(
        '--max_queries',
        type=int,
        default=None,
        help='只评测前/随机抽取的部分 query；例如 100'
    )
    parser.add_argument(
        '--sample_seed',
        type=int,
        default=42,
        help='随机抽样的种子，保证不同方法跑的是同一批 query'
    )
    parser.add_argument(
        '--subset_mode',
        choices=['first', 'random'],
        default='random',
        help='first=取前N条；random=固定seed随机抽N条'
    )

    args = parser.parse_args()

    dataset_name = args.dataset
    save_dir = args.save_dir
    llm_base_url = args.llm_base_url
    llm_name = args.llm_name

    if save_dir == 'outputs':
        save_dir = save_dir + '/' + dataset_name
    else:
        save_dir = save_dir + '_' + dataset_name

    os.makedirs(save_dir, exist_ok=True)

    corpus_path = f"reproduce/dataset/{dataset_name}_corpus.json"
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)

    docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

    force_index_from_scratch = string_to_bool(args.force_index_from_scratch)
    force_openie_from_scratch = string_to_bool(args.force_openie_from_scratch)

    # Prepare datasets and evaluation
    all_samples_path = f"reproduce/dataset/{dataset_name}.json"
    with open(all_samples_path, "r", encoding="utf-8") as f:
        all_samples = json.load(f)

    total_num_samples = len(all_samples)
    selected_indices = list(range(total_num_samples))

    if args.max_queries is not None and args.max_queries > 0 and args.max_queries < total_num_samples:
        if args.subset_mode == 'first':
            selected_indices = list(range(args.max_queries))
        else:
            rng = random.Random(args.sample_seed)
            selected_indices = sorted(rng.sample(range(total_num_samples), args.max_queries))

    samples = [all_samples[i] for i in selected_indices]

    print(
        f"[DEBUG] Evaluation setup: dataset={dataset_name}, "
        f"num_queries={len(samples)}/{total_num_samples}, "
        f"subset_mode={args.subset_mode}, seed={args.sample_seed}"
    )

    subset_record = {
        "dataset": dataset_name,
        "total_num_samples": total_num_samples,
        "num_queries_used": len(samples),
        "subset_mode": args.subset_mode,
        "sample_seed": args.sample_seed,
        "selected_indices": selected_indices
    }
    subset_record_path = os.path.join(save_dir, f"subset_{dataset_name}_{len(samples)}.json")
    with open(subset_record_path, "w", encoding="utf-8") as f:
        json.dump(subset_record, f, ensure_ascii=False, indent=2)

    all_queries = [s['question'] for s in samples]
    print(f"[DEBUG] actually running {len(all_queries)} queries")

    gold_answers = get_gold_answers(samples)

    try:
        gold_docs = get_gold_docs(samples, dataset_name)
        assert len(all_queries) == len(gold_docs) == len(gold_answers), \
            "Length of queries, gold_docs, and gold_answers should be the same."
        print(
            f"[DEBUG] evaluation enabled: "
            f"#queries={len(all_queries)}, #gold_docs={len(gold_docs)}, #gold_answers={len(gold_answers)}"
        )
    except Exception as e:
        print(f"[ERROR] get_gold_docs or evaluation setup failed: {e}")
        gold_docs = None

    config = BaseConfig(
        save_dir=save_dir,
        llm_base_url=llm_base_url,
        llm_name=llm_name,
        dataset=dataset_name,
        embedding_model_name=args.embedding_name,
        force_index_from_scratch=force_index_from_scratch,
        force_openie_from_scratch=force_openie_from_scratch,
        rerank_dspy_file_path="src/hipporag/prompts/dspy_prompts/filter_llama3.3-70B-Instruct.json",
        retrieval_top_k=200,
        linking_top_k=5,
        qa_top_k=5,
        embedding_batch_size=8,
        max_new_tokens=None,
        corpus_len=len(corpus),
        openie_mode=args.openie_mode
    )

    logging.basicConfig(level=logging.INFO)

    hipporag = HippoRAG(global_config=config)

    hipporag.index(docs)

    # Retrieval and QA
    result = hipporag.rag_qa(
        queries=all_queries,
        gold_docs=gold_docs,
        gold_answers=gold_answers
    )

    if gold_docs is not None:
        queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results = result

        final_summary = {
            "Recall@2": overall_retrieval_result.get("Recall@2", None),
            "Recall@5": overall_retrieval_result.get("Recall@5", None),
            "ExactMatch": overall_qa_results.get("ExactMatch", None),
            "F1": overall_qa_results.get("F1", None),
        }

        print("\n========== FINAL EVAL ==========")
        print(f"Recall@2   = {final_summary['Recall@2']}")
        print(f"Recall@5   = {final_summary['Recall@5']}")
        print(f"ExactMatch = {final_summary['ExactMatch']}")
        print(f"F1         = {final_summary['F1']}")
        print("================================\n")

        logging.info(f"Final evaluation summary: {final_summary}")
    else:
        queries_solutions, all_response_message, all_metadata = result
        print("[WARN] gold_docs is None, so retrieval metrics were skipped.")


if __name__ == "__main__":
    main()