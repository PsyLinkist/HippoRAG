# 功能说明：
# 1. 本脚本用于运行 HippoRAG 的检索与 QA 主流程。
# 2. 新增了完整运行日志保存能力：自动保存 print / logging / 报错 traceback / 运行参数 / 开始结束时间。
# 3. 运行日志会保存到 outputs/<dataset>/run_logs/ 下，便于后续复现实验与排查缓存、异常提分等问题。
# 4. 如果目标日志文件已存在，会先删除再写入，避免旧内容干扰。
# 5. 修复了退出阶段可能出现的非零退出码问题：会先恢复 stdout/stderr，再关闭日志文件，避免触发类似退出码 120 的误报。

import os
import sys
import time
import json
import random
import socket
import argparse
import logging
import traceback
from datetime import datetime
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


class Tee:
    """同时把输出写到终端和日志文件。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                pass

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


def safe_remove(path: str):
    if os.path.exists(path):
        os.remove(path)


def get_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_run_logging(args, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    log_dir = os.path.join(save_dir, "run_logs")
    os.makedirs(log_dir, exist_ok=True)

    run_name = f"run_{args.dataset}_{get_timestamp()}_pid{os.getpid()}"
    log_path = os.path.join(log_dir, f"{run_name}.log")
    meta_path = os.path.join(log_dir, f"{run_name}.meta.json")

    # 若同名文件存在，先删除再写新文件
    safe_remove(log_path)
    safe_remove(meta_path)

    log_fp = open(log_path, "w", encoding="utf-8", buffering=1)

    original_stdout = sys.stdout
    original_stderr = sys.stderr

    tee_stdout = Tee(original_stdout, log_fp)
    tee_stderr = Tee(original_stderr, log_fp)

    sys.stdout = tee_stdout
    sys.stderr = tee_stderr

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
        force=True,
    )

    meta = {
        "run_name": run_name,
        "dataset": args.dataset,
        "start_time": datetime.now().isoformat(),
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "command": " ".join(sys.argv),
        "args": vars(args),
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "CUDA_DEVICE_ORDER": os.environ.get("CUDA_DEVICE_ORDER"),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE"),
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[RUN] log file   : {log_path}")
    print(f"[RUN] meta file  : {meta_path}")
    print(f"[RUN] start time : {meta['start_time']}")
    print(f"[RUN] command    : {meta['command']}")

    return log_fp, log_path, meta_path, original_stdout, original_stderr


def finalize_run_logging(
    log_fp,
    log_path: str,
    meta_path: str,
    status: str,
    start_ts: float,
    original_stdout,
    original_stderr,
    extra: dict = None,
):
    end_time = datetime.now().isoformat()
    duration_sec = round(time.time() - start_ts, 3)

    print(f"[RUN] end time   : {end_time}")
    print(f"[RUN] status     : {status}")
    print(f"[RUN] duration_s : {duration_sec}")

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        meta = {}

    meta.update(
        {
            "end_time": end_time,
            "status": status,
            "duration_sec": duration_sec,
            "log_path": log_path,
        }
    )

    if extra is not None:
        meta["final_result"] = extra

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 先关闭 logging，避免后续还往 Tee 里写
    logging.shutdown()

    # 再恢复标准输出/错误，避免解释器退出时 flush 到已关闭的文件
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    # 最后关闭日志文件
    try:
        log_fp.flush()
    except Exception:
        pass
    try:
        log_fp.close()
    except Exception:
        pass


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
    parser.add_argument('--max_queries', type=int, default=None, help='只评测前/随机抽取的部分 query；例如 100')
    parser.add_argument('--sample_seed', type=int, default=42, help='随机抽样的种子，保证不同方法跑的是同一批 query')
    parser.add_argument('--subset_mode', choices=['first', 'random'], default='random', help='first=取前N条；random=固定seed随机抽N条')

    args = parser.parse_args()
    start_ts = time.time()

    dataset_name = args.dataset
    save_dir = args.save_dir
    llm_base_url = args.llm_base_url
    llm_name = args.llm_name

    if save_dir == 'outputs':
        save_dir = save_dir + '/' + dataset_name
    else:
        save_dir = save_dir + '_' + dataset_name

    os.makedirs(save_dir, exist_ok=True)
    log_fp, log_path, meta_path, original_stdout, original_stderr = setup_run_logging(args, save_dir)
    final_summary = None

    try:
        corpus_path = f"reproduce/dataset/{dataset_name}_corpus.json"
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)

        docs = [f"{doc['title']}\n{doc['text']}" for doc in corpus]

        force_index_from_scratch = string_to_bool(args.force_index_from_scratch)
        force_openie_from_scratch = string_to_bool(args.force_openie_from_scratch)

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
        if os.path.exists(subset_record_path):
            os.remove(subset_record_path)
        with open(subset_record_path, "w", encoding="utf-8") as f:
            json.dump(subset_record, f, ensure_ascii=False, indent=2)

        print(f"[RUN] subset record: {subset_record_path}")

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
            max_qa_steps=3,
            qa_top_k=5,
            graph_type="facts_and_sim_passage_node_unidirectional",
            embedding_batch_size=8,
            max_new_tokens=None,
            corpus_len=len(corpus),
            openie_mode=args.openie_mode
        )

        hipporag = HippoRAG(global_config=config)

        print("[RUN] Start indexing...")
        hipporag.index(docs)
        print("[RUN] Indexing finished.")

        print("[RUN] Start retrieval + QA...")
        result = hipporag.rag_qa(
            queries=all_queries,
            gold_docs=gold_docs,
            gold_answers=gold_answers
        )
        print("[RUN] Retrieval + QA finished.")

        if gold_docs is not None:
            queries_solutions, all_response_message, all_metadata, overall_retrieval_result, overall_qa_results = result

            final_summary = {
                "Recall@2": overall_retrieval_result.get("Recall@2", None),
                "Recall@5": overall_retrieval_result.get("Recall@5", None),
                "ExactMatch": overall_qa_results.get("ExactMatch", None),
                "F1": overall_qa_results.get("F1", None),
            }

            final_result_path = os.path.join(save_dir, f"final_result_{dataset_name}.json")
            if os.path.exists(final_result_path):
                os.remove(final_result_path)
            with open(final_result_path, "w", encoding="utf-8") as f:
                json.dump(final_summary, f, ensure_ascii=False, indent=2)

            print("\n========== FINAL EVAL ==========")
            print(f"Recall@2   = {final_summary['Recall@2']}")
            print(f"Recall@5   = {final_summary['Recall@5']}")
            print(f"ExactMatch = {final_summary['ExactMatch']}")
            print(f"F1         = {final_summary['F1']}")
            print("================================\n")
            print(f"[RUN] final result: {final_result_path}")

            logging.info(f"Final evaluation summary: {final_summary}")
        else:
            queries_solutions, all_response_message, all_metadata = result
            print("[WARN] gold_docs is None, so retrieval metrics were skipped.")

        finalize_run_logging(
            log_fp=log_fp,
            log_path=log_path,
            meta_path=meta_path,
            status="success",
            start_ts=start_ts,
            original_stdout=original_stdout,
            original_stderr=original_stderr,
            extra=final_summary,
        )

    except Exception:
        print("\n[RUN][FATAL] Program crashed. Full traceback below:\n")
        traceback.print_exc()
        finalize_run_logging(
            log_fp=log_fp,
            log_path=log_path,
            meta_path=meta_path,
            status="failed",
            start_ts=start_ts,
            original_stdout=original_stdout,
            original_stderr=original_stderr,
            extra=final_summary,
        )
        raise


if __name__ == "__main__":
    main()