# -*- coding: utf-8 -*-
"""
组会复现实验脚本 —— LoRA rank 消融 + DPO 偏好对齐
================================================================
对应精读论文：
  实验 lora → LoRA: Low-Rank Adaptation of Large Language Models (ICLR 2022)
  实验 dpo  → Direct Preference Optimization (NeurIPS 2023)

本机环境（已实测，2026-11-25）：
  解释器 : C:\\Users\\Spong\\anaconda3\\envs\\pytorch\\python.exe
  GPU    : RTX 3060 Laptop, 6GB 显存
  已装   : torch 2.13.0+cu126 / transformers 5.22.0 / peft 0.21.2
           datasets 5.1.0 / accelerate 1.15.0
  需补   : pip install -U trl          (仅实验 dpo 需要)
  可选   : pip install bitsandbytes    (想用 4-bit 时)
  国内   : set HF_ENDPOINT=https://hf-mirror.com

用法：
  python repro_lora_dpo.py --exp lora
  python repro_lora_dpo.py --exp dpo --beta 0.1
  python repro_lora_dpo.py --exp dpo --beta 0.3     # 第二组消融

重要说明：
  * 这是「缩小规模复现」(shrink-scale reproduction)：用小模型+小数据集
    验证论文的【机制与趋势】，不追求复现论文主表数字（那是 175B/6B + 多卡）。
  * 脚本是运行骨架，未在目标机器上实际执行训练。首次运行请按本机库版本
    微调个别 API 名称（transformers/trl 5.x 的改名点已在注释里标出）。
  * 所有默认超参按 6GB 显存保守设置：小 batch、短序列、梯度检查点。
"""

import argparse
import json
import os
import time

# 国内镜像（也可在命令行提前 set HF_ENDPOINT）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch


# ----------------------------------------------------------------------
# 公共工具
# ----------------------------------------------------------------------
def gpu_peak_gb():
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 3)
    return None


def reset_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"[saved] {path}")


# ----------------------------------------------------------------------
# 实验一：LoRA 的 rank 消融  —— 验证「权重更新是低秩的」
# ----------------------------------------------------------------------
def run_lora(args):
    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, TaskType, get_peft_model

    MODEL_ID = args.model
    RANKS = [1, 2, 4, 8, 16]
    results = []

    print(f"[exp1] model={MODEL_ID}  ranks={RANKS}")
    # datasets 5.x 不再解析无命名空间的 "glue"，GLUE 现托管在 nyu-mll/glue
    ds = load_dataset("nyu-mll/glue", "sst2")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    def tokenize(batch):
        # 只做分词，保留 label 列供 Trainer 使用
        return tok(batch["sentence"], truncation=True, max_length=128)

    train_ds = ds["train"].shuffle(seed=42).select(range(args.n_train)).map(
        tokenize, batched=True
    )
    eval_ds = ds["validation"].map(tokenize, batched=True)
    collator = DataCollatorWithPadding(tokenizer=tok)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        acc = float((preds == labels).mean())
        return {"accuracy": acc}

    for r in RANKS:
        print(f"\n--- rank r = {r} ---")
        reset_peak()
        t0 = time.time()

        base = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, num_labels=2
        )
        # RoBERTa / DistilBERT 的 attention 投影名不同：
        #   roberta-*   : query / key / value
        #   distilbert-*: q_lin / k_lin / v_lin
        if "distilbert" in MODEL_ID.lower():
            targets = ["q_lin", "v_lin"]
        else:
            targets = ["query", "value"]

        cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=r,
            lora_alpha=2 * r,
            lora_dropout=0.05,
            target_modules=targets,
        )
        model = get_peft_model(base, cfg)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  trainable = {trainable:,} / {total:,} ({100 * trainable / total:.4f}%)")

        # transformers 5.x 用 eval_strategy（旧版为 evaluation_strategy）
        targs = TrainingArguments(
            output_dir=f"./out_lora_r{r}",
            learning_rate=2e-4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=args.epochs,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=20,
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            report_to="none",
            seed=42,
        )
        trainer = Trainer(
            model=model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        metrics = trainer.evaluate()

        rec = {
            "rank": r,
            "trainable_params": trainable,
            "total_params": total,
            "trainable_pct": round(100 * trainable / total, 4),
            "accuracy": round(float(metrics.get("eval_accuracy", 0.0)), 4),
            "peak_vram_gb": gpu_peak_gb(),
            "seconds": round(time.time() - t0, 1),
        }
        results.append(rec)
        print(f"  -> {rec}")

        # 及时释放，避免多轮循环显存累积
        del trainer, model, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_json(results, "lora_rank_ablation.json")
    print("\n[exp1] 汇总（横轴 r，折线画 accuracy，柱状画 trainable_params）：")
    for x in results:
        print(f"  r={x['rank']:2d}  acc={x['accuracy']:.4f}  "
              f"params={x['trainable_params']:>9,}  vram={x['peak_vram_gb']} GB")
    print("\n预期：r=1~4 的 accuracy 已接近 r=16，而参数量随 r 线性增长 → 低秩假设成立。")


# ----------------------------------------------------------------------
# 实验二：DPO 偏好对齐  —— 验证「不需要奖励模型也能拉开 chosen/rejected」
# ----------------------------------------------------------------------
def run_dpo(args):
    try:
        from trl import DPOConfig, DPOTrainer
    except ImportError:
        raise SystemExit(
            "缺少 trl。请先执行：  pip install -U trl\n"
            "（PEFT / TRL 版本变动快，建议把 transformers / peft / trl 版本记进报告）"
        )
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, TaskType

    MODEL_ID = args.model
    print(f"[exp2] model={MODEL_ID}  beta={args.beta}  n_train={args.n_train}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model.config.use_cache = False  # 与 gradient_checkpointing 兼容

    raw = load_dataset("trl-lib/ultrafeedback_binarized", split="train")
    raw = raw.shuffle(seed=42)
    n = min(args.n_train, len(raw) - 500)
    train_ds = raw.select(range(n))
    eval_ds = raw.select(range(n, min(n + 400, len(raw))))

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules="all-linear",
    )

    reset_peak()
    t0 = time.time()

    cfg = DPOConfig(
        output_dir=f"./out_dpo_beta{args.beta}",
        beta=args.beta,                       # 唯一主要超参：控制偏离参考模型的强度
        learning_rate=5e-6,                   # DPO 比 SFT 小 1~2 个数量级（LoRA 下典型 5e-6）
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,        # 有效 batch = 16 pairs / step
        max_length=1024,
        max_prompt_length=512,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="no",
        bf16=False,
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to="none",
        seed=42,
    )

    # 关键：传了 peft_config 后令 ref_model=None，TRL 会用「同一网络、关掉 adapter」
    # 作为参考模型 —— 不需要第二份模型权重，这是 6GB 能跑通的核心。
    trainer_kwargs = dict(
        model=model,
        args=cfg,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peft_config,
        ref_model=None,
    )
    # trl 新版本用 processing_class，旧版本用 tokenizer
    try:
        trainer = DPOTrainer(processing_class=tok, **trainer_kwargs)
    except TypeError:
        trainer = DPOTrainer(tokenizer=tok, **trainer_kwargs)

    trainer.train()
    metrics = trainer.evaluate()

    # 从训练日志里提取 margin / reward_accuracy 曲线
    history = [h for h in trainer.state.log_history if "loss" in h]

    rec = {
        "model": MODEL_ID,
        "beta": args.beta,
        "n_train": n,
        "eval_metrics": {k: float(v) for k, v in metrics.items()
                         if isinstance(v, (int, float))},
        "final_log": history[-1] if history else None,
        "peak_vram_gb": gpu_peak_gb(),
        "seconds": round(time.time() - t0, 1),
    }
    save_json(rec, f"dpo_beta{args.beta}.json")

    print("\n[exp2] 关注三个量：")
    print("  1) loss 是否下降")
    print("  2) rewards/chosen - rewards/rejected 的 margin 是否上升、reward_accuracy 是否上升")
    print("  3) 输出平均长度是否明显变长（DPO 的长度膨胀局限）")
    print(f"\n实际记录：{json.dumps(rec, ensure_ascii=False, indent=2)}")
    print("\n再跑一组做 β 消融：")
    print(f"  python repro_lora_dpo.py --exp dpo --beta 0.3")


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="组会复现实验：LoRA rank 消融 / DPO 偏好对齐")
    p.add_argument("--exp", choices=["lora", "dpo"], required=True)
    p.add_argument("--model", default=None,
                   help="覆盖默认模型；lora 默认 roberta-base，dpo 默认 Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--n_train", type=int, default=2000, help="训练样本数（6GB 保守值）")
    p.add_argument("--epochs", type=int, default=2, help="实验一训练轮数")
    p.add_argument("--beta", type=float, default=0.1, help="实验二 DPO 的 beta")
    args = p.parse_args()

    if args.model is None:
        args.model = ("roberta-base" if args.exp == "lora"
                      else "Qwen/Qwen2.5-0.5B-Instruct")

    print("=" * 68)
    print(f"实验: {args.exp} | model: {args.model} | n_train: {args.n_train}")
    print(f"CUDA: {torch.cuda.is_available()} | "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 68)

    if args.exp == "lora":
        run_lora(args)
    else:
        run_dpo(args)


if __name__ == "__main__":
    main()
