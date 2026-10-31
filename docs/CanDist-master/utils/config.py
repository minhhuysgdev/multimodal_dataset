# -*- coding: utf-8  -*-
import argparse

def config():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir",
    )
    parser.add_argument(
        "--student1_model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--student1_config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="bert-base-uncased",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded",
    )

    # model/training args
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=128, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="BETA1 for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.98, type=float, help="BETA2 for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument("--warmup_steps", default=200, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument('--hidden_dropout_prob', default=0.3, type=float, help="hidden dropout rate")
    parser.add_argument('--embedding_dim', default=768, type=int, help="embedding dimension")
    
    # data args
    parser.add_argument('--task_name', default='banking', type=str, help="a specific dataset")
    parser.add_argument("--num_labels", default=77, type=int, help="size of the label space")
    parser.add_argument('--label_source', default='train_add_bt', type=str, help="annotation source file")

    # method args
    parser.add_argument('--is_mu', type=bool, default=True, help="Whether using mixup")
    parser.add_argument('--is_cr', type=bool, default=True, help="Whether using consistency regularization")
    parser.add_argument('--warmup_epoch', type=int, default=5, help="The number of warm-up epochs for prototype")
    parser.add_argument('--tau_loss', type=float, default=0.5, help="Threshold for small loss selection")
    parser.add_argument('--tau_conf', type=float, default=0.95, help="Threshold for high confidence selection")
    parser.add_argument('--temp_u', type=float, default=0.85, help="The temperature for unlabeled data, following UDA")
    parser.add_argument('--sharpen', type=float, default=0.85, help="The temperature for distribution sharpening")

    args = parser.parse_args()

    return args
    