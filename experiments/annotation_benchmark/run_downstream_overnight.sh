#!/usr/bin/env bash
# Chạy qua đêm: RQ2 downstream cho TẤT CẢ method, chia nhỏ từng (method x classifier).
#
# Mỗi cặp (method, classifier) là 1 lần gọi run_downstream.py riêng với --append,
# nên nếu 1 job lỗi/OOM thì các kết quả trước đó vẫn được giữ và script chạy tiếp.
#
# Cách chạy (nên dùng nohup/caffeinate để không bị ngắt khi ngủ máy):
#   nohup bash experiments/annotation_benchmark/run_downstream_overnight.sh > /dev/null 2>&1 &
#   # hoặc trên macOS:
#   caffeinate -is bash experiments/annotation_benchmark/run_downstream_overnight.sh
#
# Tuỳ chỉnh qua biến môi trường:
#   METHODS="zero_shot cot"            # mặc định: tất cả 6 method
#   CLASSIFIERS="tfidf phobert"        # mặc định: tfidf phobert xlmr qwen3 mistral
#   EPOCHS_LIGHT=10 EPOCHS_LORA=8      # epochs cho model thường / LoRA
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

if [[ -f env.sh ]]; then
  # shellcheck disable=SC1091
  source env.sh
fi

METHODS="${METHODS:-zero_shot few_shot cot freeal candist verified_d2}"
# Chạy classifier nhẹ trước, LoRA nặng sau
CLASSIFIERS="${CLASSIFIERS:-tfidf phobert xlmr qwen3 mistral}"
EPOCHS_LIGHT="${EPOCHS_LIGHT:-10}"   # tfidf/phobert/xlmr
EPOCHS_LORA="${EPOCHS_LORA:-8}"      # qwen3/mistral (LoRA)

LOG_DIR="experiments/annotation_benchmark/results/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
MAIN_LOG="$LOG_DIR/overnight_${STAMP}.log"

PASS=()
FAIL=()

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MAIN_LOG"
}

epochs_for() {
  case "$1" in
    qwen3|mistral) echo "$EPOCHS_LORA" ;;
    *) echo "$EPOCHS_LIGHT" ;;
  esac
}

log "=== Overnight downstream run bắt đầu ==="
log "Methods    : $METHODS"
log "Classifiers: $CLASSIFIERS"
log "Epochs     : light=$EPOCHS_LIGHT, lora=$EPOCHS_LORA"
log "Log chính  : $MAIN_LOG"

# Đảm bảo datasets đã build đủ
python experiments/annotation_benchmark/build_datasets.py --method all \
  >> "$MAIN_LOG" 2>&1 || log "WARNING: build_datasets.py lỗi, dùng datasets sẵn có"

# Vòng ngoài: classifier (load model 1 lần ít tốn hơn khi đổi method liên tục thì
# vẫn phải reload, nhưng nhóm theo classifier giúp job nhẹ xong sớm, có kết quả sớm)
for clf in $CLASSIFIERS; do
  ep="$(epochs_for "$clf")"
  for method in $METHODS; do
    name="${method}__${clf}"
    job_log="$LOG_DIR/${name}_${STAMP}.log"
    log ">>> BẮT ĐẦU $name (epochs=$ep) — log: $job_log"
    start=$(date +%s)

    if python experiments/annotation_benchmark/run_downstream.py \
        --method "$method" \
        --classifiers "$clf" \
        --epochs "$ep" \
        --append \
        > "$job_log" 2>&1; then
      dur=$(( $(date +%s) - start ))
      log "<<< XONG $name (${dur}s)"
      PASS+=("$name")
    else
      dur=$(( $(date +%s) - start ))
      log "<<< LỖI $name (${dur}s) — xem $job_log"
      tail -n 5 "$job_log" | sed 's/^/      /' | tee -a "$MAIN_LOG"
      FAIL+=("$name")
    fi
  done
done

log ""
log "=== TỔNG KẾT ==="
log "Thành công (${#PASS[@]}): ${PASS[*]:-—}"
log "Thất bại   (${#FAIL[@]}): ${FAIL[*]:-—}"
log "Kết quả gộp: experiments/annotation_benchmark/results/downstream_performance.csv"

if [[ ${#FAIL[@]} -gt 0 ]]; then
  exit 1
fi
