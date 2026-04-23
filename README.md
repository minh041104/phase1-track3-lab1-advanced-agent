# Lab 16 - Reflexion Agent Scaffold

Repo nay cung cap mot scaffold de xay dung va danh gia Reflexion Agent tren bai toan QA nhieu buoc.

## Muc tieu
- Hieu luong ReAct va Reflexion qua actor, evaluator, reflector.
- Co the chay mock mode de debug nhanh va chay LLM that de benchmark.
- Xuat `report.json` va `report.md` dung schema cho autograde.

## Nhung gi da co trong scaffold
- `src/reflexion_lab/schemas.py`: schema cho judge result, reflection, trace, run record.
- `src/reflexion_lab/agents.py`: loop chay ReAct va Reflexion, gom token/latency theo tung attempt.
- `src/reflexion_lab/runtime.py`: runtime `mock` va runtime OpenAI-compatible.
- `src/reflexion_lab/prompts.py`: prompt cho actor, evaluator, reflector.
- `src/reflexion_lab/reporting.py`: tong hop benchmark va xuat report.
- `run_benchmark.py`: script benchmark.
- `autograde.py`: script cham nhanh theo report schema.

## Cach chay
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Chay mock mode
```bash
python run_benchmark.py --dataset data/hotpot_mini.json --out-dir outputs/sample_run --mode mock
python autograde.py --report-path outputs/sample_run/report.json
```

### Mot so tuy chon huu ich khi benchmark
- `--limit 100`: chi lay 100 mau dau tien sau khi da shuffle/offset.
- `--offset 100`: bo qua 100 mau dau.
- `--shuffle --seed 13`: tron dataset truoc khi cat mau.
- `--workers 4`: chay song song theo example, van giu tuan tu ben trong moi example.
- `--timeout-s 120`: tang thoi gian cho moi request LLM.
- `--max-retries 4 --retry-backoff-s 2`: tu dong retry khi timeout hoac gap loi tam thoi.
- `--no-adaptive-attempts`: tat co che dung som khi Reflexion bi loop.
- `--memory-limit 3`: chi giu lai mot cua so reflection memory ngan gon.

### Chay voi OpenAI-compatible API
Can cau hinh cac bien moi truong sau:
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_BASE_URL` (tuy chon, mac dinh `https://api.openai.com/v1`)

Vi du:
```bash
set OPENAI_API_KEY=...
set OPENAI_MODEL=gpt-4o-mini
python run_benchmark.py --dataset path\to\hotpot_100.json --out-dir outputs\real_run --mode openai --limit 100 --shuffle --seed 13 --workers 4 --timeout-s 120 --max-retries 4
```

Neu van bi timeout, giam `--workers` xuong `2` hoac `3` truoc khi tang cao hon.

Neu dung Ollama hoac vLLM voi OpenAI-compatible endpoint, chi can doi `OPENAI_BASE_URL`.

### Chuan bi file HotpotQA dung schema cua lab
```bash
python prepare_hotpotqa.py path\to\hotpot_dev_distractor_v1.json data\hotpot_100.json --limit 100 --shuffle
```

Neu muon giam context de debug nhanh hon:
```bash
python prepare_hotpotqa.py path\to\hotpot_dev_distractor_v1.json data\hotpot_supporting_100.json --limit 100 --shuffle --supporting-only --max-contexts 2
```

## Yeu cau bai lab
1. Thay mock bang LLM that.
2. Chay benchmark tren it nhat 100 mau HotpotQA that.
3. Giu dung format `report.json` va `report.md`.
4. Ghi nhan token usage that tu API neu endpoint co tra `usage`.

## Luu y
- `data/hotpot_mini.json` chi de debug flow, khong du de dat yeu cau 100 mau.
- Mock mode sinh token/latency gia lap de kiem tra logic, khong phai so lieu benchmark that.
- Runtime OpenAI-compatible da doc `usage.total_tokens` neu API co tra ve; neu khong co thi fallback ve estimate.

## Bonus co the lam them
- `adaptive_max_attempts` (da co scaffold)
- `memory_compression` (da co scaffold)
- `plan_then_execute`
- `mini_lats_branching`

## Kiem tra nhanh
```bash
python run_benchmark.py --dataset data/hotpot_mini.json --out-dir outputs/sample_run --mode mock
python autograde.py --report-path outputs/sample_run/report.json
```
