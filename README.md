[ILCX 4H+ grade](https://github.com/zizonhyeontae218/ILCX_H-grade-system)

# 10-step-on-people

학교 발표를 위한 작은 한국어 1.58-bit 문장 이어쓰기 모델 패밀리입니다. `오늘 학교에서`처럼 문장 앞부분을 입력하면 그 뒤를 이어 생성합니다. 대화나 지식 질답 모델이 아닙니다.

`10-step-on-people`의 `10-step`은 프로젝트 브랜드이며 checkpoint의 실제 학습 step 수를 뜻하지 않습니다.

## 모델 버전

- **10-step-on-people**: 1,249만 token의 초기 실험 corpus로 학습한 발표용 원본입니다. 양 모델 모두 1,500 step을 학습했습니다.
- **9-step-on-people**: 링크·위키 문법·게임/유튜브 중심 문서를 제거한 한국어 위키백과 4,096만 token으로 다시 학습한 후속 실험입니다. 양 모델 모두 5,662 step을 학습했습니다.

학습 데이터와 checkpoint는 Git에 포함하지 않습니다. 재현용 소스와 설정은 저장소에 두고, 완성된 `best.pt`와 평가 자료는 GitHub Releases에서 버전별로 제공합니다.

| 버전 | 모델 | best validation loss | perplexity |
|---|---|---:|---:|
| 10-step | BitNet b1.58 | 6.6938 | 807.41 |
| 10-step | FP baseline | 6.5924 | 729.49 |
| 9-step | BitNet b1.58 | 5.3738 | 215.69 |
| 9-step | FP baseline | 5.2311 | 187.00 |

같은 구조를 두 방식으로 처음부터 학습합니다.

- `baseline`: 일반 full-precision `Linear`
- `bitnet`: forward에서 ternary weight code `{-1, 0, 1}`와 int8 activation fake quantization을 사용하는 BitNet b1.58 스타일 `BitLinear`

> “1-bit”는 관용적 이름입니다. 세 weight 상태의 정보량은 약 1.58-bit입니다. 이 프로젝트는 일반 PyTorch fake quantization을 사용하므로 실제 checkpoint가 1.58-bit로 packing되거나 자동으로 빨라지지는 않습니다.

## 설치

Python 3.11 환경에서 설치합니다. 이 PC의 MX450과 571.96 driver에는 CUDA 12.6 wheel이 안전합니다. 먼저 GPU용 PyTorch를 설치한 뒤 프로젝트를 설치합니다. 다른 PC에서는 [PyTorch 공식 설치 안내](https://pytorch.org/get-started/locally/)에서 맞는 wheel을 선택하세요.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install --force-reinstall torch==2.13.0+cu126 --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[app,dev]"
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

위 확인 결과의 첫 값이 `True`여야 본 학습에서 GPU를 사용합니다. CPU 검증만 필요하면 PyTorch 설치 URL을 `https://download.pytorch.org/whl/cpu`로 바꿀 수 있습니다.

## 빠른 실행

이미 준비된 이 PC에서 발표할 때는 루트의 `RUN_DEMO.bat`를 더블클릭하면 됩니다. 학습된 checkpoint를 확인한 뒤 Gradio 화면과 웹 브라우저를 자동으로 엽니다. 학교 제출용 탐구 보고서는 `reports/10-step-on-people_탐구보고서.docx`입니다.

```powershell
# 34.7MB token subset과 tokenizer 준비
python -m onebit_llm.prepare

# CUDA에서는 VRAM preflight 후 BitNet 최대 55분, baseline은 같은 step 수
python -m onebit_llm.train --model all --config configs/demo.yaml

# 고정 prompt 정량·정성 비교 보고서
python -m onebit_llm.evaluate

# 터미널에서 이어쓰기
python -m onebit_llm.generate --model bitnet --prompt "오늘 학교에서"

# 로컬 웹 화면
python app.py
```

구현만 빠르게 확인할 때는 `configs/smoke.yaml`을 사용합니다.

```powershell
python -m onebit_llm.train --model all --config configs/smoke.yaml
pytest
```

## 데이터

[oz1115/korean-pretraining-corpus-ko](https://huggingface.co/datasets/oz1115/korean-pretraining-corpus-ko)의 원래 validation 파일만 사용합니다. 파일은 24,396개의 512-token row, 약 1,249만 token으로 구성되어 있습니다. 이 프로젝트에서는 학교 발표용 빠른 실험을 위해 이를 seed 42로 다시 90/10 분할합니다.

- 데이터 revision: `324488e1befab3a9b4eac7571bf1557ef4a4eeec`
- tokenizer: [oz1115/korean-gpt-150m-ko](https://huggingface.co/oz1115/korean-gpt-150m-ko)
- tokenizer revision: `669560d7d1de8c3213e43a343e577a80ad6cb7ee`
- dataset card 표기 license: MIT

원 dataset card의 원문 출처 설명은 “한국어 위키백과 및 공개 한국어 텍스트” 수준으로 제한적입니다. 따라서 이 저장소는 데이터를 재배포하지 않으며 교육·시연 용도로만 사용합니다.

## 출력과 한계

학습 결과는 `artifacts/checkpoints/{bitnet,baseline}/`에, JSONL metric은 `artifacts/metrics/`에 저장됩니다. BitNet을 먼저 학습하고 실제 완료한 optimizer step 수만큼 baseline을 학습하여 비교 조건을 맞춥니다.

CUDA 학습을 시작하면 BitNet의 실제 forward/backward/optimizer step으로 micro-batch `16 → 8 → 4 → 2 → 1`을 시험합니다. 선택된 값에 맞춰 gradient accumulation을 조정해 effective batch 32를 유지하고 `artifacts/run_profile.json`에 장치와 peak VRAM을 기록합니다. batch 1도 실패하면 모델을 몰래 줄이지 않고 명확히 종료합니다.

`python -m onebit_llm.evaluate`는 양쪽 best checkpoint로 고정 prompt 4개를 생성하고 `artifacts/evaluation/results.json`과 `report.md`를 만듭니다. 보고서에는 step 0 대비 validation loss 개선 여부도 포함됩니다.

### 현재 발표용 학습 결과

MX450에서 양 모델을 각각 1,500 optimizer step 학습했습니다. 이 완료 run은 공정 비교를 위해 micro-batch 8, accumulation 4를 함께 사용했습니다.

| 모델 | 초기 validation loss | best loss | perplexity | 학습 시간 |
|---|---:|---:|---:|---:|
| BitNet b1.58 | 10.3986 | 6.6938 | 807.41 | 31분 16초 |
| FP baseline | 10.4161 | 6.5924 | 729.49 | 28분 22초 |

BitNet의 ternary 0 비율은 최종 약 31.9%였습니다. 일반 PyTorch 구현에서는 BitNet 생성이 baseline보다 느렸으며, 이는 전용 저비트 kernel을 사용하지 않았기 때문입니다. 생성 예시는 `artifacts/evaluation/report.md`에서 확인할 수 있습니다.

다음 새 run의 기본값은 micro-batch 16, accumulation 2입니다. 동일 MX450 실측 preflight에서 peak allocated VRAM 약 980MB로 통과했지만 기존 비교 run을 다시 쓰지는 않았습니다.

약 19M parameter의 작은 모델을 제한된 corpus와 시간으로 학습하므로 자연스러운 짧은 한국어 조각은 목표로 할 수 있지만 사실성, 일관성, 유해성 제어는 보장하지 않습니다. embedding과 normalization은 full precision이며 전용 저비트 kernel은 사용하지 않습니다.

## 테스트

```powershell
pytest
```

테스트는 데이터 검증, causal mask, ternary/int8 quantization, gradient, 짧은 학습, 생성 제한 및 checkpoint 오류를 확인합니다.
