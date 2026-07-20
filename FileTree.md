# File tree

```text
1BIT-LLM/
├── configs/
│   ├── demo.yaml              # 최대 110분 발표용 설정
│   ├── 9-step-on-people.yaml  # 정제 corpus 장시간 후속 학습 설정
│   └── smoke.yaml             # 빠른 구현 검증 설정
├── src/onebit_llm/
│   ├── __init__.py
│   ├── checkpoint.py          # 안전한 checkpoint 저장/로드
│   ├── config.py              # YAML-backed dataclass 설정
│   ├── data.py                # memory-map causal-LM dataset
│   ├── evaluate.py            # 고정 prompt 비교 보고서
│   ├── generate.py            # 생성 CLI
│   ├── generation.py          # sampling과 tokenizer 연결
│   ├── model.py               # 공통 decoder-only Transformer
│   ├── prepare.py             # 고정 dataset/tokenizer 준비
│   ├── prepare_clean.py       # 정제된 위키백과 후속 corpus 준비
│   ├── preflight.py           # CUDA adaptive micro-batch 실측
│   ├── quantization.py        # ternary/int8 fake quantization
│   ├── train.py               # 공정 비교 학습과 평가
│   ├── overnight.py           # 9-step 데이터·학습·평가 자동 실행
│   ├── keep_awake.py          # 야간 학습 중 Windows 절전 방지
│   └── web.py                 # Gradio Blocks UI
├── tests/
│   ├── test_generation.py
│   ├── test_evaluate.py
│   ├── test_model.py
│   ├── test_preflight.py
│   ├── test_prepare_and_data.py
│   ├── test_prepare_clean.py
│   ├── test_quantization.py
│   ├── test_train_smoke.py
│   └── test_web.py
├── scripts/
│   └── build_school_report.py # DOCX 보고서 생성과 형식 감사
├── reports/
│   └── 10-step-on-people_탐구보고서.docx # 학교 제출용 탐구 보고서
├── data/                      # 생성됨, Git 제외
│   ├── processed/
│   ├── raw/
│   └── tokenizer/
├── artifacts/                 # 생성됨, Git 제외
│   ├── checkpoints/           # 본 학습 best/last checkpoint
│   ├── evaluation/            # results.json과 report.md
│   ├── metrics/               # step별 JSONL
│   ├── smoke/                 # 기존 10-step 검증 결과
│   ├── run_profile.json       # 완료 run의 CUDA/batch 실측
│   └── summary.json           # 양 모델 학습 요약
├── .gitignore
├── .gitattributes             # Windows 배치 파일 CRLF 줄바꿈 고정
├── AGENTS.md
├── FileTree.md
├── Goal.md
├── README.md
├── RUN_DEMO.bat              # 더블클릭 발표용 데모 실행기
├── app.py
└── pyproject.toml
```
