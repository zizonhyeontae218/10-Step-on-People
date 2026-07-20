# 10-step-on-people 목표

## 최종 결과

`10-step-on-people`이라는 이름으로, `오늘 학교에서` 같은 한국어 문장 앞부분을 입력하면 baseline 또는 BitNet b1.58 소형 모델이 기본 32 token을 이어서 생성하는 학교 발표용 프로그램을 만든다. 이름의 `10-step`은 브랜드다.

## 확정 범위

- Hugging Face `oz1115/korean-pretraining-corpus-ko`의 34.7MB `val.pkl`만 내려받아 교육용 train/validation subset으로 재분할한다.
- 기존 32K BPE tokenizer를 그대로 사용한다.
- 약 19M parameter의 동일한 decoder-only Transformer 두 개를 처음부터 학습한다.
  - FP baseline
  - ternary weight와 int8 activation fake quantization을 사용하는 BitNet b1.58
- CLI와 Gradio 단일 문장 이어쓰기 화면을 제공한다.
- BitNet을 최대 55분 먼저 학습하고 baseline을 같은 optimizer step 수만큼 학습한다.
- validation loss, perplexity, parameter 수, ternary zero 비율을 JSONL로 기록한다.
- MX450 VRAM을 실측해 effective batch를 보존하는 adaptive micro-batch를 사용한다.
- 양쪽 best checkpoint의 고정 prompt 결과를 JSON과 Markdown 보고서로 만든다.

## 성공 기준

- 데이터 준비 명령이 고정 revision을 내려받아 `uint16` memory-map과 90/10 row split을 만든다.
- 단위 테스트가 ternary code, STE gradient, causal mask와 데이터 경계를 검증한다.
- smoke 설정으로 양 모델이 동일 step을 완료하고 metric 및 `best.pt`/`last.pt`를 만든다.
- CLI와 Gradio가 checkpoint를 불러와 1–64 token을 생성한다.
- 데이터, 모델 제약, “1-bit” 명칭과 실제 fake quantization의 차이를 README에 밝힌다.

## 제외 범위

- 채팅, 질답, 검색, 외부 API
- 공식 2B 모델 또는 수십억 parameter 사전학습
- 실제 ternary bit packing과 전용 저비트 kernel
- 사실성·안전성·높은 문장 품질 보장
- 배포 및 공개 hosting

## 현재 상태

- [x] 프로젝트와 의존성 구성
- [x] 고정 dataset/tokenizer 준비 파이프라인
- [x] baseline과 W1.58A8 BitNet 모델
- [x] 시간 제한·동일 step 학습 및 metric/checkpoint
- [x] 생성 CLI와 Gradio 화면
- [x] 단위 테스트 및 실제 dataset smoke 학습
- [x] 모델 패밀리 브랜딩, CUDA preflight, 평가 보고서 구현
- [x] CUDA 12.6 wheel 및 MX450 preflight 실측
- [x] 양 모델 1,500-step 발표용 본 학습
- [x] 본 학습 고정 prompt 평가 보고서
- [ ] 발표 리허설 및 결과 해석 정리
