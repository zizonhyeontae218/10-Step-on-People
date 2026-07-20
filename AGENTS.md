# AGENTS.md

## 프로젝트 목적

이 저장소는 `10-step-on-people`이라는 학교 발표용 한국어 문장 이어쓰기 MVP다. 동일한 소형 decoder-only Transformer를 FP baseline과 BitNet b1.58 방식으로 직접 학습하고, 짧은 prompt 뒤의 token을 생성한다. 채팅, 지식 질답, 실제 1.58-bit packing은 범위 밖이다. 이름의 `10-step`은 브랜드이며 실제 학습 step 수가 아니다.

## 작업 시작 전

1. `Goal.md`와 `FileTree.md`를 읽는다.
2. 파일 구조가 바뀌면 같은 작업에서 `FileTree.md`를 갱신한다.
3. 데이터와 checkpoint는 Git에 추가하지 않는다.

## 고정 인터페이스

```powershell
python -m onebit_llm.prepare
python -m onebit_llm.train --model all --config configs/demo.yaml
python -m onebit_llm.generate --model bitnet --prompt "오늘 학교에서"
python -m onebit_llm.evaluate
python app.py
pytest
```

## 구현 불변 조건

- dataset과 tokenizer revision은 `prepare.py`의 고정 값과 README를 함께 갱신하지 않는 한 변경하지 않는다.
- baseline과 BitNet은 linear 종류 외에는 같은 모델 구조와 학습 조건을 사용한다.
- BitNet은 full-precision master weight, ternary forward code `{-1, 0, 1}`, token별 int8 activation fake quantization, STE를 유지한다.
- embedding과 normalization은 full precision이다.
- pad token `0`은 loss에서 제외하고 서로 다른 source row를 하나의 학습 window로 연결하지 않는다.
- `--model all`은 BitNet 완료 step 수와 baseline step 수를 같게 유지한다.
- CUDA preflight는 모델 구조를 바꾸지 않고 micro-batch와 accumulation만 조정하며 양 모델에 같은 runtime 설정을 적용한다.
- 전용 packed storage/kernel이 없으므로 checkpoint 크기나 속도 향상을 1.58-bit 이론값과 동일시하지 않는다.
- seed, 설정, dataset 출처와 제한을 결과 및 문서에서 보존한다.

## 개발과 검증

- 새 설정은 dataclass와 YAML을 통해 전달하고 모델·데이터·학습·생성 책임을 분리한다.
- CPU에서 모든 테스트와 smoke 학습이 동작해야 하며 CUDA가 있으면 학습이 자동으로 FP16을 사용한다.
- 전체 110분 학습은 명시적으로 요청받지 않는 한 실행하지 않는다. 구현 검증에는 `configs/smoke.yaml`을 사용한다.
- 변경 후 관련 테스트를 실행한다. 최소 검증 대상은 restricted pickle, row split, causal mask, ternary/int8 code, STE gradient, 양 모델 smoke 학습, 생성 제한, checkpoint 오류다.
- 완료 보고에는 실행한 명령, 통과한 테스트, full demo 학습 여부를 구분해 적는다.
