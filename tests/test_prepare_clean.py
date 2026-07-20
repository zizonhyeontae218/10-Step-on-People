from __future__ import annotations

from tokenizers import Tokenizer

from onebit_llm.prepare_clean import clean_document, token_rows


def test_clean_document_removes_known_contamination() -> None:
    assert clean_document("게임 파일", "가" * 200) is None
    assert clean_document("일상", "https://youtube.com/watch?v=x " + "가" * 200) is None
    assert clean_document("일상", "[[파일:photo.png]] " + "가" * 200) is None


def test_clean_document_cuts_reference_tail() -> None:
    body = "학교에서 친구들과 함께 공부하며 즐거운 시간을 보냈습니다. " * 8
    cleaned = clean_document("학교생활", body + "\n외부 링크\nhttps://example.com")
    assert cleaned is not None
    assert "외부 링크" not in cleaned
    assert "http" not in cleaned


def test_token_rows_preserve_document_boundary(tmp_path) -> None:
    tokenizer = Tokenizer.from_file("data/tokenizer/tokenizer.json")
    rows = list(token_rows(tokenizer, "오늘 학교에서 친구들과 함께 공부했습니다. " * 50))
    assert rows
    assert all(row.shape == (512,) for row in rows)
    assert rows[-1][rows[-1] != 0][-1] == 2
