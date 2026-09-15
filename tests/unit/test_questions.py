"""Parsing and validating the uploaded questions file."""

import json

import pytest

from app.api.errors import InvalidQuestionsError, QuestionTooLongError, TooManyQuestionsError
from app.api.routes import _parse_questions
from app.config import Settings


def parse(payload: object, settings: Settings) -> list[str]:
    return _parse_questions(json.dumps(payload).encode(), settings)


def test_accepts_a_bare_list(settings: Settings) -> None:
    assert parse(["Which cloud providers?", "What are the SLAs?"], settings) == [
        "Which cloud providers?",
        "What are the SLAs?",
    ]


def test_accepts_an_object_with_a_questions_key(settings: Settings) -> None:
    assert parse({"questions": ["Which cloud providers?"]}, settings) == ["Which cloud providers?"]


def test_strips_whitespace_and_drops_blanks(settings: Settings) -> None:
    assert parse(["  padded  ", "   ", ""], settings) == ["padded"]


def test_rejects_malformed_json(settings: Settings) -> None:
    with pytest.raises(InvalidQuestionsError):
        _parse_questions(b"[oops", settings)


def test_rejects_a_shape_that_is_neither_list_nor_questions_object(settings: Settings) -> None:
    with pytest.raises(InvalidQuestionsError):
        parse({"prompts": ["nope"]}, settings)


def test_rejects_non_string_questions(settings: Settings) -> None:
    with pytest.raises(InvalidQuestionsError, match="must be a string"):
        parse(["fine", 42], settings)


def test_rejects_an_empty_question_list(settings: Settings) -> None:
    with pytest.raises(InvalidQuestionsError):
        parse([], settings)


def test_enforces_the_question_count_limit(settings: Settings) -> None:
    settings.max_questions = 3
    with pytest.raises(TooManyQuestionsError) as caught:
        parse([f"question {index}?" for index in range(4)], settings)

    assert caught.value.code == "TOO_MANY_QUESTIONS"
    assert caught.value.status_code == 422
    assert "received 4" in caught.value.message


def test_enforces_the_question_length_limit(settings: Settings) -> None:
    settings.max_question_chars = 10
    with pytest.raises(QuestionTooLongError) as caught:
        parse(["x" * 11], settings)

    assert caught.value.code == "QUESTION_TOO_LONG"
