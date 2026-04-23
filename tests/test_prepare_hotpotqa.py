from prepare_hotpotqa import convert_hotpotqa_record, convert_hotpotqa_records


def test_convert_hotpotqa_record_preserves_core_fields():
    raw = {
        "_id": "abc123",
        "level": "hard",
        "question": "What river flows through the city where Ada Lovelace was born?",
        "answer": "River Thames",
        "supporting_facts": [["Ada Lovelace", 0], ["London", 0]],
        "context": [
            ["Ada Lovelace", ["Ada Lovelace was born in London, England."]],
            ["London", ["London is crossed by the River Thames."]],
        ],
    }

    converted = convert_hotpotqa_record(raw, index=1)

    assert converted is not None
    assert converted["qid"] == "abc123"
    assert converted["difficulty"] == "hard"
    assert converted["gold_answer"] == "River Thames"
    assert len(converted["context"]) == 2


def test_convert_hotpotqa_records_can_filter_supporting_context():
    raw_records = [
        {
            "_id": "abc123",
            "level": "medium",
            "question": "Who wrote The Hobbit?",
            "answer": "J. R. R. Tolkien",
            "supporting_facts": [["J. R. R. Tolkien", 0]],
            "context": [
                ["J. R. R. Tolkien", ["J. R. R. Tolkien wrote The Hobbit."]],
                ["Noise", ["This paragraph should be removed when supporting_only is true."]],
            ],
        }
    ]

    converted = convert_hotpotqa_records(raw_records, supporting_only=True)

    assert len(converted) == 1
    assert len(converted[0]["context"]) == 1
    assert converted[0]["context"][0]["title"] == "J. R. R. Tolkien"
