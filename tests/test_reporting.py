from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report
from src.reflexion_lab.runtime import MockRuntime
from src.reflexion_lab.utils import load_dataset


def test_report_contains_overall_and_agent_failure_breakdowns():
    examples = load_dataset("data/hotpot_mini.json")
    runtime = MockRuntime()
    records = [ReActAgent(runtime=runtime).run(example) for example in examples]
    records += [
        ReflexionAgent(max_attempts=3, runtime=runtime).run(example) for example in examples
    ]

    report = build_report(records, dataset_name="hotpot_mini.json", mode="mock")

    assert "overall" in report.failure_modes
    assert "react" in report.failure_modes
    assert "reflexion" in report.failure_modes
    assert report.meta["mode"] == "mock"
    assert "mock_mode_for_autograding" in report.extensions
