"""What a full run does, per configuration.

The stage graph had no test, which is why it went unnoticed that
``python -m src.pipeline --config configs/wide.yaml`` would train two networks and
seventeen call type models the wide set was never meant to have, then rewrite its
committed report with sections that are not in it. Hours of GPU and a broken
reproduce job, from the command the README tells you to run.

A configuration now says which models it trains. These tests hold that line.
"""

from __future__ import annotations

import pytest

from src.config import PROJECT_ROOT, load_config
from src.config.sections import ConfigError, PipelineConfig
from src.pipeline import build_stages

CONFIGS = {"base_10k": "base.yaml", "base_5k": "base_5k.yaml", "wide_10k": "wide.yaml"}

NETWORKS = {"cnn", "cnn_small"}


def stages(filename: str) -> list[str]:
    cfg = load_config(f"configs/{filename}")
    return [stage.name for stage in build_stages(cfg, f"configs/{filename}", skip_download=True)]


def test_a_config_that_declares_nothing_trains_everything():
    """base.yaml has no pipeline section, so its stage list must not have moved."""
    assert load_config("configs/base.yaml").pipeline.models is None
    assert set(stages("base.yaml")) >= {*NETWORKS, "calltypes_spermwhale", "calltypes_killerwhale"}


def test_the_wide_set_trains_trees_and_nothing_else():
    built = stages("wide.yaml")

    assert "trees" in built
    assert not NETWORKS & set(built), "eleven classes of network is hours of GPU nobody asked for"
    assert not [name for name in built if name.startswith("calltypes_")]
    assert "explain" not in built, "there is no checkpoint to explain"


def test_the_narrow_band_trains_only_what_it_has_results_for():
    built = stages("base_5k.yaml")

    assert "cnn_small" in built
    assert "cnn" not in built
    assert not [name for name in built if name.startswith("calltypes_")]


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_a_full_run_would_rebuild_the_report_and_touch_nothing_else(name):
    """Every committed configuration is finished, so only the report should rerun.

    The report is cheap and always reruns by design. Anything else appearing here
    means a stage would fit a model whose results are not committed, and the next
    report build would then disagree with the one in the repository.
    """
    filename = CONFIGS[name]
    cfg = load_config(f"configs/{filename}")
    if not (cfg.paths.reports / name).exists():
        pytest.skip(
            f"no results for {name}; run python -m src.pipeline --config configs/{filename}"
        )

    pending = [
        stage.name
        for stage in build_stages(cfg, f"configs/{filename}", skip_download=True)
        if not stage.done()
    ]
    assert pending == ["report"], f"{name} would also run {[p for p in pending if p != 'report']}"


def test_an_unknown_key_in_the_section_is_refused(tmp_path):
    """A typo here would silently restore the behaviour the section exists to stop."""
    source = (PROJECT_ROOT / "configs" / "wide.yaml").read_text(encoding="utf-8")
    broken = tmp_path / "broken.yaml"
    broken.write_text(source.replace("  call_types: []", "  call_type: []"), encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown keys in pipeline"):
        load_config(broken)


def test_a_declared_list_has_to_be_a_list(tmp_path):
    source = (PROJECT_ROOT / "configs" / "wide.yaml").read_text(encoding="utf-8")
    broken = tmp_path / "broken.yaml"
    broken.write_text(source.replace("  call_types: []", "  call_types: yes"), encoding="utf-8")

    with pytest.raises(ConfigError, match="must be a list"):
        load_config(broken)


def test_the_default_allows_everything():
    empty = PipelineConfig()

    assert empty.allows("cnn")
    assert empty.call_type_species(("SpermWhale",)) == ("SpermWhale",)
    assert not PipelineConfig(models=("xgboost",)).allows("cnn")
    assert PipelineConfig(call_types=()).call_type_species(("SpermWhale",)) == ()
