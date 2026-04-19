"""Step definitions for features/data/vocab_cache.feature."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pyarrow as pa  # type: ignore[import]
import pyarrow.parquet as pq  # type: ignore[import]
from behave import given, then, when  # type: ignore[import]


TEMP_DIR = Path("/tmp/aegir_vocab_cache_bdd")


@given("a synthetic GitTables dir with {n:d} parquets and DBpedia annotations")
def step_given_synthetic_gittables(context, n):
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    TEMP_DIR.mkdir(parents=True)
    for i in range(n):
        schema = pa.schema(
            [("col_a", pa.string()), ("col_b", pa.string())],
            metadata={
                b"gittables": json.dumps(
                    {
                        "dbpedia_semantic_column_types": {
                            "col_a": {
                                "id": f"http://dbpedia.org/ontology/Type{i}_A"
                            },
                            "col_b": {
                                "id": f"http://dbpedia.org/ontology/Type{i}_B"
                            },
                        }
                    }
                ).encode()
            },
        )
        table = pa.table({"col_a": ["x"] * 3, "col_b": ["y"] * 3}, schema=schema)
        pq.write_table(table, TEMP_DIR / f"t{i}.parquet")
    context.data_dir = TEMP_DIR


@given("the env var AEGIR_GITTABLES_VOCAB_CACHE_DISABLE=1")
def step_given_disable(context):
    os.environ["AEGIR_GITTABLES_VOCAB_CACHE_DISABLE"] = "1"


@when("I load the vocab twice in the same process")
def step_when_load_twice(context):
    from aegir.data.table_dataset import _load_or_discover_gittables_labels

    context.v1 = _load_or_discover_gittables_labels(
        context.data_dir, "dbpedia_semantic_column_types", "dbpedia"
    )
    context.v2 = _load_or_discover_gittables_labels(
        context.data_dir, "dbpedia_semantic_column_types", "dbpedia"
    )


@when("I load the vocab once")
def step_when_load_once(context):
    from aegir.data.table_dataset import _load_or_discover_gittables_labels

    context.v1 = _load_or_discover_gittables_labels(
        context.data_dir, "dbpedia_semantic_column_types", "dbpedia"
    )
    os.environ.pop("AEGIR_GITTABLES_VOCAB_CACHE_DISABLE", None)


@then("the second call returns an identical vocab")
def step_then_identical(context):
    assert context.v1 == context.v2, "vocabs differ across calls"


@then("a .aegir_vocab_dbpedia.json sidecar exists next to the data")
def step_then_sidecar_exists(context):
    sidecar = context.data_dir / ".aegir_vocab_dbpedia.json"
    assert sidecar.exists(), f"sidecar not written at {sidecar}"


@then("no .aegir_vocab_dbpedia.json sidecar is written")
def step_then_no_sidecar(context):
    sidecar = context.data_dir / ".aegir_vocab_dbpedia.json"
    assert not sidecar.exists(), f"sidecar unexpectedly written at {sidecar}"
