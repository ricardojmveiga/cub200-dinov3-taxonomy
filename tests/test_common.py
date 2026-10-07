"""Cross-platform helpers: paths, thread policy, device pick, labels."""
from reprocub import common


def test_paths_exist():
    assert common.DATA.is_dir()
    assert common.LABELS_JSON.exists()
    assert common.FIGURES.is_dir()


def test_threads_cores_minus_one():
    import os
    assert common.n_threads() == max(1, (os.cpu_count() or 2) - 1)
    assert common.configure_cpu_threads() >= 1


def test_device_is_valid():
    assert common.get_device() in ("cuda", "mps", "cpu")
    assert common.get_device("cpu") == "cpu"                 # forced CPU always works


def test_labels_counts():
    labels, oracle, n = common.load_labels()
    assert n == 11788
    assert oracle == {"order": 13, "family": 37, "genus": 121, "species": 200}
    for r in common.RANKS:
        assert len(labels[r]) == 11788


def test_env_summary():
    e = common.env_summary()
    assert e["numpy"] and e["python"] and e["threads"] >= 1


def test_configure_cpu_threads_sets_env_and_returns_n():
    import os
    n = common.configure_cpu_threads()                       # default: cores-1
    assert n == common.n_threads()
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert os.environ[var] == str(n)                     # env pinned to cores-1


def test_device_label_is_descriptive():
    assert "CPU" in common.device_label("cpu")               # always resolvable
    assert common.device_label("mps")                        # non-empty on any host


def test_timed_records_elapsed_seconds():
    sink = {}
    with common.timed("unit", sink):
        pass
    assert "unit" in sink and isinstance(sink["unit"], float) and sink["unit"] >= 0
