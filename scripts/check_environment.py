"""Confere o ambiente do projeto HAR (T04).

Verifica:
  1. versao do Python (3.11 ou 3.12);
  2. bibliotecas necessarias e suas versoes;
  3. configuracao em configs/default.yaml (lida via har.config.load_config);
  4. arquivos do dataset UCI HAR no data_dir definido na configuracao;
  5. dimensoes esperadas (561 colunas, 7.352 e 2.947 linhas);
  6. ausencia de participantes em comum entre treino e teste;
  7. se n_splits cabe no numero de participantes do treino (GroupKFold).

Uso (na raiz do repositorio):
    uv run python scripts/check_environment.py
    # ou, com venv ativado: python scripts/check_environment.py

Sai com codigo 0 se tudo estiver certo e 1 se houver alguma falha.
"""

from __future__ import annotations

import sys
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Permite importar har.config mesmo se o pacote nao foi instalado com -e.
sys.path.insert(0, str(REPO_ROOT / "src"))

SUPPORTED_PYTHON = ((3, 11), (3, 12))

# (nome do modulo para importar, nome do pacote no pip)
# Obrigatorios: o que o projeto ja declara no pyproject.toml / uv.lock.
REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("joblib", "joblib"),
    ("yaml", "pyyaml"),
    ("pytest", "pytest"),
]
# Opcionais: previstos no plano (EDA, figuras, qualidade), mas ainda nao
# declarados como dependencia. Geram so aviso. Se o grupo decidir adiciona-los,
# basta mover para REQUIRED_PACKAGES.
OPTIONAL_PACKAGES = [
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("jupyterlab", "jupyterlab"),
    ("ruff", "ruff"),
]

REQUIRED_FILES = [
    "features.txt",
    "activity_labels.txt",
    "train/X_train.txt",
    "train/y_train.txt",
    "train/subject_train.txt",
    "test/X_test.txt",
    "test/y_test.txt",
    "test/subject_test.txt",
]

EXPECTED_N_FEATURES = 561
EXPECTED_TRAIN_ROWS = 7352
EXPECTED_TEST_ROWS = 2947


class Report:
    """Acumula os resultados e imprime de forma padronizada."""

    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def ok(self, message: str) -> None:
        print(f"[OK]    {message}")

    def warn(self, message: str) -> None:
        self.warnings += 1
        print(f"[AVISO] {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"[FALHA] {message}")


def check_python(report: Report) -> None:
    print("\n== Python ==")
    current = sys.version_info[:2]
    label = f"{current[0]}.{current[1]}.{sys.version_info[2]}"
    if current in SUPPORTED_PYTHON:
        report.ok(f"Python {label}")
    else:
        report.fail(f"Python {label} nao suportado; use 3.11 ou 3.12")
    print(f"        executavel: {sys.executable}")


def check_packages(report: Report) -> None:
    print("\n== Bibliotecas ==")
    for module_name, pip_name in REQUIRED_PACKAGES:
        try:
            import_module(module_name)
            report.ok(f"{pip_name} {version(pip_name)}")
        except ImportError:
            report.fail(f"{pip_name} nao instalado (uv sync ou pip install -r requirements.txt)")
        except PackageNotFoundError:
            report.warn(f"{pip_name} importa, mas a versao nao foi encontrada")
    for module_name, pip_name in OPTIONAL_PACKAGES:
        try:
            import_module(module_name)
            report.ok(f"{pip_name} {version(pip_name)} (opcional)")
        except ImportError:
            report.warn(f"{pip_name} nao instalado (opcional)")
        except PackageNotFoundError:
            report.warn(f"{pip_name} importa, mas a versao nao foi encontrada")


def load_settings(report: Report) -> dict | None:
    """Le configs/default.yaml usando o mesmo codigo do projeto (har.config)."""
    print("\n== Configuracao ==")
    try:
        from har.config import load_config

        config = load_config()
    except Exception as exc:  # noqa: BLE001 - queremos reportar qualquer erro
        report.fail(f"nao foi possivel carregar configs/default.yaml: {exc}")
        return None

    report.ok("configs/default.yaml carregado via har.config.load_config")
    print(f"        data_dir: {config['data_dir']}")
    print(
        f"        n_splits: {config['n_splits']} | "
        f"random_state: {config['random_state']} | "
        f"selection_metric: {config['selection_metric']}"
    )
    return config


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return sum(1 for line in handle if line.strip())


def first_row_width(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                return len(line.split())
    return 0


def read_ids(path: Path) -> set[int]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return {int(line) for line in handle if line.strip()}


def check_data_files(report: Report, data_dir: Path) -> bool:
    """Retorna True se todos os arquivos existem."""
    print("\n== Arquivos do dataset ==")
    if not data_dir.exists():
        report.fail(
            f"pasta {data_dir} nao encontrada. Baixe o dataset "
            "(archive.ics.uci.edu/dataset/240) e extraia nela; o zip pode "
            "conter outro zip dentro."
        )
        return False

    all_present = True
    for relative in REQUIRED_FILES:
        if (data_dir / relative).is_file():
            report.ok(relative)
        else:
            report.fail(f"{relative} nao encontrado")
            all_present = False
    return all_present


def check_shapes_and_subjects(report: Report, data_dir: Path, n_splits: int) -> None:
    print("\n== Dimensoes e participantes ==")
    x_train = data_dir / "train" / "X_train.txt"
    x_test = data_dir / "test" / "X_test.txt"
    y_train = data_dir / "train" / "y_train.txt"
    y_test = data_dir / "test" / "y_test.txt"
    s_train = data_dir / "train" / "subject_train.txt"
    s_test = data_dir / "test" / "subject_test.txt"

    expected = [
        ("X_train", x_train, EXPECTED_TRAIN_ROWS),
        ("y_train", y_train, EXPECTED_TRAIN_ROWS),
        ("subject_train", s_train, EXPECTED_TRAIN_ROWS),
        ("X_test", x_test, EXPECTED_TEST_ROWS),
        ("y_test", y_test, EXPECTED_TEST_ROWS),
        ("subject_test", s_test, EXPECTED_TEST_ROWS),
    ]
    for name, path, rows in expected:
        found = count_lines(path)
        if found == rows:
            report.ok(f"{name}: {found} linhas")
        else:
            report.fail(f"{name}: {found} linhas (esperado {rows})")

    for name, path in (("X_train", x_train), ("X_test", x_test)):
        width = first_row_width(path)
        if width == EXPECTED_N_FEATURES:
            report.ok(f"{name}: {width} colunas")
        else:
            report.fail(f"{name}: {width} colunas (esperado {EXPECTED_N_FEATURES})")

    try:
        train_ids = read_ids(s_train)
        test_ids = read_ids(s_test)
    except ValueError:
        report.fail("subject_train/subject_test com valores nao numericos")
        return

    print(f"        participantes no treino: {len(train_ids)}; no teste: {len(test_ids)}")
    overlap = train_ids & test_ids
    if overlap:
        report.fail(f"participantes em comum entre treino e teste: {sorted(overlap)}")
    else:
        report.ok("nenhum participante em comum entre treino e teste")

    if n_splits <= len(train_ids):
        report.ok(f"n_splits={n_splits} cabe nos {len(train_ids)} participantes do treino")
    else:
        report.fail(
            f"n_splits={n_splits} maior que o numero de participantes do treino "
            f"({len(train_ids)}); o GroupKFold nao funciona assim"
        )


def main() -> int:
    print(f"Repositorio: {REPO_ROOT}")
    report = Report()

    check_python(report)
    check_packages(report)
    config = load_settings(report)
    if config is not None and check_data_files(report, Path(config["data_dir"])):
        check_shapes_and_subjects(report, Path(config["data_dir"]), int(config["n_splits"]))

    print("\n== Resumo ==")
    print(f"        falhas: {report.failures} | avisos: {report.warnings}")
    if report.failures:
        print("Ambiente com problemas. Corrija as falhas acima; se travar, "
              "peca ajuda ao grupo no mesmo dia.")
        return 1
    print("Ambiente pronto.")
    return 0


if __name__ == "__main__":
    sys.exit(main())