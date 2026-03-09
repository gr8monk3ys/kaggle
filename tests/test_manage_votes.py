import os
import stat
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANAGE = ROOT / "manage.sh"


def test_manage_votes_parses_kernel_csv_rows(tmp_path):
    fake_home = tmp_path / "home"
    kaggle_dir = fake_home / ".kaggle"
    kaggle_dir.mkdir(parents=True)
    (kaggle_dir / "kaggle.json").write_text('{"username":"u","key":"k"}\n', encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir(parents=True)
    fake_kaggle = fake_bin / "kaggle"
    fake_kaggle.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "if [[ \"${1:-}\" == \"kernels\" && \"${2:-}\" == \"list\" ]]; then",
                "  if [[ \" $* \" == *\" --csv \"* ]]; then",
                "    cat <<'CSV'",
                "ref,totalVotes",
                "alice/kernel-one,3",
                "alice/kernel-two,8",
                "CSV",
                "  else",
                "    echo \"kernels list\"",
                "  fi",
                "  exit 0",
                "fi",
                "if [[ \"${1:-}\" == \"datasets\" && \"${2:-}\" == \"list\" ]]; then",
                "  cat <<'TXT'",
                "ref title size",
                "---- ----- ----",
                "alice/ds-one ds-one 1MB",
                "TXT",
                "  exit 0",
                "fi",
                "echo \"unexpected args: $*\" >&2",
                "exit 1",
                "",
            ]
        ),
        encoding="utf-8",
    )
    fake_kaggle.chmod(fake_kaggle.stat().st_mode | stat.S_IEXEC)

    env = os.environ.copy()
    env["HOME"] = str(fake_home)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"

    result = subprocess.run(
        ["bash", str(MANAGE), "votes"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "No kernels found." not in result.stdout
    assert "kernel-one" in result.stdout
    assert "kernel-two" in result.stdout
