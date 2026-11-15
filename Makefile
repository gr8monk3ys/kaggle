.PHONY: verify

# Runs exactly what the gating CI job (ci.yml -> quality) runs.
verify:
	bash scripts/verify.sh
