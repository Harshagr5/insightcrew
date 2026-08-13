.PHONY: eval eval-fast eval-fake test

MODEL_FAST   ?= nvidia_nim/meta/llama-3.1-8b-instruct
MODEL_STRONG ?= nvidia_nim/meta/llama-3.3-70b-instruct

# Full evaluation: both models, including critic behaviour. Needs NVIDIA_API_KEY.
eval:
	@rm -f evals/RESULTS.md
	python evals/run_eval.py --model $(MODEL_FAST)   --with-critic
	python evals/run_eval.py --model $(MODEL_STRONG) --with-critic
	@echo "Wrote evals/RESULTS.md"

# Just the fast/free model.
eval-fast:
	@rm -f evals/RESULTS.md
	python evals/run_eval.py --model $(MODEL_FAST) --with-critic

# No LLM: oracle planner, sanity-checks the scorer.
eval-fake:
	python evals/run_eval.py --fake

test:
	pytest -q
