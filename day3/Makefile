all:
	@$(MAKE) --no-print-directory clean > /dev/null
	@$(MAKE) --no-print-directory test
	@$(MAKE) --no-print-directory clean > /dev/null

test:
	pytest -rA -vv

clean:
	rm -rf .pytest_cache
	rm -rf */.pytest_cache
	rm -rf __pycache__
	rm -rf */__pycache__
