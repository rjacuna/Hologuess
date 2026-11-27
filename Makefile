.PHONY: test test-upstream

test:
	sage -python -m unittest discover -s tests -v

test-upstream:
	sage -python -m unittest discover -s tests -p 'test_upstream*.py' -v
