syntax-only:
	# Just check syntax.
	python3 -m py_compile python_snippets.py


lint:
	pylint python_snippets.py
	pyflakes python_snippets.py

run:
	python3 python_snippets.py
