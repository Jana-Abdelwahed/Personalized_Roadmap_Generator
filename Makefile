.PHONY: install run test lint

install:
	pip install -r requirements.txt

run:
	streamlit run src/app/streamlit_app.py

test:
	pytest -q

lint:
	python -m pyflakes src || true
