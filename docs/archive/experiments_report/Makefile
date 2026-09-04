.PHONY: pdf tables figures clean
pdf:
	latexmk -pdf -interaction=nonstopmode main.tex
tables:
	cd .. && pixi run -e dev python experiments_report/scripts/extract_esmda.py && pixi run -e dev python experiments_report/scripts/extract_filtering.py && pixi run -e dev python experiments_report/scripts/extract_filter_smoothing.py && pixi run -e dev python experiments_report/scripts/extract_comparison.py && pixi run -e dev python experiments_report/scripts/make_conclusion_figures.py
figures:
	cd .. && for s in scripts/make_esmda_figures.py scripts/make_filtering_figures.py scripts/make_filter_smoothing_figures.py scripts/make_comparison_figures.py; do pixi run -e dev python experiments_report/$$s; done
clean:
	latexmk -C
