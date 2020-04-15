OUTDIR = output
TMPDIR = tmp
LINEAR_RESULT_FILE_NAME = paper.dat
LINEAR_RESULT=$(TMPDIR)/linear/$(LINEAR_RESULT_FILE_NAME)
NONLINEAR_RESULT=$(TMPDIR)/nonlinear/
MKDIR_P = mkdir -p

all: $(OUTDIR)/tables/toy_benchmarks $(OUTDIR)/tables/prec_recall_arfs $(OUTDIR)/figures/importance_plots $(OUTDIR)/figures/featsel_threshold $(OUTDIR)/tables/NL_toy_benchmarks

.PHONY : clean all test

$(LINEAR_RESULT): runner/experiment_pipeline.py
		python runner/experiment_pipeline.py --iters 25 --filename "$(LINEAR_RESULT_FILE_NAME)"

$(OUTDIR)/tables/toy_benchmarks : $(LINEAR_RESULT) runner/paper_output.py
		python runner/paper_output.py --resfile $(LINEAR_RESULT)

$(OUTDIR)/tables/prec_recall_arfs : $(LINEAR_RESULT) prec_recall_on_AR_set.py
		python prec_recall_on_AR_set.py

$(OUTDIR)/figures/importance_plots: create_importance_figures.py
		python create_importance_figures.py

$(OUTDIR)/figures/featsel_threshold: plot_featsel_with_stat_threshold.py 
		python plot_featsel_with_stat_threshold.py

$(OUTDIR)/tables/NL_toy_benchmarks : run_NL_comparison.py
		python run_NL_comparison.py

clean :
	rm -r $(OUTDIR)
	rm -r $(TMPDIR)