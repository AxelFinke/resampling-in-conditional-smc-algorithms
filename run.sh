# By default, these commands only re-draw the figures from the existing
# data stored in the /data subfolder.
#
# To replicate the results from scratch, set the flag
# simulated_data <- TRUE
# in the first four R scripts; for the fifth R script, please instead re-run the
# script run_simulation_study_state_space_models.R first.
#
# However, note that re-running the simulation studies can take a while.

cd ./script/r

R < create_figure_1.R --no-save # creates Figure 1 (which illustrates the weighting bias)
R < create_figure_2.R --no-save # creates Figure 2 (which illustrates the analytical lower bound)
R < create_figure_3.R --no-save # creates Figure 2 (which illustrates the truncation bias)
R < create_figures_for_real_data_study.R --no-save # creates the figures for the real-data experiment from Section 3.3
R < create_figures_for_simulated_data_study.R --no-save # creates the figures for the simulated-data experiments from Section 3.2 (also writes Tables 1 and 2 to the command line)

# The above commands generate the figures as tikz code which could
# directly be included into a LaTeX document via the tikzpicture environment.
# The remaining script compiles these to PDF.

cd ../../tex

latexmk -pdf figures.tex # compiles the figures to PDF;
latexmk -c figures.tex # removes all auxiliary files

mv figures.pdf ../output/

cd "../output"

pdftk figures.pdf burst && rm figures.pdf # split PDF file of figures into individual sheets

mv pg_0001.pdf weighting_bias.pdf # Figure 1
mv pg_0002.pdf bounds.pdf # Figure 2
mv pg_0003.pdf truncation_bias.pdf # Figure 3
mv pg_0004.pdf likelihood_estimates.pdf # Figure 4
mv pg_0005.pdf smd_linear_gaussian_state_space_model.pdf # Figure 5
mv pg_0006.pdf smd_stochastic_volatility_model.pdf # Figure 7
mv pg_0007.pdf smd_nonlinear_state_space_model.pdf # Figure 8
mv pg_0008.pdf smd_lorenz_state_space_model.pdf # Figure 9
mv pg_0009.pdf mse_linear_gaussian_state_space_model.pdf # Figure 10
mv pg_0010.pdf mse_stochastic_volatility_model.pdf # Figure 11
mv pg_0011.pdf mse_nonlinear_state_space_model.pdf # Figure 12
mv pg_0012.pdf mse_lorenz_state_space_model.pdf # Figure 13
mv pg_0013.pdf likelihood_estimates_real_data.pdf # Figure 6

cd ..










