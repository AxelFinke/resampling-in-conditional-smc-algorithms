manuscriptdir="/home/axel/Dropbox/Apps/Overleaf/conditional_resampling_in_smc_algorithms/"
rdir="/home/axel/Dropbox/Apps/Overleaf/conditional_resampling_in_smc_algorithms/r_code/"

# Create .tex figures (note, this can take quite a bit of time!)
cd "$rdir/script/r"
R < create_figures.R --no-save # WARNING: this may need a machine with a lot of RAM
R < create_figures_gaussian_ssm.R --no-save

sleep 1

# Compile figures to PDF
cd "$rdir/tex/arxiv"
latexmk -pdf figures
sleep 2

pdftk figures.pdf burst output "$rdir/output/fig/arxiv"/page_%02d.pdf
#rm figures.pdf
cd "$rdir/output/fig/arxiv"
mv page_01.pdf toy_model_bias_linear_scale.pdf
mv page_02.pdf toy_model_bias_log_scale.pdf
mv page_03.pdf toy_model_autocorrelation.pdf

# Compile figures to PDF
cd "$rdir/tex/arxiv"
latexmk -pdf figures_ssm
sleep 2

pdftk figures_ssm.pdf burst output "$rdir/output/fig/arxiv"/page_%02d.pdf
cd "$rdir/output/fig/arxiv"
mv page_01.pdf gaussian_ssm.pdf


# mv page_01.pdf gaussian_ssm.pdf


sleep 2

mv *.pdf "$manuscriptdir/figures"

# sleep 5

# rm -r "$manuscriptdir/temp/"

#sleep 3

#rm figures.pdf
# todo: remove the .tex files

#



