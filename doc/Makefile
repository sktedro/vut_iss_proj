NAME=doc

ALL: $(NAME).tex
	latex $(NAME).tex
	pdflatex $(NAME).tex

pdf: ALL

clean:
	rm -f *.log *.dvi *.pdf *.aux *.out *.zip
