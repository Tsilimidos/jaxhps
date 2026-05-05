FILENAME = main

PDFLATEX = pdflatex
BIBER = biber

FILENAME_BEAMER = ${FILENAME}

all:
	make presentation

presentation-base:
	${PDFLATEX} ${FILENAME_BEAMER}.tex
	${PDFLATEX} ${FILENAME_BEAMER}.tex
	${BIBER} ${FILENAME_BEAMER}
	${PDFLATEX} ${FILENAME_BEAMER}.tex

bibliography-presentation:

presentation:
	make presentation-base
	make bibliography-presentation

cleanup:
	rm -f *.aux *.log *.out *.gz *.nav *-pics.pdf *.snm *.toc *.bbl *.blg *.xml *.bcf *-blx.bib *.dvi *.lof *.lot
	
clean:
	make cleanup
	rm ${FILENAME}.pdf