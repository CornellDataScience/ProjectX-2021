(TeX-add-style-hook
 "neurips_2021"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("geometry" "verbose=true" "letterpaper")))
   (TeX-run-style-hooks
    "environ"
    "lineno"
    "natbib"
    "geometry")
   (TeX-add-symbols
    '("answerTODO" ["argument"] 0)
    '("answerNA" ["argument"] 0)
    '("answerNo" ["argument"] 0)
    '("answerYes" ["argument"] 0)
    '("patchBothAmsMathEnvironmentsForLineno" 1)
    '("patchAmsMathEnvironmentForLineno" 1)
    "acksection"
    "section"
    "subsection"
    "subsubsection"
    "paragraph"
    "subparagraph"
    "subsubsubsection"
    "ftype"
    "maketitle"
    "thanks"
    "And"
    "AND")
   (LaTeX-add-environments
    "ack"))
 :latex)

