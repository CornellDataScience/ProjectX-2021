(TeX-add-style-hook
 "research_proposal"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("neurips_2021" "nonatbib") ("graphicx" "pdftex")))
   (TeX-run-style-hooks
    "latex2e"
    "article"
    "art10"
    "neurips_2021"
    "inputenc"
    "fontenc"
    "hyperref"
    "url"
    "booktabs"
    "amsfonts"
    "nicefrac"
    "microtype"
    "xcolor"
    "graphicx")
   (TeX-add-symbols
    "RR"
    "Nat"
    "CC")
   (LaTeX-add-labels
    "gen_inst"
    "headings"
    "others"
    "sample-table"))
 :latex)

