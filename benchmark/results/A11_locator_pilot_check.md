# A11: figure locator, checked by eye on the 10 pilots (2026-11-23)

Done before any tool output of this round was read or scored (A11 §6).

**What was checked.** `ground_truth/locate_key_figures.py --role pilot --draw`
found every key figure of the 10 pilot papers on its page (57 figures) and
drew each box with its match score. Every drawn page was looked at
(contact sheets of all 56 pages, plus full-size views of the lowest scores).

**One bug found and fixed first.** PLOS captions carry equation images,
and the key took a figure's first `<graphic>` anywhere inside it, so for 7
figures of two PLOS Computational Biology pilots (PMC5991653, PMC8244857)
the "figure" was an equation image (match 0.32-0.39, boxes on the wrong
place). The key now takes the figure's own graphic, outside its caption
(`jats_to_common.figure_image`, test added). The file-name match also
stopped treating ".g001" as an extension.

**Result after the fix.** 57 of 57 figures located on the right page with
the box on the figure. Match scores 0.769-0.99; the lowest are sparse line
art (a schematic, chemical structures), still correctly placed.

**Threshold.** `LOCATE_THRESHOLD = 0.6` (harness/a11_measures.py): every
correct match scored >= 0.769; the only wrong matches seen (the bug above)
scored <= 0.39. 0.6 sits in the empty gap between them. A scored figure
below 0.6 counts as "not located", is left out and counted.
