# IEEE Transactions-style two-column paper: typographic and structural specification

Research status: only one search returned usable verified content. Direct fetches of ieee.org, CTAN, MIT and the IEEE Author Center failed (DNS failures, proxy 403). Curl downloads of IEEEtran.cls were also blocked. Everything marked [V] is supported by a cited search result. Everything marked [R] is recalled from background knowledge and is unverified. Check [R] items against the real IEEEtran.cls (`\textwidth`, `\columnsep`, `\topmargin` are set in the class body) or compile a template to confirm.

## 1. What are the exact font sizes, margins, column dimensions and caption conventions of the IEEEtran journal format?

### Takeaway
Journal mode is US Letter, 10 pt Times, two columns, with columns of 3.5 in (21 pc) and a gutter of about 1 pc. Captions are 8 pt or smaller than body text, Table captions are centred small caps above the table, and Fig. captions sit below. Only the column widths and the class options are verified.

### Cited Findings
- IEEEtran class options: `conference`, `journal`, `technote`, `peerreview`, `peerreviewca` select the format mode. Default font size is 10pt (9/10/11/12pt available). Default paper is letterpaper. Default is twocolumn. Technotes should use 9pt. `peerreview` mode is the double-spaced review layout. Some journals ask for captions removed and figures/tables on separate pages at initial submission — [IEEEtran.cls header comments, as summarized by search](https://ece.iisc.ac.in/~spcom/2018/img/IEEEtran.cls)
- Official IEEE graphics widths: one column 3.5 in (88.9 mm, 21 picas); two columns 7.16 in (182 mm, 43 picas); an older guide says 7 1/16 in for two columns. Maximum depth in the older guide is 9 in (54 picas), leaving room for the caption — [Proceedings of the IEEE guidelines (via search summary)](https://proceedingsoftheieee.ieee.org/resources/guidelines-for-figures-and-tables/)
- Resolution: more than 300 dpi for colour and grayscale, more than 600 dpi for black-and-white line art. Graphics should not be sized narrower than one column — same source.

### Recalled, unverified [R]
- Page: US Letter, 8.5 x 11 in (215.9 x 279.4 mm).
- Text block: width about 7.16 in (43 pc), two columns of 3.5 in (21 pc) with a column gutter of about 1 pc (0.17 in, about 4.2 mm). This is inferred from the verified figure widths. The class may use slightly different values (some versions use 0.25 in).
- Side margins about 0.67 in each (8.5 - 7.16, halved). Top margin about 0.75 in. Bottom margin about 1 in. Verify against the .cls.
- Body: Times (`mathptmx` or `newtxtext`/`newtxmath`), 10 pt on 12 pt leading, justified, paragraph indent about 1 em, no inter-paragraph space.
- Sizes (10 pt journal): title 24 pt (`\Huge`-equivalent, not bold, Times); author names 11 pt; affiliations 10 pt italic (author block is centred); abstract and index terms 9 pt bold (journal style: "Abstract—" bold, text bold in many Transactions, regular in others); section headings 10 pt small caps centred; subsection headings 10 pt italic, flush left; caption 8 pt (`\footnotesize`) or 9 pt; table body 8 pt; references 8 pt on about 9.5 pt leading (`\footnotesize`); footnotes 8 pt.
- Captions: `Fig. 1.` below the figure, text centred if one line, justified if longer. `TABLE I` above the table, centred small caps, with the table title in small caps on the next line. In current Transactions style the figure caption is 8 pt regular. Do not use vertical rules in tables.

### Inferences
- A faithful LaTeX build needs `\documentclass[journal]{IEEEtran}`, `\usepackage{cite}`, `\usepackage{amsmath,amssymb,graphicx,booktabs,algorithmic/algorithm2e}`, and `\usepackage{newtxtext,newtxmath}` is optional. Note the class itself defines caption styles, so avoid the `caption` package, or configure it to match.

### Gaps
- Exact margin values, exact sizes (title, abstract, captions) could not be verified from the class file or the IEEE templates.

## 2. How are the title block, abstract, index terms, headers, footers, headings, equations, tables, figures, algorithms, citations and references formatted?

### Takeaway
Everything below is [R] recalled from IEEEtran and published Transactions papers, with no verified source this session. It describes the canonical look and gives literal examples.

### Cited Findings
- None verified beyond Section 1. (The IEEEtran header comments summarised above confirm the journal/peerreview modes only.)

### Recalled, unverified [R]
**Title block.** Title centred, large, not bold. Authors centred below in the form `First M. Last, Member, IEEE, Second Author, Senior Member, IEEE, and Third Author, Fellow, IEEE`. Affiliations are conveyed via a first-page footnote, not under the names, in Transactions style (the conference style puts them under the names). First-page footnote lines, 8 pt, separated by a short rule:
- "Manuscript received 12 March 2025; revised 2 June 2025; accepted 15 July 2025. Date of publication 1 August 2025; date of current version 10 September 2025."
- "This work was supported in part by the National Natural Science Foundation of China under Grant 61234567. (Corresponding author: Jane Doe.)"
- "The authors are with the Department of X, University Y, City, Country (e-mail: a@b.edu)."
- "Digital Object Identifier 10.1109/TIV.2025.1234567" (final line).

**Abstract.** One paragraph, 150-250 words (the commonly stated norm is no more than 200-250). Begins "Abstract—" with an em dash, bold or italic lead-in. No citations, no equations, no abbreviations undefined, no references to figures.
**Index Terms—** Alphabetical list, 4-10 terms, separated by commas, ending with a full stop, from the IEEE Taxonomy where possible. Example: `Index Terms—Autonomous vehicles, federated learning, non-IID data, privacy, vehicular networks.`

**Header and footer.**
- Published layout: even pages carry the page number at the left and "IEEE TRANSACTIONS ON INTELLIGENT VEHICLES, VOL. 10, NO. 3, MARCH 2025" in small caps; odd pages carry the short title (author running title in small caps) at the left and the page number at the right. The footer carries the copyright and DOI line (the first page only).
- Review layout (`peerreview`): double-spaced, single column, line-numbered, with a "> REPLACE THIS LINE WITH YOUR PAPER IDENTIFICATION NUMBER (DOUBLE-CLICK HERE TO EDIT) <" header.

**Headings.** Level 1: `I. INTRODUCTION`, centred, small caps, Roman numeral and full stop. Level 2: `A. Problem Formulation`, flush left, italic, title case. Level 3: `1) Local Training:` run-in italic, with the Arabic numeral and a closing parenthesis. Acknowledgment heading is unnumbered: `ACKNOWLEDGMENT` (the singular spelling, per IEEE). Appendices: `APPENDIX A`, `APPENDIX B` (or a single `APPENDIX`), with section titles in small caps. Proofs go there.

**Equations.** Numbered on the right in parentheses, flush right: `(1)`, `(2)`. Equations are centred on the column. Reference as "(1)" or "(1)-(3)" (IEEE style: "(1)", never "Eq. (1)" at the start of a sentence — "Equation (1)" is allowed at a sentence start). Punctuate displayed equations as part of the sentence. Variables italic, vectors bold, operators upright (`\mathrm`, `\operatorname`). Long equations break at an operator with `split` or `multline`, and span both columns only in an exceptional case.

**Tables.** Caption above. `TABLE I` on the first line (small caps, centred), then the title in small caps below it (`\textsc`). Horizontal rules only (booktabs-like: top, header, bottom), with no vertical lines in modern Transactions. Text left-aligned, numbers aligned on the decimal point (`siunitx` `S` columns) with a consistent number of decimals. Best results bold, second best underlined. Units in column headers: "Accuracy (%)". Table text 8 pt. Wide tables use `table*` and are placed at the top of a page.

**Figures.** `Fig. 1.` followed by a caption, below the figure, ending with a full stop. Sub-figures labelled "(a)", "(b)" beneath each panel, referenced as "Fig. 2(a)". In text: "Fig. 1" (abbreviated even at the start of a sentence is spelt out as "Figure 1"; IEEE says to abbreviate "Fig." except at the start of a sentence). Single-column float: `figure`; two-column float: `figure*` (top of the page only, via `[!t]`).

**Algorithms.** Boxed or ruled float titled `Algorithm 1: Name`, with `Input:` / `Output:` lines, numbered lines, 8-9 pt. Common packages: `algorithm` + `algpseudocode`/`algorithmic`, or `algorithm2e`. Keep it to one column wide where possible.

**Citations.** Numbered in square brackets in order of first appearance: `[1]`, `[2], [3]`, `[2]-[4]` (use the `cite` package for compressed ranges). The bracket sits before punctuation, with a space before it. Name the authors only with "Smith et al. [5]" or "[5]". Never use the citation as a noun ("in [5]" is acceptable).

**References.** Section `REFERENCES` (small caps, centred), 8 pt, hanging indent with bracketed labels. Style `IEEEtran.bst`. Examples (literal):
- Journal: `[1] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas, "Communication-efficient learning of deep networks from decentralized data," in Proc. 20th Int. Conf. Artif. Intell. Statist., 2017, pp. 1273-1282.` (this is in fact a conference entry; for a journal use the form below)
- Journal: `[2] Q. Yang, Y. Liu, T. Chen, and Y. Tong, "Federated machine learning: Concept and applications," ACM Trans. Intell. Syst. Technol., vol. 10, no. 2, pp. 1-19, Feb. 2019.`
- Conference: `[3] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, "Federated optimization in heterogeneous networks," in Proc. Mach. Learn. Syst., 2020, pp. 429-450.`
- arXiv: `[4] P. Kairouz et al., "Advances and open problems in federated learning," 2019, arXiv:1912.04977.`
- Book: `[5] I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. Cambridge, MA, USA: MIT Press, 2016.`
Rules: authors are written initials first with the surname after; up to six authors are listed, and more than six becomes the first author plus "et al."; the article title is in quotation marks with sentence case; the journal name is abbreviated and italic; the volume and number are `vol. X, no. Y`; the month is abbreviated (Jan., Feb., Mar., Apr., May, Jun., Jul., Aug., Sep., Oct., Nov., Dec.), and a DOI may be appended as `doi: 10.1109/...`.

**Biographies and acknowledgments.** The biography block appears at the end of the paper, after the references: a 1 in x 1.25 in photo at the left of the paragraph, the name in bold, and the membership grade ("Member, IEEE"), followed by a paragraph of about 80-150 words (birth, degrees, appointments, research interests). Produced with `\begin{IEEEbiography}[{\includegraphics[...]}]{Name}`; `IEEEbiographynophoto` when there is no photo. The Acknowledgment appears before the references, in the same size as the body text.

**Length.** A regular paper is typically 10-14 two-column pages in a final version for TIV/TITS, although the exact limit is unverified (many IEEE Transactions have an overlength charge after 8 pages, and a limit of 12-16 pages is common). Correspondence/brief papers: 4-6 pages.

### Inferences
- Because the footnotes and the header lines depend on the journal, treat the examples as a template, and check one recent paper from the target journal.

### Gaps
- Per-journal limits on pages (TIV, TITS, TNNLS, IoT-J), the exact footers, and the abstract word limit were not verified.

## 3. Which typographic details make a paper look "professional" versus amateur?

### Takeaway
All [R]. The main things are consistent floats, no widows/orphans, legible figure text and aligned numbers.

### Recalled, unverified [R]
- No single-line widows or orphans at the top or bottom of a column. Use `\widowpenalty=10000 \clubpenalty=10000` and balance the last page columns (`\usepackage{balance}` or `\IEEEtriggeratref`).
- Floats: figures and tables at the top of columns (`[!t]`), never in the middle of a paragraph. Every float is cited in the text before it appears, and floats keep the order of their numbers.
- Avoid overfull boxes (long URLs in references: use `\usepackage{url}` and `\allowbreak`). Hyphenation stays on; avoid more than two consecutive hyphenated lines.
- Units: a non-breaking space between number and unit (`10~ms`, `5~GHz`, `$\SI{5}{\mega\hertz}$`). A thin space is not needed. "%" with no space ("85%"), IEEE style: "85%".
- Numbers: spell out one to nine in text, use digits for 10 and above and for any number with a unit. Decimal leading zero ("0.5"). A consistent number of decimals in a table column.
- Abbreviations are defined at first use in the abstract and in the body. Avoid starting a sentence with a symbol. Do not use "etc." in formal text sparingly.
- Fonts embedded in the PDF (Type 1/TrueType, no Type 3), vector figures (PDF/EPS) rather than screenshots, and consistent colours that also work in greyscale (different markers and line styles).
- Consistent capitalisation in headings (title case for section titles, sentence case for reference titles), and a consistent spelling variant (US).

### Gaps
- None of these was checked against the IEEE Editorial Style Manual in this session.

## 4. Figure dimensions and minimum text size

### Takeaway
A single-column figure is 88.9 mm (3.5 in) wide; a double-column figure is 182 mm (7.16 in) wide. Text inside a figure should end up about 8 pt after scaling (the caption size); a lower limit of about 6-7 pt is [R] and unverified.

### Cited Findings
- One column 3.5 in (88.9 mm, 21 pc); two columns 7.16 in (182 mm, 43 pc); not narrower than one column; height up to 9 in (54 pc) in the older guide, leaving room for the caption; resolution > 300 dpi colour/greyscale, > 600 dpi line art — [Proceedings of the IEEE guidelines (via search summary)](https://proceedingsoftheieee.ieee.org/resources/guidelines-for-figures-and-tables/)

### Inferences
- Create matplotlib figures at the final size, `figsize=(3.5, 2.4)` for single column and `(7.16, 2.8)` for double column, and set `font.size` to 8 and `savefig(dpi=300, bbox_inches="tight")` as a PDF, so that no rescaling happens. For a 3.5 in wide figure with a height of 2.2-2.6 in, a ratio of about 4:3 to 3:2 is typical.

### Gaps
- IEEE's official minimum text size inside figures was not found. The "8 pt, no less than about 6 pt" guidance is from background knowledge and unverified.

## 5. Brief comparison: ACM acmart sigconf and NeurIPS/ICML

### Takeaway
All [R]. These differ mainly in column layout and reference style.

### Recalled, unverified [R]
- ACM `acmart` `sigconf`: US Letter, two columns, 9 pt Linux Libertine body text, Biolinum sans-serif headings (bold, numbered 1, 1.1, no Roman numerals), title block with authors and affiliations under the title, a CCS concepts block and Keywords (not "Index Terms"), ACM Reference Format box on page 1, author-year or numeric `ACM-Reference-Format` references, captions below figures and also below tables (tables captioned above is also allowed), column width about 3.33 in with a 0.33 in gutter (about 7 in text width).
- NeurIPS/ICML: single column (NeurIPS: 5.5 in text width, 11 pt Times, 13.5 in / 9 in height; ICML: two-column, 6.75 in text width, 10 pt Times, with a 0.25 in column gap in the two-column ICML format). Title at the top of the first page, with an abstract in a narrower indented block. Numbered sections (1, 1.1), tables are captioned above and figures below (booktabs rules), references in author-year (ICML) or numeric (NeurIPS), a page limit of 8-9 pages for the main text, and an unlimited appendix.
- Key differences from IEEE: no Roman numerals, no small-caps headings, no "Index Terms", no biography blocks, and the body is smaller or larger.

### Gaps
- Numbers are unverified. Check the current year's official style files.

## Overall gaps

- Official IEEE sources (Editorial Style Manual, IEEEtran_HOWTO, journal author information pages) were not reachable, so almost all values are [R].
- Suggested verification: obtain `IEEEtran.cls` v1.8b and `IEEEtran_HOWTO.pdf` from CTAN offline, compile the journal template, and measure `\textwidth`, `\columnsep`, `\topmargin`, and caption fonts with `\showthe`.
