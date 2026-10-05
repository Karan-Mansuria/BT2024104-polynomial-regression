"""Build the four-page assignment report from completed, measured results."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

import reportlab
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                               TableStyle, Image, PageBreak)


def build(results, destination):
    # Embed the font family so bold text renders consistently in PDF viewers.
    fonts = Path(reportlab.__file__).parent / 'fonts'
    pdfmetrics.registerFont(TTFont('ReportSans', str(fonts / 'Vera.ttf')))
    pdfmetrics.registerFont(TTFont('ReportSans-Bold', str(fonts / 'VeraBd.ttf')))
    pdfmetrics.registerFontFamily('ReportSans', normal='ReportSans', bold='ReportSans-Bold',
                                  italic='ReportSans', boldItalic='ReportSans-Bold')
    selections = {p: json.loads((results / p / 'selection.json').read_text())
                  for p in ('var1', 'var2')}
    audits = {p: json.loads((results / p / 'audit.json').read_text())
              for p in selections}
    checks = {p: json.loads((results / p / 'prediction_verification.json').read_text())
              for p in selections}
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('TitleCustom', fontName='ReportSans-Bold', fontSize=21,
                              leading=27, textColor=colors.HexColor('#14374c'), spaceAfter=12))
    styles.add(ParagraphStyle('BodyCustom', fontName='ReportSans', fontSize=9.5,
                              leading=14, spaceAfter=9))
    styles.add(ParagraphStyle('SmallCustom', fontName='ReportSans', fontSize=8,
                              leading=10, spaceAfter=6))
    styles.add(ParagraphStyle('BulletCustom', fontName='ReportSans', fontSize=9,
                              leading=13, leftIndent=12, bulletIndent=1, spaceAfter=7))
    styles.add(ParagraphStyle('BulletSmall', fontName='ReportSans', fontSize=8.5,
                              leading=12, leftIndent=12, bulletIndent=1, spaceAfter=6))
    styles.add(ParagraphStyle('SectionCustom', fontName='ReportSans-Bold', fontSize=12,
                              leading=17, spaceBefore=10, spaceAfter=7,
                              textColor=colors.HexColor('#14374c')))
    story = []

    def para(text, style='BodyCustom'):
        return Paragraph(text, styles[style])

    def add(text, style='BodyCustom'):
        story.append(para(text, style))

    def bullet(text, small=False):
        story.append(Paragraph(text, styles['BulletSmall' if small else 'BulletCustom'],
                               bulletText='\u2022'))

    def table(rows, widths):
        content = [[para('<b>' + escape(str(cell)) + '</b>' if i == 0 or j == 0
                        else escape(str(cell)), 'SmallCustom')
                    for j, cell in enumerate(row)] for i, row in enumerate(rows)]
        item = Table(content, colWidths=widths, hAlign='LEFT')
        item.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e5eef3')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LINEBELOW', (0, 0), (-1, 0), .6, colors.HexColor('#9eb4c2')),
            ('LINEBELOW', (0, 1), (-1, -1), .3, colors.HexColor('#d6e0e6')),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ]))
        story.append(item)
        story.append(Spacer(1, 8))

    add('Polynomial Regression', 'TitleCustom')
    add('Machine Learning Assignment 1 | <b>BT2024104</b>', 'BodyCustom')
    add('<b>GitHub:</b> <a href="https://github.com/Karan-Mansuria/BT2024104-polynomial-regression" '
        'color="#14374c">github.com/Karan-Mansuria/BT2024104-polynomial-regression</a>',
        'SmallCustom')
    add('The aim is to predict the Net Power Score from six turbine settings in var1 '
        'and the Thermal Anomaly Score from three spatial coordinates in var2. '
        'I fitted a separate polynomial regression model to each dataset and used '
        'cross-validation to choose its degree and regularization. The supplied '
        'test files were used only after the models had been selected.')
    table([
        ['Problem', 'Training rows', 'Input variables', 'Degrees evaluated'],
        ['var1', audits['var1']['rows'], 'x1 to x6', '1 to 10'],
        ['var2', audits['var2']['rows'], 'x1 to x3', '1 to 20'],
    ], [75, 110, 140, 165])
    add('Data preparation', 'SectionCustom')
    bullet('<b>Data checks.</b> CSV column names and numeric values were checked; no missing or nonfinite '
        'values were accepted. Rows were retained in their original order. There '
        f"were {audits['var1']['duplicate_rows']} and {audits['var2']['duplicate_rows']} "
        'duplicate complete training rows in var1 and var2, respectively. '
        f"Var2 had {audits['var2']['duplicate_inputs']} repeated input rows with differing "
        'targets; these observations were retained. No imputation or row deletion was needed.')
    add('Polynomial features and scaling', 'SectionCustom')
    bullet('<b>Total-degree expansion.</b> A degree-d model includes every power and interaction whose exponents sum '
        'to at most d. For example, x1<super>2</super>x2 is a degree-3 term. With p '
        'inputs, the expansion contains (p+d)!/(p!d!) - 1 terms; an intercept is '
        'fitted separately. The maximum expansions contain 8,007 terms for var1 '
        'and 1,770 for var2, so regularization is useful when fitting these models.')
    bullet('<b>Scaling within each fold.</b> I standardized the inputs, generated polynomial '
        'features, and standardized the expanded columns before regression. Both '
        'scalers used only that fold\'s training rows. Validation rows received the '
        'same fitted transformations. This prevents their statistics from affecting '
        'the fitted model and puts polynomial terms on comparable scales.')
    add('Model selection', 'SectionCustom')
    bullet('<b>Ridge search.</b> Five shuffled folds (seed 2024104) were shared across candidates. Ridge was '
        'evaluated at every allowed degree using one singular-value decomposition '
        'per degree/fold. Alpha exponents ranged from -8 to 8 by decades, with a '
        'local 0.25-exponent refinement. This reuses an exact Ridge calculation '
        'without changing the polynomial model.')
    bullet('<b>LASSO and Elastic Net.</b> Ridge shrinks coefficients with an L2 penalty; LASSO uses an L1 penalty '
        'that can set coefficients to zero. Elastic Net combines both penalties. '
        'At each problem\'s two best Ridge degrees, LASSO and Elastic Net were compared '
        'using alpha exponents from -8 to 6 and Elastic Net ratios 0.1, 0.3, 0.5, '
        '0.7, and 0.9. Sparse fits used tolerance 0.001 and at most 2,000 iterations. '
        'Candidates failing convergence on any fold were excluded. The eligible '
        'candidate with minimum mean validation MSE was selected; R2 was also recorded.')
    add('<b>Evaluation:</b> MSE is the average squared prediction error. R<super>2</super> = 1 - SSE/SST compares '
        'the prediction error with variation around the validation target mean. '
        'Tables show the mean and sample standard deviation across five folds. '
        'Ridge was solved by SVD; LASSO and Elastic Net used coordinate descent.', 'SmallCustom')

    for problem, title in [('var1', 'Steam turbine optimization'),
                           ('var2', 'Thermal reservoir mapping')]:
        story.append(PageBreak())
        selected = selections[problem]
        add(f'{problem} | {title}', 'TitleCustom')
        ratio = f"{selected['ratio']:.2g}" if selected['method'] == 'ElasticNet' else 'Not applicable'
        table([
            ['Selected setting', 'Value'],
            ['Method / degree', f"{selected['method']} / {selected['degree']}"],
            ['Alpha / L1 ratio', f"{selected['alpha']:.6g} / {ratio}"],
            ['Polynomial terms', selected['polynomial_features']],
            ['CV MSE (mean +/- sample SD)', f"{selected['mean_cv_mse']:.6f} +/- {selected['std_cv_mse']:.6f}"],
            ['CV R2 (mean +/- sample SD)', f"{selected['mean_cv_r2']:.6f} +/- {selected['std_cv_r2']:.6f}"],
        ], [245, 245])
        story.append(Image(str(results / problem / 'degree_comparison.png'), width=470, height=235))
        add('Figure: best eligible alpha at each tested degree/method. Blue hollow circles '
            'show Elastic Net; orange crosses show LASSO. The right panel magnifies '
            'the shortlisted comparison; a distant Ridge score can lie outside its '
            'range. Sparse models were evaluated only at the two shortlisted degrees.', 'SmallCustom')
        add('Why this model was selected', 'SectionCustom')
        if problem == 'var1':
            bullet('<b>Degree comparison.</b> Ridge MSE fell from 9.863 at degree 1 to 0.508 at degree 5, then '
                'rose to 0.987 at degree 10. Extra polynomial terms beyond degree 5 '
                'therefore did not improve validation performance in the Ridge search. '
                'Degrees 5 and 6 were taken forward for LASSO and Elastic Net.')
            bullet('<b>Regularizer comparison.</b> At degree 5, LASSO reduced MSE to <b>0.348086</b>; Elastic Net was almost '
                'identical at 0.348270. This small difference does not establish a '
                'meaningful advantage for LASSO, but it wins under the minimum-MSE '
                'selection rule. Degree-6 LASSO had MSE 0.359550, so the selected '
                'model also used fewer polynomial terms.')
        else:
            bullet('<b>Degree comparison.</b> Low-degree models underfit var2: Ridge MSE was 35.451 at degree 1 '
                'and 1.442 at degree 5. It decreased to 0.235048 at degree 12, '
                'then increased to 0.283836 at degree 20. Degree 12 was the '
                'minimum in the tested Ridge curve; degree 11 was very close '
                'at 0.235464.')
            bullet('<b>Regularizer comparison.</b> The best shortlisted Elastic Net model had MSE 0.236355 at degree '
                '11, while the best LASSO model had MSE 0.242271 at degree 11. '
                'Ridge was selected by the same minimum-MSE rule used for var1. '
                'Its small advantage over degree-11 Ridge and Elastic Net should '
                'be interpreted cautiously, rather than as a decisive difference.')
        bullet(f"<b>Final fit.</b> The selected pipeline was fitted again on all <b>{audits[problem]['rows']} "
            '</b>'
            'training rows and passed the convergence check. The reported choice '
            'is the best eligible configuration evaluated, not a claim that every '
            'possible polynomial model was searched.', small=True)

    story.append(PageBreak())
    add('Prediction checks and conclusions', 'TitleCustom')
    for problem in selections:
        story.append(Image(str(results / problem / 'oof_diagnostics.png'), width=440, height=201))
    add('The out-of-fold predictions broadly follow the observed values for both '
        'datasets. The residual plots show remaining errors around zero; var1 has '
        'several larger errors. These plots alone do not establish that errors '
        'are independent or have constant variance.', 'SmallCustom')
    add('Interpretation', 'SectionCustom')
    bullet('<b>Selected models:</b> degree-5 LASSO for var1 and degree-12 Ridge for '
        'var2. Their validation results support different levels of polynomial '
        'complexity for the two problems. Since the same folds were used to select '
        'the configurations, these CV scores may be optimistic; they are not '
        'independent estimates of hidden-test performance.', small=True)
    bullet('<b>Search limitations.</b> LASSO and Elastic Net were searched only at two shortlisted degrees. '
        'Their alpha grids and iteration budgets were finite, so better settings '
        'elsewhere could have been missed. Fits that failed convergence were '
        'excluded, not treated as evidence of poor predictive accuracy.', small=True)
    add('Final predictions', 'SectionCustom')
    bullet(f"<b>Format and verification.</b> Each prediction file contains exactly <b>{checks['var1']['rows']} rows</b> and one "
        '<b>y</b> column, matching sample_submission.csv, with no extra index. All '
        'predictions are finite. Reloaded CSV values were compared against the saved '
        'model in original test-row order, including a separate batched inference check.', small=True)
    bullet('<b>Submission files:</b> BT2024104_pred_var1.csv and BT2024104_pred_var2.csv. '
        'Training and inference are in assignment.py; Ridge, LASSO, Elastic Net and '
        'fold-local preprocessing are in search.py. Fold metrics, hyperparameters, '
        'convergence records and OOF predictions are retained with the results.', small=True)

    destination.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(destination), pagesize=(595, 842),
                            rightMargin=50, leftMargin=50, topMargin=40, bottomMargin=42,
                            title='Polynomial Regression - BT2024104', author='BT2024104')
    doc.build(story)
    print(destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, default=Path('output/pdf/BT2024104_report.pdf'))
    args = parser.parse_args()
    build(args.results, args.output)
