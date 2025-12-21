# دستورالعمل ساخت گزارش (فارسی)

پیش‌نیازها:

- توزیع TeX (مثل MacTeX) نصب باشد.
- بستهٔ `minted` به همراه Python و `pygments` نصب باشد.

پیش‌خوانی سریع:

1. وارد شاخهٔ قالب گزارش شوید:

```bash
cd /Users/tahamajs/Documents/uni/DGM/OtherTermAssignments/CA1/report/DGM_Report_Template
```

2. برای تولید PDF با `latexmk` (پیشنهادی):

```bash
latexmk -pdf main.tex
```

یا با `pdflatex` و اجرای دوباره برای رفع مراجع متقابل:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

نکات:

- اگر از `minted` استفاده می‌کنید، هنگام اجرای `pdflatex` از گزینهٔ `-shell-escape` استفاده کنید:

```bash
pdflatex -shell-escape main.tex
```

- تصاویر باید در پوشهٔ `images/` قرار گیرند و نام‌ها در فایل‌های `.tex` منطبق باشند.

- در صورت نیاز به پاکسازی فایل‌های واسط از:

```bash
latexmk -c
```

موفق باشید.
