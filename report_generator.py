from xhtml2pdf import pisa
from flask import render_template

def generate_pdf(personal, result, output_path):
    try:
            html = render_template("report_template.html", personal=personal, result=result)
            with open(output_path, "w+b") as f:
                    pisa.CreatePDF(html, dest=f)
    except Exception as e:
        print('[PDF Generation Error]', e)

