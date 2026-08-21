import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_reports"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORTS_DATA = [
    {
        "filename": "Infosys_Q1_FY25.pdf",
        "title": "Infosys Reports Strong Q1 FY25 Results",
        "quarter": "Q1 FY25",
        "revenue": "₹39,315 crore",
        "net_profit": "₹6,368 crore",
        "margin": "21.1%",
        "growth": "3.6%",
        "dividend": "an interim dividend of ₹20 per share",
        "commentary": "\"We had a strong start to FY25 with broad-based growth across financial services and European markets. Client commitments in generative AI projects accelerated significantly during Q1 FY25.\"",
        "risks": "Macroeonomic uncertainties in discretionary tech spending and elongated deal conversion cycles in North American retail operations."
    },
    {
        "filename": "Infosys_Q2_FY25.pdf",
        "title": "Infosys Announces Q2 FY25 Financial Results",
        "quarter": "Q2 FY25",
        "revenue": "₹40,986 crore",
        "net_profit": "₹6,506 crore",
        "margin": "21.5%",
        "growth": "5.1%",
        "dividend": "an interim dividend of ₹21 per share",
        "commentary": "\"Demand commentary remained positive in cloud transformation and enterprise automation. Large deal TCV reached $2.4 billion for Q2 FY25, highlighting our execution capabilities.\"",
        "risks": "Potential wage inflation, currency volatility, and delayed decision-making in large transformation contracts."
    },
    {
        "filename": "Infosys_Q3_FY25.pdf",
        "title": "Infosys Reports Q3 FY25 Earnings",
        "quarter": "Q3 FY25",
        "revenue": "₹41,850 crore",
        "net_profit": "₹6,710 crore",
        "margin": "21.8%",
        "growth": "4.8%",
        "dividend": "a special dividend of ₹18 per share",
        "commentary": "\"Q3 FY25 saw sustained margin expansion driven by Project Maximus cost optimization. Demand for AI-led cost efficiency solutions remains robust among Fortune 500 clients.\"",
        "risks": "Geopolitical headwinds, supply chain disruptions, and higher furloughs during third quarter holiday seasonality."
    },
    {
        "filename": "Infosys_Q4_FY25.pdf",
        "title": "Infosys Delivers Q4 & Full Year FY25 Results",
        "quarter": "Q4 FY25",
        "revenue": "₹42,500 crore",
        "net_profit": "₹6,890 crore",
        "margin": "22.0%",
        "growth": "6.2%",
        "dividend": "a final dividend of ₹22 per share",
        "commentary": "\"We concluded FY25 with solid momentum in cloud and digital services. FY25 total revenues reached ₹164,651 crore with steady operating margins across all four quarters.\"",
        "risks": "Talent attrition in niche AI skillsets and pricing pressure in legacy application maintenance contracts."
    }
]

def generate_pdf(data: dict):
    filepath = OUTPUT_DIR / data["filename"]
    doc = SimpleDocTemplate(str(filepath), pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=16, leading=20, spaceAfter=12)
    h2_style = ParagraphStyle('H2Style', parent=styles['Heading2'], fontSize=12, leading=16, spaceBefore=8, spaceAfter=4)
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=10, leading=14, spaceAfter=6)
    
    story = []
    story.append(Paragraph(f"INFOSYS LIMITED - QUARTERLY PRESS RELEASE", title_style))
    story.append(Paragraph(f"<b>{data['title']}</b> ({data['quarter']})", h2_style))
    story.append(Spacer(1, 10))
    
    # Section 1: Financials
    story.append(Paragraph("1. FINANCIAL HIGHLIGHTS & REVENUE PERFORMANCE", h2_style))
    p1 = f"Revenues for {data['quarter']} were <b>{data['revenue']}</b>, representing year-on-year growth of {data['growth']} in constant currency terms. Net profit for {data['quarter']} reached <b>{data['net_profit']}</b>, demonstrating solid financial discipline. Operating margin for {data['quarter']} stood at <b>{data['margin']}</b>, reflecting efficiency gains under our continuous margin improvement programs."
    story.append(Paragraph(p1, body_style))
    
    # Section 2: Segment Growth
    story.append(Paragraph("2. SEGMENT PERFORMANCE & GROWTH LEADS", h2_style))
    p2 = f"During {data['quarter']}, the Manufacturing and Hi-Tech segment emerged as the fastest-growing segment with 7.8% YoY growth, supported by enterprise cloud and AI modernization deals. Financial Services recorded stable growth at 4.1% YoY."
    story.append(Paragraph(p2, body_style))
    
    # Section 3: Management Commentary
    story.append(Paragraph("3. MANAGEMENT COMMENTARY ON DEMAND", h2_style))
    p3 = f"Management Commentary: {data['commentary']}"
    story.append(Paragraph(p3, body_style))
    
    # Section 4: Capital Allocation & Dividend
    story.append(Paragraph("4. CAPITAL ALLOCATION & DIVIDEND DECLARATION", h2_style))
    p4 = f"In line with our capital allocation framework, the Board of Directors for {data['quarter']} declared <b>{data['dividend']}</b> for all eligible shareholders."
    story.append(Paragraph(p4, body_style))
    
    # Section 5: Risks & Headwinds
    story.append(Paragraph("5. RISKS AND HEADWINDS", h2_style))
    p5 = f"Key risks and headwinds identified for {data['quarter']} include: {data['risks']}"
    story.append(Paragraph(p5, body_style))
    
    doc.build(story)
    print(f"Generated PDF: {filepath}")

def main():
    print("Generating sample financial report PDFs...")
    for data in REPORTS_DATA:
        generate_pdf(data)
    print("All 4 quarterly sample PDFs generated successfully.")

if __name__ == "__main__":
    main()
