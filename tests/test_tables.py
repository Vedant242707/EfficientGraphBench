import csv
from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZipFile

import pytest
from typer.testing import CliRunner

from efficientgraphbench.cli.main import app
from efficientgraphbench.datasets.prepare import prepare_csv
from efficientgraphbench.datasets.tables import read_table


def workbook_fixture(path, sheets, formula=False):
    # Minimal OOXML fixture tests literal IDs and real reader behavior.
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "[Content_Types].xml",
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/></Types>',  # noqa: E501
        )
        archive.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>'
            + "".join(
                f'<sheet name="{name}" sheetId="{i}" r:id="rId{i}"/>'
                for i, name in enumerate(sheets, 1)
            )
            + "</sheets></workbook>",
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(
                f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'  # noqa: E501
                for i in range(1, len(sheets) + 1)
            )
            + "</Relationships>",
        )
        for i, rows in enumerate(sheets.values(), 1):
            contents = "".join(
                "<row>"
                + "".join(
                    '<c t="inlineStr"><is><t>' + escape(str(value)) + "</t></is></c>"
                    for value in row
                )
                + "</row>"
                for row in rows
            )
            if formula:
                contents += "<row><c><f>1+1</f><v>2</v></c></row>"
            archive.writestr(
                f"xl/worksheets/sheet{i}.xml",
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
                + contents
                + "</sheetData></worksheet>",
            )


def test_excel_graph_roundtrip_and_selection(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples/custom_graph"
    sheets = {}
    for name, filename in (("Nodes", "nodes.csv"), ("Edges", "edges.csv")):
        with (example / filename).open() as stream:
            sheets[name] = list(csv.reader(stream))
    path = tmp_path / "graph.xlsx"
    workbook_fixture(path, sheets)
    with pytest.raises(ValueError, match="Choose --sheet"):
        read_table(path)
    summary = prepare_csv(
        path,
        path,
        tmp_path / "ready",
        nodes_options={"sheet": "Nodes"},
        edges_options={"sheet": "Edges"},
    )
    assert summary["nodes"] == 12
    assert summary["classes"] == 2
    result = CliRunner().invoke(app, ["preview", "--file", str(path), "--sheet", "Nodes"])
    assert result.exit_code == 0, result.output
    assert "node_id" in result.output


def test_excel_literal_ids_and_formula_rejection(tmp_path):
    path = tmp_path / "ids.xlsx"
    workbook_fixture(path, {"Nodes": [["id", "value"], ["001", "2"]]})
    assert read_table(path).iloc[0, 0] == "001"
    workbook_fixture(path, {"Nodes": [["id"], ["001"]]}, formula=True)
    with pytest.raises(ValueError, match="formulas"):
        read_table(path)


@pytest.mark.parametrize("content", ["a,a\n1,2", "a,b\n1,2,3", ",b\n1,2", "a,b\n"])
def test_invalid_csv_table(tmp_path, content):
    path = tmp_path / "bad.csv"
    path.write_text(content)
    with pytest.raises(ValueError):
        read_table(path)


def test_tsv_header_offset_and_invalid_selectors(tmp_path):
    path = tmp_path / "data.tsv"
    path.write_text("title\nid\tvalue\n001\t5\n")
    assert read_table(path, header_row=2).iloc[0, 0] == "001"
    with pytest.raises(ValueError, match="positive"):
        read_table(path, header_row=0)
    with pytest.raises(ValueError, match="selectors"):
        read_table(path, sheet="Nodes")


def pdf_fixture(path, ruled=True):
    # Tiny real PDF: two columns and two rows, optionally with grid lines.
    content = (
        b"BT /F1 12 Tf 20 75 Td (id) Tj 100 0 Td (value) Tj "
        b"-100 -30 Td (001) Tj 100 0 Td (5) Tj ET\n"
    )
    if ruled:
        content += (
            b"10 30 m 210 30 l S 10 60 m 210 60 l S 10 90 m 210 90 l S 10 30 m 10 90 l S "
            b"110 30 m 110 90 l S 210 30 m 210 90 l S\n"
        )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 240 120] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"endstream",
    ]
    payload = b"%PDF-1.4\n"
    offsets = [0]
    for i, obj in enumerate(objects, 1):
        offsets.append(len(payload))
        payload += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(payload)
    payload += b"xref\n0 6\n0000000000 65535 f \n" + b"".join(
        f"{offset:010d} 00000 n \n".encode() for offset in offsets[1:]
    )
    payload += f"trailer << /Size 6 /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    path.write_bytes(payload)


def test_real_pdf_table_and_unreadable_input(tmp_path):
    path = tmp_path / "table.pdf"
    pdf_fixture(path)
    frame = read_table(path, page=1, table=1)
    assert frame.columns.tolist() == ["id", "value"]
    assert frame.iloc[0].tolist() == ["001", "5"]
    with pytest.raises(ValueError, match="only 1 pages"):
        read_table(path, page=2)
    with pytest.raises(ValueError, match="No readable"):
        read_table(path, table=2)
    pdf_fixture(path, ruled=False)
    with pytest.raises(ValueError, match="No readable"):
        read_table(path)
