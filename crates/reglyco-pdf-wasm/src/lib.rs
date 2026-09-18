//! Small, separately loaded PDF renderer for the versioned browser report.

use serde_json::Value;
use wasm_bindgen::prelude::*;

fn ascii(value: &str) -> String {
    value
        .chars()
        .map(|character| {
            if character.is_ascii_graphic() || character == ' ' {
                character
            } else {
                '?'
            }
        })
        .collect::<String>()
        .replace('\\', "\\\\")
        .replace('(', "\\(")
        .replace(')', "\\)")
}

fn field<'a>(value: &'a Value, path: &[&str]) -> &'a str {
    path.iter()
        .try_fold(value, |current, key| current.get(*key))
        .and_then(Value::as_str)
        .unwrap_or("")
}

fn render(bundle: &Value) -> Vec<u8> {
    let report = bundle.get("report").unwrap_or(&Value::Null);
    let lines = [
        field(report, &["title"]),
        "",
        field(report, &["summary"]),
        "",
        field(bundle, &["workflow"]),
        field(bundle, &["status"]),
        field(report, &["engineVersion"]),
        field(report, &["generatedAt"]),
        field(report, &["inputSha256"]),
        "",
        "Generated locally by the versioned ReGlyco PDF renderer.",
        "See report.json and provenance in the ZIP for the complete scientific record.",
    ];
    let mut stream = String::from("BT\n/F1 18 Tf\n54 780 Td\n");
    for (index, line) in lines.iter().enumerate() {
        if index == 1 {
            stream.push_str("/F1 10 Tf\n");
        }
        stream.push_str(&format!("({}) Tj\n0 -18 Td\n", ascii(line)));
    }
    stream.push_str("ET");
    let objects = [
        "<< /Type /Catalog /Pages 2 0 R >>".to_string(),
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>".to_string(),
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>".to_string(),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>".to_string(),
        format!("<< /Length {} >>\nstream\n{}\nendstream", stream.len(), stream),
    ];
    let mut output = String::from("%PDF-1.4\n");
    let mut offsets = Vec::new();
    for (index, object) in objects.iter().enumerate() {
        offsets.push(output.len());
        output.push_str(&format!("{} 0 obj\n{}\nendobj\n", index + 1, object));
    }
    let xref = output.len();
    output.push_str(&format!(
        "xref\n0 {}\n0000000000 65535 f \n",
        objects.len() + 1
    ));
    for offset in offsets {
        output.push_str(&format!("{offset:010} 00000 n \n"));
    }
    output.push_str(&format!(
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n",
        objects.len() + 1
    ));
    output.into_bytes()
}

#[wasm_bindgen]
pub fn render_pdf(bundle: JsValue) -> Result<js_sys::Uint8Array, JsValue> {
    let bundle: Value = serde_wasm_bindgen::from_value(bundle)
        .map_err(|error| JsValue::from_str(&format!("invalid report bundle: {error}")))?;
    Ok(js_sys::Uint8Array::from(render(&bundle).as_slice()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_a_pdf_from_an_in_memory_report() {
        let bytes = render(
            &serde_json::json!({ "workflow": "validate", "status": "succeeded", "report": { "title": "ReGlyco", "summary": "Valid", "engineVersion": "1", "generatedAt": "now", "inputSha256": "abc" } }),
        );
        assert!(bytes.starts_with(b"%PDF-1.4"));
        assert!(bytes.ends_with(b"%%EOF\n"));
    }
}
