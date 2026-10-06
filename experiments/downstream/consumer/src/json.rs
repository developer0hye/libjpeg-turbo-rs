//! A minimal JSON writer.
//!
//! Hand-written so the consumer depends on nothing but the codecs under test:
//! every extra crate would change the binary sizes and build times this
//! harness reports, and would add packages to the reviewed lock file.

use std::fmt::Write;

#[derive(Debug, Clone)]
pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    Integer(i128),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn object(entries: Vec<(&str, Json)>) -> Json {
        Json::Object(
            entries
                .into_iter()
                .map(|(key, value)| (key.to_string(), value))
                .collect(),
        )
    }

    pub fn str(value: &str) -> Json {
        Json::String(value.to_string())
    }

    pub fn int(value: impl Into<i128>) -> Json {
        Json::Integer(value.into())
    }

    /// Non-finite floats (an infinite PSNR) have no JSON spelling; they are
    /// written as `null` and the Markdown says "inf".
    pub fn num(value: f64) -> Json {
        if value.is_finite() {
            Json::Number(value)
        } else {
            Json::Null
        }
    }

    pub fn opt<T>(value: Option<T>, convert: impl FnOnce(T) -> Json) -> Json {
        value.map(convert).unwrap_or(Json::Null)
    }

    pub fn render(&self) -> String {
        let mut out: String = String::new();
        self.write_into(&mut out, 0);
        out.push('\n');
        out
    }

    fn write_into(&self, out: &mut String, indent: usize) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(value) => out.push_str(if *value { "true" } else { "false" }),
            Json::Number(value) => {
                let _ = write!(out, "{value}");
            }
            Json::Integer(value) => {
                let _ = write!(out, "{value}");
            }
            Json::String(value) => write_string(out, value),
            Json::Array(items) => {
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push_str("[\n");
                for (index, item) in items.iter().enumerate() {
                    push_indent(out, indent + 1);
                    item.write_into(out, indent + 1);
                    if index + 1 < items.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                push_indent(out, indent);
                out.push(']');
            }
            Json::Object(entries) => {
                if entries.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push_str("{\n");
                for (index, (key, value)) in entries.iter().enumerate() {
                    push_indent(out, indent + 1);
                    write_string(out, key);
                    out.push_str(": ");
                    value.write_into(out, indent + 1);
                    if index + 1 < entries.len() {
                        out.push(',');
                    }
                    out.push('\n');
                }
                push_indent(out, indent);
                out.push('}');
            }
        }
    }
}

fn push_indent(out: &mut String, indent: usize) {
    for _ in 0..indent {
        out.push_str("  ");
    }
}

fn write_string(out: &mut String, value: &str) {
    out.push('"');
    for character in value.chars() {
        match character {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            control if (control as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", control as u32);
            }
            other => out.push(other),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escapes_and_nests() {
        let value: Json = Json::object(vec![
            ("a", Json::str("x\"y\n")),
            (
                "b",
                Json::Array(vec![Json::int(1), Json::num(f64::INFINITY)]),
            ),
        ]);
        assert_eq!(
            value.render(),
            "{\n  \"a\": \"x\\\"y\\n\",\n  \"b\": [\n    1,\n    null\n  ]\n}\n"
        );
    }
}
