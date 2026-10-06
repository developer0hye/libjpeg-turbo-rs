//! Source scanner shared by the two P4-141 `unsafe` gates
//! (`tests/parser_unsafe_gate.rs`, criterion 5, and
//! `tests/unsafe_inventory_gate.rs`, criterion 4).
//!
//! Included by path from each gate (`#[path = "helpers/unsafe_scan.rs"]`)
//! rather than through `helpers/mod.rs`, so the gates do not pull the C-tool
//! discovery helpers into binaries that never spawn a C tool.
//!
//! **What the scanner understands.** [`unsafe_token_lines`] is a small lexer,
//! not a line heuristic: `//` comments, nested `/* */` comments, `"…"` strings
//! with backslash escapes, raw strings (`r"…"`, `r#"…"#`, with `b`/`c`
//! prefixes), and char/byte literals (including `'"'`) are skipped, so an
//! `unsafe` hidden after a `"\"//"` or inside a multi-line literal is still
//! found, and prose about `unsafe` in any comment form is not. The deliberate
//! residue: a raw identifier `r#unsafe` is reported (a loud false positive,
//! and not something these sources should contain anyway).
//!
//! [`enclosing_item`] names the item a site belongs to — the nearest
//! preceding `fn` declaration, or the `impl`/`trait` header when the token is
//! `unsafe impl` / `unsafe trait` — so the inventory can be keyed by
//! (file, item) and a site that *moves* between functions is a diff, not
//! just one that is added. A macro-template name (`$inner`) is qualified
//! with its `macro_rules!` name, because one file's two macros routinely
//! declare the same template.
//!
//! Attribution reads the **masked** copy the same pass produces, in which
//! the contents of every comment and literal are blanked, not the raw text:
//! a `fn` declaration inside a `/* */` block or a multi-line raw string, or
//! the phrase `unsafe impl Send for X` in a trailing `//` comment, would
//! otherwise name the row.
//!
//! It is a deterministic attribution both the gate and the inventory author
//! use, not a parser: a closure or a nested `fn` declared earlier in the same
//! body is attributed to that inner `fn`, and that is fine as long as both
//! sides agree.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// One lexer pass: where the code `unsafe` tokens are, and a copy of the
/// source with every comment blanked to spaces.
struct Scan {
    /// 1-based line numbers of code `unsafe` tokens.
    unsafe_lines: Vec<usize>,
    /// The source with the *contents* of every comment and literal replaced
    /// by spaces, newlines kept, so line and column positions still line up
    /// with the original. String delimiters survive, so `extern "C" fn` is
    /// still recognisable as a declaration. Item attribution reads this,
    /// never the raw text: otherwise a `fn` declaration inside a `/* */`
    /// block or a multi-line raw string, or the phrase `unsafe impl Send for
    /// X` in a trailing `//` comment, would name the row.
    code: String,
}

/// Blank one position unless it is a newline, so the masked copy keeps the
/// original's line structure.
fn blank(masked: &mut [char], index: usize) {
    if let Some(slot) = masked.get_mut(index) {
        if *slot != '\n' {
            *slot = ' ';
        }
    }
}

/// Line numbers (1-based) of every whole-word `unsafe` token that is code:
/// outside `//` and nested `/* */` comments, outside string, raw-string,
/// char and byte literals.
pub fn unsafe_token_lines(text: &str) -> Vec<usize> {
    scan(text).unsafe_lines
}

fn scan(text: &str) -> Scan {
    let chars: Vec<char> = text.chars().collect();
    let len: usize = chars.len();
    let mut masked: Vec<char> = chars.clone();
    let at = |i: usize| -> char { chars.get(i).copied().unwrap_or('\0') };
    let is_ident = |c: char| c.is_alphanumeric() || c == '_';

    let mut lines: Vec<usize> = Vec::new();
    let mut line: usize = 1;
    let mut i: usize = 0;
    while i < len {
        let c: char = at(i);
        if c == '\n' {
            line += 1;
            i += 1;
        } else if c == '/' && at(i + 1) == '/' {
            // Line comment: skip to end of line (the `\n` is counted above).
            while i < len && at(i) != '\n' {
                blank(&mut masked, i);
                i += 1;
            }
        } else if c == '/' && at(i + 1) == '*' {
            // Block comment, nestable per the Rust reference.
            let mut depth: usize = 1;
            blank(&mut masked, i);
            blank(&mut masked, i + 1);
            i += 2;
            while i < len && depth > 0 {
                if at(i) == '/' && at(i + 1) == '*' {
                    depth += 1;
                    blank(&mut masked, i);
                    blank(&mut masked, i + 1);
                    i += 2;
                } else if at(i) == '*' && at(i + 1) == '/' {
                    depth -= 1;
                    blank(&mut masked, i);
                    blank(&mut masked, i + 1);
                    i += 2;
                } else {
                    if at(i) == '\n' {
                        line += 1;
                    }
                    blank(&mut masked, i);
                    i += 1;
                }
            }
        } else if c == '"' {
            // String literal with backslash escapes. The delimiters survive
            // masking so `extern "C" fn` still tokenizes as a qualifier.
            i += 1;
            while i < len && at(i) != '"' {
                if at(i) == '\\' {
                    blank(&mut masked, i);
                    i += 1;
                }
                if at(i) == '\n' {
                    line += 1;
                }
                blank(&mut masked, i);
                i += 1;
            }
            i += 1;
        } else if c == '\'' {
            // Char literal (`'x'`, `'\n'`, `'"'`) or a lifetime (`'a`).
            if at(i + 1) == '\\' {
                blank(&mut masked, i + 1);
                i += 2;
                while i < len && at(i) != '\'' {
                    blank(&mut masked, i);
                    i += 1;
                }
                i += 1;
            } else if at(i + 2) == '\'' {
                blank(&mut masked, i + 1);
                i += 3;
            } else {
                i += 1;
            }
        } else if is_ident(c) {
            // Identifier or keyword; also catches raw-string openers.
            let start: usize = i;
            while i < len && is_ident(at(i)) {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let raw_prefix: bool = word == "r" || word == "br" || word == "cr";
            if raw_prefix && (at(i) == '"' || at(i) == '#') {
                // Raw string: `r"…"`, `r#"…"#`, with any number of hashes.
                let mut hashes: usize = 0;
                while at(i) == '#' {
                    hashes += 1;
                    i += 1;
                }
                if at(i) == '"' {
                    i += 1;
                    loop {
                        if i >= len {
                            break;
                        }
                        if at(i) == '"' && (1..=hashes).all(|h| at(i + h) == '#') {
                            i += 1 + hashes;
                            break;
                        }
                        if at(i) == '\n' {
                            line += 1;
                        }
                        blank(&mut masked, i);
                        i += 1;
                    }
                }
            } else if word == "unsafe" {
                lines.push(line);
            }
        } else {
            i += 1;
        }
    }
    Scan {
        unsafe_lines: lines,
        code: masked.into_iter().collect(),
    }
}

/// The name declared by a `fn` declaration line, if `line` is one.
///
/// Accepts the qualifiers the crate uses in front of `fn` — visibility,
/// `const`, `async`, `unsafe`, `extern "…"`, `default` — and macro-template
/// names (`fn $name(`), which is how the NEON colour kernels are declared.
/// A `unsafe fn (…)` *type* (no name after `fn`) is not a declaration.
///
/// Expects a comment-masked line (see [`Scan::code`]); prose is not filtered
/// here. A `fn` sharing its line with other code (`impl T { fn f() { … } }`
/// written on one line) is rejected by the qualifier check and the site
/// falls through to whatever declaration precedes it — a shape this crate
/// does not use, pinned by `attribution_handles_the_shapes_the_crate_uses`.
fn declared_fn_name(line: &str) -> Option<String> {
    let trimmed: &str = line.trim();
    let tokens: Vec<&str> = trimmed.split_whitespace().collect();
    let fn_index: usize = tokens.iter().position(|t| *t == "fn")?;
    let qualifier = |t: &str| -> bool {
        t == "const"
            || t == "async"
            || t == "unsafe"
            || t == "extern"
            || t == "default"
            || t.starts_with("pub")
            || t.starts_with('"')
    };
    if !tokens[..fn_index].iter().all(|t| qualifier(t)) {
        return None;
    }
    let after: &str = tokens.get(fn_index + 1)?;
    let name: &str = after.split(['(', '<']).next().unwrap_or("");
    let is_name = |s: &str| {
        let body: &str = s.strip_prefix('$').unwrap_or(s);
        !body.is_empty() && body.chars().all(|c| c.is_alphanumeric() || c == '_')
    };
    is_name(name).then(|| name.to_string())
}

/// The first occurrence of `keyword` in `line` that is a whole token, so
/// `impl` does not match inside `simpl` or `implementation`.
fn find_keyword(line: &str, keyword: &str) -> Option<usize> {
    let is_ident = |c: char| c.is_alphanumeric() || c == '_';
    let mut from: usize = 0;
    while let Some(offset) = line.get(from..)?.find(keyword) {
        let start: usize = from + offset;
        let end: usize = start + keyword.len();
        let before: Option<char> = line[..start].chars().next_back();
        let after: Option<char> = line[end..].chars().next();
        if !before.is_some_and(is_ident) && !after.is_some_and(is_ident) {
            return Some(start);
        }
        from = end;
    }
    None
}

/// The `impl …` / `trait …` header on a line, with generic parameter lists
/// removed and whitespace collapsed: `unsafe impl<T: Send> Sync for OnceBox<T>
/// {}` becomes `impl Sync for OnceBox`.
fn item_header(line: &str, keyword: &str) -> Option<String> {
    let start: usize = find_keyword(line, keyword)?;
    let rest: &str = &line[start..];
    let end: usize = rest.find(['{', ';']).unwrap_or(rest.len());
    let mut out: String = String::new();
    let mut depth: usize = 0;
    for c in rest[..end].chars() {
        match c {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            _ if depth == 0 => out.push(c),
            _ => {}
        }
    }
    Some(out.split_whitespace().collect::<Vec<&str>>().join(" "))
}

/// The `macro_rules!` name at or above `from`, if any. Two macros in one
/// file routinely declare the same template name (`$inner`), so a
/// `$`-prefixed key is qualified with the macro that generates it —
/// otherwise moving an `unsafe` from one macro to its sibling would change
/// no count and the gate would stay green.
fn macro_name_above(lines: &[&str], from: usize) -> Option<String> {
    (0..=from).rev().find_map(|i| {
        let line: &str = lines[i];
        let start: usize = find_keyword(line, "macro_rules")?;
        let rest: &str = line[start + "macro_rules".len()..].trim_start();
        let rest: &str = rest.strip_prefix('!')?.trim_start();
        let name: String = rest
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        (!name.is_empty()).then_some(name)
    })
}

/// The inventory key for the `unsafe` token on `lineno` (1-based) of
/// `lines` — which must be comment-masked (see [`Scan::code`]): the
/// `impl`/`trait` header for `unsafe impl` / `unsafe trait`, otherwise the
/// nearest `fn` declaration at or above the site (qualified by its
/// `macro_rules!` name when that declaration is a template), or `(module)`
/// when there is none.
fn enclosing_item(lines: &[&str], lineno: usize) -> String {
    let index: usize = lineno.saturating_sub(1);
    if let Some(site) = lines.get(index) {
        let compact: String = site.split_whitespace().collect::<Vec<&str>>().join(" ");
        for keyword in ["impl", "trait"] {
            if compact.contains(&format!("unsafe {keyword} "))
                || compact.contains(&format!("unsafe {keyword}<"))
            {
                if let Some(header) = item_header(&compact, keyword) {
                    return header;
                }
            }
        }
    }
    let declaration: Option<(usize, String)> = (0..=index.min(lines.len().saturating_sub(1)))
        .rev()
        .find_map(|i| declared_fn_name(lines[i]).map(|name| (i, name)));
    match declaration {
        Some((at, name)) if name.starts_with('$') => match macro_name_above(lines, at) {
            Some(macro_name) => format!("{macro_name}::{name}"),
            None => name,
        },
        Some((_, name)) => name,
        None => "(module)".to_string(),
    }
}

/// One code `unsafe` token: its line, the item it belongs to, and the
/// trimmed source line, for reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnsafeSite {
    pub line: usize,
    pub item: String,
    pub text: String,
}

/// Every code `unsafe` token in `text`, attributed to its enclosing item.
pub fn unsafe_sites_in(text: &str) -> Vec<UnsafeSite> {
    let source: Vec<&str> = text.lines().collect();
    let scanned: Scan = scan(text);
    let code: Vec<&str> = scanned.code.lines().collect();
    scanned
        .unsafe_lines
        .into_iter()
        .map(|line| UnsafeSite {
            line,
            item: enclosing_item(&code, line),
            text: source.get(line - 1).map_or("", |l| l.trim()).to_string(),
        })
        .collect()
}

/// Every code `unsafe` token in the file at `path`.
pub fn unsafe_sites(path: &Path) -> Vec<UnsafeSite> {
    let text: String =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    unsafe_sites_in(&text)
}

/// Every `.rs` file under `dir`, recursively, in sorted order.
pub fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut paths: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    paths.sort();
    for path in paths {
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// `path` relative to `root`, with forward slashes on every platform.
pub fn relative(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}
