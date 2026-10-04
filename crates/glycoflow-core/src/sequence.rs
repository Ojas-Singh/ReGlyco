//! GLYCAM condensed sequence parsing (`builder.parse_glycam`).

use std::sync::OnceLock;

use regex::Regex;

use crate::error::{Error, Result};

const MODS: &str = r"(?:\[[0-9]+[A-Za-z]+(?:,[0-9]+[A-Za-z]+)*\])?";

fn re_tail() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"([ab])([0-9])-(OH|OME)$").unwrap())
}
fn re_mod_at() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^\[[0-9]+[A-Za-z]+(?:,[0-9]+[A-Za-z]+)*\]").unwrap())
}
fn re_res() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(&format!(r"^([A-Za-z0-9]+?)({MODS})([ab])([0-9])-([0-9])")).unwrap()
    })
}
fn re_last() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(&format!(r"^([A-Za-z0-9]+)({MODS})$")).unwrap())
}

/// One monosaccharide of the sequence tree.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    /// e.g. "DGlcpNAc"
    pub name: String,
    /// e.g. ["3S", "6S"]
    pub mods: Vec<String>,
    /// 'a' or 'b'
    pub anomer: char,
    /// anomeric carbon number (1, or 2 for ulosonic acids / ketoses)
    pub cpos: u32,
    /// parent oxygen position; None at the reducing end
    pub ppos: Option<u32>,
    /// child node indices (branch first, as Python builds them)
    pub children: Vec<usize>,
    /// GLYCAM residue code (resname[1:]) once resolved
    pub code: String,
}

/// Parsed sequence: node arena, the reducing-end node and the aglycone residue name.
#[derive(Clone, Debug)]
pub struct Sequence {
    pub nodes: Vec<Node>,
    pub root: usize,
    pub aglycone: String,
}

/// Position of a modification such as "3S" -> 3.
pub fn mod_pos(m: &str) -> u32 {
    let digits: String = m.chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().unwrap_or(0)
}

fn split_mods(group: &str) -> Vec<String> {
    if group.is_empty() {
        Vec::new()
    } else {
        group[1..group.len() - 1]
            .split(',')
            .map(str::to_string)
            .collect()
    }
}

enum Tok {
    Open,
    Close,
    Node(usize),
}

/// Parse GLYCAM condensed notation (e.g. "DGalpb1-4DGlcpNAcb1-OH"). The reducing end may be written
/// "...b1-OH", "...b1-OME" or bare (free reducing end, beta unless overridden by the builder).
pub fn parse_glycam(seq: &str) -> Result<Sequence> {
    let s_full: String = seq.trim().chars().filter(|c| *c != ' ').collect();
    let mut s: &str = &s_full;
    let (mut aglycone, mut tail_anomer, mut tail_cpos) = ("ROH".to_string(), None, 1u32);
    if let Some(c) = re_tail().captures(s) {
        aglycone = match &c[3] {
            "OH" => "ROH",
            _ => "OME",
        }
        .to_string();
        tail_anomer = c[1].chars().next();
        tail_cpos = c[2].parse().unwrap();
        s = &s[..c.get(0).unwrap().start()];
    }
    let mut nodes: Vec<Node> = Vec::new();
    let mut tokens: Vec<Tok> = Vec::new();
    let mut i = 0;
    while i < s.len() {
        let rest = &s[i..];
        if rest.starts_with(']') {
            tokens.push(Tok::Close);
            i += 1;
        } else if rest.starts_with('[') && !re_mod_at().is_match(rest) {
            tokens.push(Tok::Open);
            i += 1;
        } else if let Some(m) = re_res().captures(rest) {
            nodes.push(Node {
                name: m[1].to_string(),
                mods: split_mods(&m[2]),
                anomer: m[3].chars().next().unwrap(),
                cpos: m[4].parse().unwrap(),
                ppos: Some(m[5].parse().unwrap()),
                children: Vec::new(),
                code: String::new(),
            });
            tokens.push(Tok::Node(nodes.len() - 1));
            i += m.get(0).unwrap().end();
        } else {
            let m = re_last().captures(rest).ok_or_else(|| {
                let near: String = rest.chars().take(30).collect();
                Error::Parse(format!("cannot parse GLYCAM sequence near {near:?}"))
            })?;
            nodes.push(Node {
                name: m[1].to_string(),
                mods: split_mods(&m[2]),
                anomer: tail_anomer.unwrap_or('b'),
                cpos: tail_cpos,
                ppos: None,
                children: Vec::new(),
                code: String::new(),
            });
            tokens.push(Tok::Node(nodes.len() - 1));
            i = s.len();
        }
    }
    let root = match tokens.last() {
        Some(Tok::Node(k)) if nodes[*k].ppos.is_none() => *k,
        _ => {
            return Err(Error::Parse(
                "sequence must end with the reducing-end residue".into(),
            ))
        }
    };
    let mut parent = root;
    let mut stack = Vec::new();
    for tok in tokens[..tokens.len() - 1].iter().rev() {
        match tok {
            Tok::Close => stack.push(parent),
            Tok::Open => {
                parent = stack
                    .pop()
                    .ok_or_else(|| Error::Parse("unbalanced branch brackets".into()))?;
            }
            Tok::Node(k) => {
                nodes[parent].children.push(*k);
                parent = *k;
            }
        }
    }
    if !stack.is_empty() {
        return Err(Error::Parse("unbalanced branch brackets".into()));
    }
    Ok(Sequence {
        nodes,
        root,
        aglycone,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn branched_tree() {
        let s = parse_glycam("DManpa1-6[DManpa1-3]DManpb1-4DGlcpNAcb1-OH").unwrap();
        assert_eq!(s.aglycone, "ROH");
        let root = &s.nodes[s.root];
        assert_eq!(root.name, "DGlcpNAc");
        assert_eq!(root.anomer, 'b');
        let man = &s.nodes[root.children[0]];
        assert_eq!(man.ppos, Some(4));
        let kids: Vec<_> = man
            .children
            .iter()
            .map(|&k| s.nodes[k].ppos.unwrap())
            .collect();
        assert_eq!(kids, vec![3, 6]); // branch first, as Python
    }

    #[test]
    fn modifications() {
        let s = parse_glycam("DGalp[3S,6S]b1-4DGlcpNAc[6S]").unwrap();
        let root = &s.nodes[s.root];
        assert_eq!(root.mods, vec!["6S"]);
        assert_eq!(root.anomer, 'b');
        assert_eq!(s.nodes[root.children[0]].mods, vec!["3S", "6S"]);
        assert!(parse_glycam("DGalpb1-4[DGlcpNAcb1-OH").is_err());
    }
}
