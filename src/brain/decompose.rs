/*!
Decomposition — the pipeline's first step, splitting a prompt into
independently-routable sub-tasks before Brain routing.

This has been a bare `TODO` in `pipeline.rs`'s docstring since the first
commit. `RuleBasedDecomposer` is a deliberately conservative default: false
positives here (splitting a single-intent prompt) actively hurt quality,
while false negatives (leaving a genuinely multi-part prompt whole) just
fall back to today's behavior — one pass through the pipeline treating it as
a single request, which already works. So it only splits on explicit,
low-ambiguity structural markers and passes everything else through
unchanged.

A trained/prompted decomposer is a strictly better future upgrade once
there's data for it, but doesn't fit the same `Option<LinearHead>`-or-
heuristic shape as the critic reward head — decomposition needs a real
generation (rewriting/splitting text), not a linear probe on an embedding —
so that's future work, not attempted here.
*/

pub trait Decomposer: Send {
    /// Split `prompt` into sub-tasks to run independently. Returns
    /// `vec![prompt.to_string()]` unchanged when nothing splits cleanly.
    fn decompose(&self, prompt: &str) -> Vec<String>;
}

pub struct RuleBasedDecomposer;

impl Decomposer for RuleBasedDecomposer {
    fn decompose(&self, prompt: &str) -> Vec<String> {
        if let Some(items) = split_enumerated_list(prompt) {
            return items;
        }
        if let Some(items) = split_and_also_questions(prompt) {
            return items;
        }
        vec![prompt.to_string()]
    }
}

/// Splits numbered (`1.`/`1)`) or bulleted (`-`/`*`) lists into separate
/// sub-tasks — only when *every* non-empty line carries such a marker, so a
/// prompt that merely mentions a list inside prose doesn't get mis-split.
fn split_enumerated_list(prompt: &str) -> Option<Vec<String>> {
    let lines: Vec<&str> = prompt.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    if lines.len() < 2 {
        return None;
    }

    let mut items = Vec::with_capacity(lines.len());
    for line in &lines {
        items.push(strip_list_marker(line)?.to_string());
    }

    Some(items)
}

fn strip_list_marker(line: &str) -> Option<&str> {
    if let Some(rest) = line.strip_prefix("- ").or_else(|| line.strip_prefix("* ")) {
        return Some(rest.trim());
    }

    let digits_end = line.char_indices().take_while(|(_, c)| c.is_ascii_digit()).count();
    if digits_end == 0 {
        return None;
    }
    let rest = &line[digits_end..];
    let rest = rest.strip_prefix('.').or_else(|| rest.strip_prefix(')'))?;
    Some(rest.trim())
}

/// Splits on an explicit "...? and also ...?" / "...? Also, ...?" pattern —
/// two independently-answerable questions joined by an explicit conjunction.
/// A single "?" or an ambiguous bare "and" isn't enough to trigger this.
fn split_and_also_questions(prompt: &str) -> Option<Vec<String>> {
    // ASCII-only: `to_lowercase()` can change byte length for non-ASCII
    // text, which would desync `idx` (found in `lower`) from `prompt`'s own
    // byte offsets. Bailing out here is the safe, conservative choice this
    // module is meant to make anyway — and it means `idx` (found in `lower`)
    // stays valid for slicing `prompt` directly, since ASCII lowercasing is
    // always a 1-byte-to-1-byte mapping.
    if !prompt.is_ascii() || prompt.matches('?').count() < 2 {
        return None;
    }

    let lower = prompt.to_lowercase();
    let also_idx = find_word(&lower, "also")?;

    // Strip a trailing conjunction between the two questions, e.g.
    // "...France? And" -> "...France?", so both "? Also, ...?" and
    // "? And also, ...?" phrasings work.
    let mut head = prompt[..also_idx].trim();
    head = head.trim_end_matches(',').trim_end();
    if let Some(stripped) = head.strip_suffix("And").or_else(|| head.strip_suffix("and")) {
        head = stripped.trim_end_matches(',').trim_end();
    }

    let tail = prompt[also_idx + "also".len()..].trim_start_matches(',').trim();

    if head.ends_with('?') && !tail.is_empty() {
        Some(vec![head.to_string(), tail.to_string()])
    } else {
        None
    }
}

/// Find `word` in `haystack` as a whole word (not a substring of a longer
/// word), returning its byte offset.
fn find_word(haystack: &str, word: &str) -> Option<usize> {
    let bytes = haystack.as_bytes();
    let mut start = 0;
    while let Some(rel) = haystack[start..].find(word) {
        let idx = start + rel;
        let before_ok = idx == 0 || !bytes[idx - 1].is_ascii_alphanumeric();
        let after_idx = idx + word.len();
        let after_ok = after_idx >= bytes.len() || !bytes[after_idx].is_ascii_alphanumeric();
        if before_ok && after_ok {
            return Some(idx);
        }
        start = idx + word.len();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_intent_prompt_unchanged() {
        let d = RuleBasedDecomposer;
        let prompt = "Write a function that reverses a linked list.";
        assert_eq!(d.decompose(prompt), vec![prompt.to_string()]);
    }

    #[test]
    fn splits_bulleted_list() {
        let d = RuleBasedDecomposer;
        let prompt = "- Fix the login bug\n- Write tests for it";
        assert_eq!(
            d.decompose(prompt),
            vec!["Fix the login bug".to_string(), "Write tests for it".to_string()]
        );
    }

    #[test]
    fn splits_numbered_list() {
        let d = RuleBasedDecomposer;
        let prompt = "1. Summarize this doc\n2. Translate it to French";
        assert_eq!(
            d.decompose(prompt),
            vec!["Summarize this doc".to_string(), "Translate it to French".to_string()]
        );
    }

    #[test]
    fn splits_and_also_questions() {
        let d = RuleBasedDecomposer;
        let prompt = "What's the capital of France? And also, what's its population?";
        assert_eq!(
            d.decompose(prompt),
            vec![
                "What's the capital of France?".to_string(),
                "what's its population?".to_string()
            ]
        );
    }

    #[test]
    fn single_question_unchanged() {
        let d = RuleBasedDecomposer;
        let prompt = "What's the capital of France?";
        assert_eq!(d.decompose(prompt), vec![prompt.to_string()]);
    }
}
