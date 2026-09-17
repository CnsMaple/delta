//! Character-level "word diff" marking, ported from
//! dsh-plugin-git `src/client/diff.ts`, which itself ports VSCode's
//! `vs/editor/common/diff/defaultLinesDiffComputer` (MIT).
//!
//! It replaces delta's token-level Levenshtein edit inference (`align.rs` /
//! `edits.rs`) with VSCode's exact algorithm chain for marking the changed
//! characters *within* an already line-paired change block:
//!
//! 1. Both sides become character sequences over their TRIMMED lines joined by
//!    `\n` elements (`CharSeq`, VSCode's `LinesSliceCharSequence` with
//!    `ignoreTrimWhitespace=true`).
//! 2. One sequence diff over the whole block — DP LCS below 500 elements
//!    (diagonal-preferring), Myers `O(ND)` above.
//! 3. The heuristic chain, exactly as `refineDiff` orders it:
//!    `optimize_sequence_diffs` (join-by-shifting twice, then boundary-score
//!    shifting), `extend_diffs_to_words`, `remove_short_matches` (equal gaps of
//!    at most 2 chars merge), `remove_very_short_matching_text` (cap-130 power
//!    formula + short prefix/suffix extension to whole lines).
//! 4. Ranges map back to per-line spans; whitespace outside the trimmed range is
//!    never marked.
//!
//! Git already fixed line pairing, so the line-alignment stage is not re-run:
//! each hunk's deletion run against its addition run is one change block.
//!
//! ponytail: elements are Unicode scalars (`char`), whereas VSCode / the TS
//! source use UTF-16 code units. This is identical for BMP text (the practical
//! case, incl. CJK); astral-plane characters (e.g. emoji) count as 1 element
//! here vs 2 in VSCode. Upgrade path: switch `elements` to `Vec<u16>` via
//! `encode_utf16` if exact astral parity is ever needed.

use std::cmp::min;
use std::collections::HashMap;

/// Above this product, a block falls back to whole-line marks — the viewer bound
/// VSCode expresses as its time budget.
const BLOCK_PRODUCT_LIMIT: i64 = 25_000_000;
/// Below this combined length the sequence diff uses the DP table, as VSCode's
/// refineDiff selects.
const DP_LENGTH_LIMIT: i64 = 500;

/// Word-level emphasis on one side of a diff.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mark {
    Del,
    Ins,
}

/// One marked or plain piece of a diff line. `text` is a slice of the original
/// line; `mark == None` means the text survives on both sides.
///
/// For a zero-width bar (`text.is_empty()`), `spans_line` records whether the
/// other side's change covers a whole line (its range includes a `\n` separator).
/// Such a bar is redundant in side-by-side (the counterpart line has its own row)
/// and is dropped; an intra-line bar is kept and rendered as a placeholder cell.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiffSpan<'a> {
    pub text: &'a str,
    pub mark: Option<Mark>,
    pub spans_line: bool,
}

// ───────────────────────────────── ranges ─────────────────────────────────

/// A half-open offset range, mirroring VSCode's `OffsetRange`. Indices are `i64`
/// to mirror JS number arithmetic (no unsigned underflow during shifting).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct OffsetRange {
    start: i64,
    end: i64, // endExclusive
}

impl OffsetRange {
    fn len(self) -> i64 {
        self.end - self.start
    }
    fn is_empty(self) -> bool {
        self.end <= self.start
    }
}

fn join_ranges(a: OffsetRange, b: OffsetRange) -> OffsetRange {
    OffsetRange {
        start: a.start.min(b.start),
        end: a.end.max(b.end),
    }
}

fn intersect_ranges(a: OffsetRange, b: OffsetRange) -> Option<OffsetRange> {
    let start = a.start.max(b.start);
    let end = a.end.min(b.end);
    if start <= end {
        Some(OffsetRange { start, end })
    } else {
        None
    }
}

fn intersects_ranges(a: OffsetRange, b: OffsetRange) -> bool {
    a.start < b.end && b.start < a.end
}

/// One change between the two sequences: a range on each side at once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SequenceDiff {
    seq1: OffsetRange,
    seq2: OffsetRange,
}

fn diff_ends(d: SequenceDiff) -> (i64, i64) {
    (d.seq1.end, d.seq2.end)
}

fn join_diffs(a: SequenceDiff, b: SequenceDiff) -> SequenceDiff {
    SequenceDiff {
        seq1: join_ranges(a.seq1, b.seq1),
        seq2: join_ranges(a.seq2, b.seq2),
    }
}

fn delta_diff(d: SequenceDiff, offset: i64) -> SequenceDiff {
    SequenceDiff {
        seq1: OffsetRange {
            start: d.seq1.start + offset,
            end: d.seq1.end + offset,
        },
        seq2: OffsetRange {
            start: d.seq2.start + offset,
            end: d.seq2.end + offset,
        },
    }
}

fn delta_start_diff(d: SequenceDiff, offset: i64) -> SequenceDiff {
    SequenceDiff {
        seq1: OffsetRange {
            start: d.seq1.start + offset,
            end: d.seq1.end,
        },
        seq2: OffsetRange {
            start: d.seq2.start + offset,
            end: d.seq2.end,
        },
    }
}

fn delta_end_diff(d: SequenceDiff, offset: i64) -> SequenceDiff {
    SequenceDiff {
        seq1: OffsetRange {
            start: d.seq1.start,
            end: d.seq1.end + offset,
        },
        seq2: OffsetRange {
            start: d.seq2.start,
            end: d.seq2.end + offset,
        },
    }
}

fn intersect_diffs(a: SequenceDiff, b: SequenceDiff) -> Option<SequenceDiff> {
    let s1 = intersect_ranges(a.seq1, b.seq1)?;
    let s2 = intersect_ranges(a.seq2, b.seq2)?;
    Some(SequenceDiff { seq1: s1, seq2: s2 })
}

/// The equal runs between the given diffs, as `SequenceDiff.invert`.
fn invert_diffs(
    diffs: &[SequenceDiff],
    doc1_length: i64,
    doc2_length: i64,
) -> Vec<SequenceDiff> {
    let mut result: Vec<SequenceDiff> = Vec::new();
    let mut prev: Option<SequenceDiff> = None;
    for cur in diffs {
        let start = prev.map(diff_ends).unwrap_or((0, 0));
        result.push(SequenceDiff {
            seq1: OffsetRange {
                start: start.0,
                end: cur.seq1.start,
            },
            seq2: OffsetRange {
                start: start.1,
                end: cur.seq2.start,
            },
        });
        prev = Some(*cur);
    }
    let start = prev.map(diff_ends).unwrap_or((0, 0));
    result.push(SequenceDiff {
        seq1: OffsetRange {
            start: start.0,
            end: doc1_length,
        },
        seq2: OffsetRange {
            start: start.1,
            end: doc2_length,
        },
    });
    result
}

// ───────────────────────────────── sequences ─────────────────────────────────

/// One side of a block: its lines trimmed and joined by `\n` elements, mirroring
/// `LinesSliceCharSequence` with `considerWhitespaceChanges=false`.
struct CharSeq {
    /// Chars of the trimmed lines, separated by `'\n'`.
    elements: Vec<char>,
    /// Element offset where each line's trimmed text starts.
    line_start: Vec<usize>,
    /// Element offset just past each line's trimmed text (its separator, or end).
    line_trimmed_end: Vec<usize>,
    /// Leading-whitespace width per line. Kept 0 in whitespace-aware mode (leading
    /// whitespace is now part of the diffed elements, so marks may cover it).
    trimmed_ws: Vec<usize>,
    /// Per line: byte offsets of each char plus a final `line.len()`, for
    /// mapping char indices back into the original `&str`.
    char_bytes: Vec<Vec<usize>>,
}

impl CharSeq {
    fn new(lines: &[&str]) -> Self {
        let mut elements: Vec<char> = Vec::new();
        let mut line_start = Vec::with_capacity(lines.len());
        let mut line_trimmed_end = Vec::with_capacity(lines.len());
        let mut trimmed_ws = Vec::with_capacity(lines.len());
        let mut char_bytes = Vec::with_capacity(lines.len());
        for (at, line) in lines.iter().enumerate() {
            // byte offsets of every char, plus the terminator
            let mut offs: Vec<usize> = line.char_indices().map(|(i, _)| i).collect();
            offs.push(line.len());
            char_bytes.push(offs);

            // Consider whitespace changes (VSCode's considerWhitespaceChanges=true):
            // keep leading/trailing whitespace so indentation changes are marked,
            // and only strip the single '\n' that delta's prepare() appends.
            let body = line.strip_suffix('\n').unwrap_or(line);
            trimmed_ws.push(0);
            line_start.push(elements.len());
            elements.extend(body.chars());
            line_trimmed_end.push(elements.len());
            if at < lines.len() - 1 {
                elements.push('\n');
            }
        }
        Self {
            elements,
            line_start,
            line_trimmed_end,
            trimmed_ws,
            char_bytes,
        }
    }

    fn length(&self) -> i64 {
        self.elements.len() as i64
    }

    /// Char at element `offset`; out of range yields `'\u{0}'` so that two
    /// out-of-range accesses compare equal, matching JS `arr[i]` semantics.
    fn element(&self, offset: i64) -> char {
        *self
            .elements
            .get(offset.max(0) as usize)
            .unwrap_or(&'\u{0}')
    }

    fn is_strongly_equal(&self, offset1: i64, offset2: i64) -> bool {
        self.element(offset1) == self.element(offset2)
    }

    fn text(&self, range: OffsetRange) -> String {
        let start = range.start.max(0) as usize;
        let end = (range.end as usize).min(self.elements.len());
        self.elements[start..end].iter().collect()
    }

    /// `translateOffset`: the line index a sequence offset falls on.
    fn line_of(&self, offset: i64) -> usize {
        let offset = offset.clamp(0, self.length());
        let mut low = 0usize;
        let mut high = self.line_start.len() - 1;
        while low < high {
            let mid = (low + high + 1) >> 1;
            if self.line_start[mid] as i64 <= offset {
                low = mid;
            } else {
                high = mid - 1;
            }
        }
        low
    }

    /// `countLinesIn`: the line-number delta between the range's ends.
    fn count_lines_in(&self, range: OffsetRange) -> usize {
        let end = range.end.min(self.length());
        self.line_of(end) - self.line_of(range.start)
    }

    /// `extendToFullLines`: the element range of every line the range touches.
    fn extend_to_full_lines(&self, range: OffsetRange) -> OffsetRange {
        let mut start_line = 0usize;
        for (at, &s) in self.line_start.iter().enumerate() {
            if s as i64 <= range.start {
                start_line = at;
            }
        }
        let mut end = self.length();
        for (at, &s) in self.line_start.iter().enumerate() {
            if range.end <= s as i64 {
                end = s as i64;
                break;
            }
            let _ = at;
        }
        OffsetRange {
            start: self.line_start[start_line] as i64,
            end,
        }
    }

    /// `findWordContaining`: ASCII letters/digits, as VSCode's `isWordChar`.
    fn find_word_containing(&self, offset: i64) -> Option<OffsetRange> {
        if offset < 0 || offset >= self.length() || !is_word_char(char_code(self.element(offset)))
        {
            return None;
        }
        let mut start = offset;
        while start > 0
            && is_word_char(char_code(self.element(start - 1)))
        {
            start -= 1;
        }
        let mut end = offset;
        while end < self.length() && is_word_char(char_code(self.element(end))) {
            end += 1;
        }
        Some(OffsetRange { start, end })
    }

    /// `getBoundaryScore`: the category table that anchors insertion placement.
    fn boundary_score(&self, length: i64) -> i64 {
        let prev = if length > 0 {
            char_code(self.element(length - 1))
        } else {
            -1
        };
        let next = if length < self.length() {
            char_code(self.element(length))
        } else {
            -1
        };
        let prev_cat = boundary_category(prev);
        let next_cat = boundary_category(next);
        if prev_cat == 7 && next_cat == 8 {
            return 0;
        }
        if prev_cat == 8 {
            return 150;
        }
        let mut score = 0;
        if prev_cat != next_cat {
            score += 10;
            if prev_cat == 0 && next_cat == 1 {
                score += 1;
            }
        }
        score += CATEGORY_BOUNDARY[prev_cat as usize];
        score += CATEGORY_BOUNDARY[next_cat as usize];
        score
    }

    /// Slice `line` between char indices `[a, b)` using cached byte offsets.
    fn slice<'a>(line: &'a str, bytes: &[usize], a: usize, b: usize) -> &'a str {
        &line[bytes[a]..bytes[b]]
    }
}

fn char_code(c: char) -> i64 {
    c as u32 as i64
}

fn is_word_char(code: i64) -> bool {
    (97..=122).contains(&code) || (65..=90).contains(&code) || (48..=57).contains(&code)
}

/// CharBoundaryCategory: 0 lower, 1 upper, 2 digit, 3 end, 4 other, 5 separator,
/// 6 space, 7 CR, 8 LF.
fn boundary_category(code: i64) -> i64 {
    match code {
        10 => 8,
        13 => 7,
        _ if is_space_code(code) => 6,
        c if (97..=122).contains(&c) => 0,
        c if (65..=90).contains(&c) => 1,
        c if (48..=57).contains(&c) => 2,
        -1 => 3,
        44 | 59 => 5,
        _ => 4,
    }
}

fn is_space_code(code: i64) -> bool {
    code == 32
        || code == 9
        || code == 11
        || code == 12
        || code == 0xa0
        || code == 0x1680
        || (0x2000..=0x200a).contains(&code)
        || code == 0x2028
        || code == 0x2029
        || code == 0x202f
        || code == 0x205f
        || code == 0x3000
        || code == 0xfeff
}

const CATEGORY_BOUNDARY: [i64; 9] = [0, 0, 0, 10, 2, 30, 3, 10, 10];

// ───────────────────────────────── algorithms ─────────────────────────────────

fn trivial_diff(seq1: &CharSeq, seq2: &CharSeq) -> SequenceDiff {
    SequenceDiff {
        seq1: OffsetRange {
            start: 0,
            end: seq1.length(),
        },
        seq2: OffsetRange {
            start: 0,
            end: seq2.length(),
        },
    }
}

/// DynamicProgrammingDiffing: LCS with diagonal preference (no equality score,
/// as refineDiff calls it).
fn dp_diff(seq1: &CharSeq, seq2: &CharSeq) -> Vec<SequenceDiff> {
    if seq1.length() == 0 || seq2.length() == 0 {
        return vec![trivial_diff(seq1, seq2)];
    }
    let n = seq1.length() as usize;
    let m = seq2.length() as usize;
    let cols = m + 1;
    let mut lcs_lengths = vec![0i64; n * cols];
    let mut directions = vec![0u8; n * cols];
    let mut lengths = vec![0i64; n * cols];
    for s1 in 0..n {
        for s2 in 0..m {
            let horizontal_len = if s1 == 0 { 0 } else { lcs_lengths[(s1 - 1) * cols + s2] };
            let vertical_len = if s2 == 0 { 0 } else { lcs_lengths[s1 * cols + s2 - 1] };
            let extended_seq_score: i64;
            if seq1.element(s1 as i64) == seq2.element(s2 as i64) {
                let mut sc = if s1 == 0 || s2 == 0 {
                    0
                } else {
                    lcs_lengths[(s1 - 1) * cols + (s2 - 1)]
                };
                if s1 > 0 && s2 > 0 && directions[(s1 - 1) * cols + (s2 - 1)] == 3 {
                    sc += lengths[(s1 - 1) * cols + (s2 - 1)];
                }
                sc += 1;
                extended_seq_score = sc;
            } else {
                extended_seq_score = -1;
            }
            let new_value = horizontal_len.max(vertical_len).max(extended_seq_score);
            if new_value == extended_seq_score {
                lengths[s1 * cols + s2] =
                    (if s1 > 0 && s2 > 0 {
                        lengths[(s1 - 1) * cols + (s2 - 1)]
                    } else {
                        0
                    }) + 1;
                directions[s1 * cols + s2] = 3;
            } else if new_value == horizontal_len {
                lengths[s1 * cols + s2] = 0;
                directions[s1 * cols + s2] = 1;
            } else {
                lengths[s1 * cols + s2] = 0;
                directions[s1 * cols + s2] = 2;
            }
            lcs_lengths[s1 * cols + s2] = new_value;
        }
    }
    let mut result: Vec<SequenceDiff> = Vec::new();
    let mut last1 = n as i64;
    let mut last2 = m as i64;
    let report =
        |result: &mut Vec<SequenceDiff>, s1: i64, s2: i64, last1: &mut i64, last2: &mut i64| {
            if s1 + 1 != *last1 || s2 + 1 != *last2 {
                result.push(SequenceDiff {
                    seq1: OffsetRange {
                        start: s1 + 1,
                        end: *last1,
                    },
                    seq2: OffsetRange {
                        start: s2 + 1,
                        end: *last2,
                    },
                });
            }
            *last1 = s1;
            *last2 = s2;
        };
    let mut s1 = n as i64 - 1;
    let mut s2 = m as i64 - 1;
    while s1 >= 0 && s2 >= 0 {
        match directions[s1 as usize * cols + s2 as usize] {
            3 => {
                report(&mut result, s1, s2, &mut last1, &mut last2);
                s1 -= 1;
                s2 -= 1;
            }
            1 => s1 -= 1,
            _ => s2 -= 1,
        }
    }
    report(&mut result, -1, -1, &mut last1, &mut last2);
    result.reverse();
    result
}

/// MyersDiffAlgorithm: O(ND) with the diagonal bounds VSCode adds.
fn myers_diff(seq1: &CharSeq, seq2: &CharSeq) -> Vec<SequenceDiff> {
    if seq1.length() == 0 || seq2.length() == 0 {
        return vec![trivial_diff(seq1, seq2)];
    }
    let n = seq1.length();
    let m = seq2.length();
    let get_x_after_snake = |x: i64, y: i64| -> i64 {
        let mut x = x;
        let mut y = y;
        while x < n && y < m && seq1.element(x) == seq2.element(y) {
            x += 1;
            y += 1;
        }
        x
    };
    #[derive(Clone)]
    struct Path {
        prev: Option<Box<Path>>,
        x: i64,
        y: i64,
        len: i64,
    }
    let mut v: HashMap<i64, i64> = HashMap::new();
    let mut paths: HashMap<i64, Option<Box<Path>>> = HashMap::new();
    let x0 = get_x_after_snake(0, 0);
    v.insert(0, x0);
    paths.insert(
        0,
        if x0 == 0 {
            None
        } else {
            Some(Box::new(Path {
                prev: None,
                x: 0,
                y: 0,
                len: x0,
            }))
        },
    );
    let mut d: i64 = 0;
    let mut k: i64 = 0;
    let mut done = false;
    while !done {
        d += 1;
        let lower_bound = -min(d, m + (d % 2));
        let upper_bound = min(d, n + (d % 2));
        let mut kk = lower_bound;
        while kk <= upper_bound {
            let max_xofd_line_top = if kk == upper_bound {
                -1
            } else {
                *v.get(&(kk + 1)).unwrap_or(&-1)
            };
            let max_xofd_line_left = if kk == lower_bound {
                -1
            } else {
                v.get(&(kk - 1)).copied().unwrap_or(-1) + 1
            };
            let x = max_xofd_line_top.max(max_xofd_line_left).min(n);
            let y = x - kk;
            if x > n || y > m {
                kk += 2;
                continue;
            }
            let new_max_x = get_x_after_snake(x, y);
            v.insert(kk, new_max_x);
            let last_path = if x == max_xofd_line_top {
                paths.get(&(kk + 1)).cloned().unwrap_or(None)
            } else {
                paths.get(&(kk - 1)).cloned().unwrap_or(None)
            };
            paths.insert(
                kk,
                if new_max_x != x {
                    Some(Box::new(Path {
                        prev: last_path,
                        x,
                        y,
                        len: new_max_x - x,
                    }))
                } else {
                    last_path
                },
            );
            if *v.get(&kk).unwrap_or(&-1) == n && (*v.get(&kk).unwrap_or(&-1) - kk) == m {
                done = true;
                k = kk;
                break;
            }
            kk += 2;
        }
    }
    let mut path = paths.get(&k).cloned().unwrap_or(None);
    let mut result: Vec<SequenceDiff> = Vec::new();
    let mut last1 = n;
    let mut last2 = m;
    loop {
        let (end_x, end_y) = match &path {
            Some(p) => (p.x + p.len, p.y + p.len),
            None => (0, 0),
        };
        if end_x != last1 || end_y != last2 {
            result.push(SequenceDiff {
                seq1: OffsetRange {
                    start: end_x,
                    end: last1,
                },
                seq2: OffsetRange {
                    start: end_y,
                    end: last2,
                },
            });
        }
        match path {
            Some(p) => {
                last1 = p.x;
                last2 = p.y;
                path = p.prev;
            }
            None => break,
        }
    }
    result.reverse();
    result
}

// ───────────────────────────────── heuristics ─────────────────────────────────

/// optimizeSequenceDiffs: join-by-shifting twice, then boundary-score shifting.
fn optimize_sequence_diffs(
    seq1: &CharSeq,
    seq2: &CharSeq,
    diffs: Vec<SequenceDiff>,
) -> Vec<SequenceDiff> {
    let result = join_sequence_diffs_by_shifting(seq1, seq2, diffs);
    let result = join_sequence_diffs_by_shifting(seq1, seq2, result);
    shift_sequence_diffs(seq1, seq2, result)
}

fn join_sequence_diffs_by_shifting(
    seq1: &CharSeq,
    seq2: &CharSeq,
    diffs: Vec<SequenceDiff>,
) -> Vec<SequenceDiff> {
    if diffs.is_empty() {
        return diffs;
    }
    let mut result: Vec<SequenceDiff> = vec![diffs[0]];
    for i in 1..diffs.len() {
        let prev_result = result[result.len() - 1];
        let mut cur = diffs[i];
        if cur.seq1.is_empty() || cur.seq2.is_empty() {
            let length = cur.seq1.start - prev_result.seq1.end;
            let mut d = 1i64;
            while d <= length {
                if seq1.element(cur.seq1.start - d) != seq1.element(cur.seq1.end - d)
                    || seq2.element(cur.seq2.start - d) != seq2.element(cur.seq2.end - d)
                {
                    break;
                }
                d += 1;
            }
            d -= 1;
            if d == length {
                let n = result.len() - 1;
                result[n] = SequenceDiff {
                    seq1: OffsetRange {
                        start: prev_result.seq1.start,
                        end: cur.seq1.end - length,
                    },
                    seq2: OffsetRange {
                        start: prev_result.seq2.start,
                        end: cur.seq2.end - length,
                    },
                };
                continue;
            }
            cur = delta_diff(cur, -d);
        }
        result.push(cur);
    }
    let mut result2: Vec<SequenceDiff> = Vec::new();
    for i in 0..result.len().saturating_sub(1) {
        let mut cur = result[i];
        let mut next_result = result[i + 1];
        if cur.seq1.is_empty() || cur.seq2.is_empty() {
            let length = next_result.seq1.start - cur.seq1.end;
            let mut d = 0i64;
            while d < length {
                if !seq1.is_strongly_equal(cur.seq1.start + d, cur.seq1.end + d)
                    || !seq2.is_strongly_equal(cur.seq2.start + d, cur.seq2.end + d)
                {
                    break;
                }
                d += 1;
            }
            if d == length {
                result[i + 1] = SequenceDiff {
                    seq1: OffsetRange {
                        start: cur.seq1.start + length,
                        end: next_result.seq1.end,
                    },
                    seq2: OffsetRange {
                        start: cur.seq2.start + length,
                        end: next_result.seq2.end,
                    },
                };
                continue;
            }
            if d > 0 {
                cur = delta_diff(cur, d);
            }
        }
        let _ = &mut next_result;
        result2.push(cur);
    }
    if !result.is_empty() {
        let last = result[result.len() - 1];
        result2.push(last);
    }
    result2
}

fn shift_sequence_diffs(
    seq1: &CharSeq,
    seq2: &CharSeq,
    mut diffs: Vec<SequenceDiff>,
) -> Vec<SequenceDiff> {
    for i in 0..diffs.len() {
        let prev_diff = if i > 0 { Some(diffs[i - 1]) } else { None };
        let diff = diffs[i];
        let next_diff = if i + 1 < diffs.len() {
            Some(diffs[i + 1])
        } else {
            None
        };
        let seq1_valid = OffsetRange {
            start: prev_diff.map(|d| d.seq1.end + 1).unwrap_or(0),
            end: next_diff.map(|d| d.seq1.start - 1).unwrap_or(seq1.length()),
        };
        let seq2_valid = OffsetRange {
            start: prev_diff.map(|d| d.seq2.end + 1).unwrap_or(0),
            end: next_diff.map(|d| d.seq2.start - 1).unwrap_or(seq2.length()),
        };
        if diff.seq1.is_empty() {
            diffs[i] = shift_diff_to_better_position(diff, seq1, seq2, seq1_valid, seq2_valid);
        } else if diff.seq2.is_empty() {
            let swapped = SequenceDiff {
                seq1: diff.seq2,
                seq2: diff.seq1,
            };
            let shifted = shift_diff_to_better_position(swapped, seq2, seq1, seq2_valid, seq1_valid);
            diffs[i] = SequenceDiff {
                seq1: shifted.seq2,
                seq2: shifted.seq1,
            };
        }
    }
    diffs
}

fn shift_diff_to_better_position(
    diff: SequenceDiff,
    seq1: &CharSeq,
    _seq2_ref_unused: &CharSeq,
    seq1_valid: OffsetRange,
    seq2_valid: OffsetRange,
) -> SequenceDiff {
    // seq1 here is the *target* sequence (the one that is unchanged for this
    // pure insertion); _seq2_ref_unused is the other. Kept for signature clarity.
    let max_shift_limit = 100;
    // The TS uses `seq2` for the strongly-equal walk and boundary scores on both
    // seq1/seq2. Here the arguments already reflect that orientation.
    let seq2 = _seq2_ref_unused;
    let mut delta_before = 1i64;
    while diff.seq1.start - delta_before >= seq1_valid.start
        && diff.seq2.start - delta_before >= seq2_valid.start
        && seq2.is_strongly_equal(diff.seq2.start - delta_before, diff.seq2.end - delta_before)
        && delta_before < max_shift_limit
    {
        delta_before += 1;
    }
    delta_before -= 1;
    let mut delta_after = 0i64;
    while diff.seq1.start + delta_after < seq1_valid.end
        && diff.seq2.end + delta_after < seq2_valid.end
        && seq2.is_strongly_equal(diff.seq2.start + delta_after, diff.seq2.end + delta_after)
        && delta_after < max_shift_limit
    {
        delta_after += 1;
    }
    if delta_before == 0 && delta_after == 0 {
        return diff;
    }
    let mut best_delta = 0i64;
    let mut best_score = -1i64;
    for delta in -delta_before..=delta_after {
        let score = seq1.boundary_score(diff.seq1.start + delta)
            + seq2.boundary_score(diff.seq2.start + delta)
            + seq2.boundary_score(diff.seq2.end + delta);
        if score > best_score {
            best_score = score;
            best_delta = delta;
        }
    }
    delta_diff(diff, best_delta)
}

/// Merge adjacent change diffs. VSCode's `removeShortMatches` merges diffs
/// separated by an equal gap of up to 2 characters; we use gap <= 1 - enough to
/// coalesce frequent small edits (avoiding "a diff every other character"), while
/// keeping a genuinely separate short common run (e.g. a 2-char CJK word like
/// 系统) from being absorbed into a surrounding substitution.
fn remove_short_matches(diffs: &[SequenceDiff]) -> Vec<SequenceDiff> {
    let mut result: Vec<SequenceDiff> = Vec::new();
    for s in diffs {
        match result.last_mut() {
            None => result.push(*s),
            Some(last) => {
                if s.seq1.start - last.seq1.end <= 1 || s.seq2.start - last.seq2.end <= 1 {
                    *last = join_diffs(*last, *s);
                } else {
                    result.push(*s);
                }
            }
        }
    }
    result
}

/// extendDiffsToEntireWordIfAppropriate.
fn extend_diffs_to_words(
    seq1: &CharSeq,
    seq2: &CharSeq,
    diffs: Vec<SequenceDiff>,
) -> Vec<SequenceDiff> {
    let mut queue = invert_diffs(&diffs, seq1.length(), seq2.length());
    let mut additional: Vec<SequenceDiff> = Vec::new();
    let mut last_point: (i64, i64) = (0, 0);

    while !queue.is_empty() {
        let next = queue.remove(0);
        if next.seq1.is_empty() {
            continue;
        }
        // scanWord(start), then scanWord(end-1)
        scan_word(
            seq1,
            seq2,
            &mut queue,
            &mut additional,
            &mut last_point,
            next.seq1.start,
            next.seq2.start,
            next,
        );
        let (e1, e2) = diff_ends(next);
        scan_word(
            seq1,
            seq2,
            &mut queue,
            &mut additional,
            &mut last_point,
            e1 - 1,
            e2 - 1,
            next,
        );
    }
    merge_sorted_diffs(&diffs, &additional)
}

fn scan_word(
    seq1: &CharSeq,
    seq2: &CharSeq,
    queue: &mut Vec<SequenceDiff>,
    additional: &mut Vec<SequenceDiff>,
    last_point: &mut (i64, i64),
    offset1: i64,
    offset2: i64,
    equal_mapping: SequenceDiff,
) {
    if offset1 < last_point.0 || offset2 < last_point.1 {
        return;
    }
    let (w1, w2) = match (seq1.find_word_containing(offset1), seq2.find_word_containing(offset2)) {
        (Some(a), Some(b)) => (a, b),
        _ => return,
    };
    let mut w = SequenceDiff { seq1: w1, seq2: w2 };
    let equal_part = intersect_diffs(w, equal_mapping);
    let mut equal_chars1 = equal_part.map(|e| e.seq1.len()).unwrap_or(0);
    let mut equal_chars2 = equal_part.map(|e| e.seq2.len()).unwrap_or(0);
    while !queue.is_empty() {
        let next = queue[0];
        if !intersects_ranges(w.seq1, next.seq1) && !intersects_ranges(w.seq2, next.seq2) {
            break;
        }
        let (v1, v2) =
            match (seq1.find_word_containing(next.seq1.start), seq2.find_word_containing(next.seq2.start)) {
                (Some(a), Some(b)) => (a, b),
                _ => break,
            };
        let v = SequenceDiff { seq1: v1, seq2: v2 };
        if let Some(next_equal) = intersect_diffs(v, next) {
            equal_chars1 += next_equal.seq1.len();
            equal_chars2 += next_equal.seq2.len();
        }
        w = join_diffs(w, v);
        if w.seq1.end >= next.seq1.end {
            queue.remove(0);
        } else {
            break;
        }
    }
    if equal_chars1 + equal_chars2 < (w.seq1.len() + w.seq2.len()) * 2 / 3 {
        additional.push(w);
    }
    *last_point = diff_ends(w);
}

fn merge_sorted_diffs(diffs1: &[SequenceDiff], diffs2: &[SequenceDiff]) -> Vec<SequenceDiff> {
    let mut result: Vec<SequenceDiff> = Vec::new();
    let mut one: Vec<SequenceDiff> = diffs1.to_vec();
    let mut two: Vec<SequenceDiff> = diffs2.to_vec();
    while !one.is_empty() || !two.is_empty() {
        let take_one = {
            let a = one.first();
            let b = two.first();
            a.is_some() && (b.is_none() || a.unwrap().seq1.start < b.unwrap().seq1.start)
        };
        let next = if take_one {
            one.remove(0)
        } else {
            two.remove(0)
        };
        if let Some(last) = result.last_mut() {
            if last.seq1.end >= next.seq1.start {
                *last = join_diffs(*last, next);
                continue;
            }
        }
        result.push(next);
    }
    result
}

/// removeVeryShortMatchingTextBetweenLongDiffs: the cap-130 join rule and the
/// short prefix/suffix extension.
fn remove_very_short_matching_text(
    seq1: &CharSeq,
    seq2: &CharSeq,
    diffs: Vec<SequenceDiff>,
) -> Vec<SequenceDiff> {
    if diffs.is_empty() {
        return diffs;
    }
    let mut current = diffs;
    let mut counter = 0;
    let mut should_repeat = true;
    while should_repeat && counter < 10 {
        counter += 1;
        should_repeat = false;
        let mut result: Vec<SequenceDiff> = vec![current[0]];
        for i in 1..current.len() {
            let cur = current[i];
            let last_result = result[result.len() - 1];
            let unchanged_range = OffsetRange {
                start: last_result.seq1.end,
                end: cur.seq1.start,
            };
            let unchanged_line_count = seq1.count_lines_in(unchanged_range);
            if unchanged_line_count > 5 || unchanged_range.len() > 500 {
                result.push(cur);
                continue;
            }
            let unchanged_text = seq1.text(unchanged_range);
            let trimmed = unchanged_text.trim();
            if trimmed.chars().count() > 20 || trimmed.contains('\n') {
                result.push(cur);
                continue;
            }
            let max = 2.0f64 * 40.0 + 50.0; // 130
            let cap = |v: f64| v.min(max);
            let power = |v: f64| v.powf(1.5);
            // VSCode: power(cap(countLinesIn(r)*40 + rangeLen(r))) per side, summed, then power.
            let term = |s: &CharSeq, r: OffsetRange| {
                power(cap(s.count_lines_in(r) as f64 * 40.0 + r.len() as f64))
            };
            let before = power(
                term(seq1, last_result.seq1) + term(seq2, last_result.seq2),
            );
            let after = power(term(seq1, cur.seq1) + term(seq2, cur.seq2));
            if before + after > max.powf(1.5).powf(1.5) * 1.3 {
                should_repeat = true;
                let n = result.len() - 1;
                result[n] = join_diffs(last_result, cur);
            } else {
                result.push(cur);
            }
        }
        current = result;
    }

    let mut out: Vec<SequenceDiff> = Vec::new();
    for i in 0..current.len() {
        let mut new_diff = current[i];
        let prev = if i > 0 { Some(current[i - 1]) } else { None };
        let next = if i + 1 < current.len() {
            Some(current[i + 1])
        } else {
            None
        };
        let mark_changed = |text: &str, nd: SequenceDiff| -> bool {
            !text.is_empty()
                && text.trim().chars().count() <= 3
                && nd.seq1.len() + nd.seq2.len() > 100
        };
        let full_range1 = seq1.extend_to_full_lines(new_diff.seq1);
        let prefix = seq1.text(OffsetRange {
            start: full_range1.start,
            end: new_diff.seq1.start,
        });
        if mark_changed(&prefix, new_diff) {
            new_diff = delta_start_diff(new_diff, -(prefix.chars().count() as i64));
        }
        let suffix = seq1.text(OffsetRange {
            start: new_diff.seq1.end,
            end: full_range1.end,
        });
        if mark_changed(&suffix, new_diff) {
            new_diff = delta_end_diff(new_diff, suffix.chars().count() as i64);
        }
        let available = SequenceDiff {
            seq1: OffsetRange {
                start: prev.map(|p| p.seq1.end).unwrap_or(0),
                end: next.map(|n| n.seq1.start).unwrap_or(i64::MAX),
            },
            seq2: OffsetRange {
                start: prev.map(|p| p.seq2.end).unwrap_or(0),
                end: next.map(|n| n.seq2.start).unwrap_or(i64::MAX),
            },
        };
        let clipped = match intersect_diffs(new_diff, available) {
            Some(c) => c,
            None => continue,
        };
        if let Some(last) = out.last_mut() {
            if last.seq1.end == clipped.seq1.start && last.seq2.end == clipped.seq2.start {
                *last = join_diffs(*last, clipped);
                continue;
            }
        }
        out.push(clipped);
    }
    out
}

// ───────────────────────────────── block entry ─────────────────────────────────

/// Mark one deletion run against one addition run — one change block in VSCode
/// terms: a single character diff over both trimmed-line sequences, the heuristic
/// chain, then ranges mapped back to per-line spans.
pub fn block_diff_runs<'a>(
    del_run: &[&'a str],
    add_run: &[&'a str],
) -> (Vec<Vec<DiffSpan<'a>>>, Vec<Vec<DiffSpan<'a>>>) {
    let whole_del: Vec<Vec<DiffSpan>> = del_run
        .iter()
        .map(|t| whole_line(t, Mark::Del))
        .collect();
    let whole_ins: Vec<Vec<DiffSpan>> = add_run
        .iter()
        .map(|t| whole_line(t, Mark::Ins))
        .collect();
    if del_run.is_empty() || add_run.is_empty() {
        return (whole_del, whole_ins);
    }
    let seq1 = CharSeq::new(del_run);
    let seq2 = CharSeq::new(add_run);
    if seq1.length() * seq2.length() > BLOCK_PRODUCT_LIMIT {
        return (whole_del, whole_ins);
    }
    let mut diffs = if seq1.length() + seq2.length() < DP_LENGTH_LIMIT {
        dp_diff(&seq1, &seq2)
    } else {
        myers_diff(&seq1, &seq2)
    };
    diffs = optimize_sequence_diffs(&seq1, &seq2, diffs);
    diffs = extend_diffs_to_words(&seq1, &seq2, diffs);
    diffs = remove_short_matches(&diffs);
    diffs = remove_very_short_matching_text(&seq1, &seq2, diffs);
    // A line is a "whitespace pair" if it maps 1:1 to a counterpart with equal
    // trimmed content (a pure indentation change) — its whitespace marks are kept.
    let ws = |lines: &[&str], other: &[&str]| -> Vec<bool> {
        (0..lines.len())
            .map(|i| other.get(i).is_some_and(|o| lines[i].trim() == o.trim()))
            .collect()
    };
    let del_ws = ws(del_run, add_run);
    let add_ws = ws(add_run, del_run);
    (
        diffs_to_line_spans(del_run, &seq1, &seq2, &diffs, true, &del_ws),
        diffs_to_line_spans(add_run, &seq2, &seq1, &diffs, false, &add_ws),
    )
}

/// A whole-line span pair for an unpaired deletion or addition.
fn whole_line<'a>(text: &'a str, mark: Mark) -> Vec<DiffSpan<'a>> {
    if text.is_empty() {
        Vec::new()
    } else {
        vec![DiffSpan {
            text,
            mark: Some(mark),
            spans_line: false,
        }]
    }
}

/// Cut each raw line at its side's marked ranges. A change whose side is empty
/// renders as a zero-width bar (VSCode's empty-change marker); a whitespace-only
/// change containing a line separator renders a placeholder at that line's end
/// (a split/join boundary).
fn diffs_to_line_spans<'a>(
    lines: &[&'a str],
    seq: &CharSeq,
    other_seq: &CharSeq,
    diffs: &[SequenceDiff],
    left: bool,
    ws_pair: &[bool],
) -> Vec<Vec<DiffSpan<'a>>> {
    let line_char_counts: Vec<usize> = lines.iter().map(|l| l.chars().count()).collect();
    let mut marks: Vec<Vec<bool>> = lines
        .iter()
        .map(|l| vec![false; l.chars().count()])
        .collect();
    // (char position within line, drop): `drop` is true for a bar whose counterpart
    // is a whole line that carries content (it gets its own row); intra-line bars and
    // pure-newline boundary bars have drop=false and render as placeholders.
    let mut bars: Vec<Vec<(usize, bool)>> = vec![Vec::new(); lines.len()];
    let len = seq.length();
    for diff in diffs {
        let range = if left { diff.seq1 } else { diff.seq2 };
        let other = if left { diff.seq2 } else { diff.seq1 };
        let other_has_nl = (other.start.max(0)..other.end).any(|k| other_seq.element(k) == '\n');
        let other_has_content =
            (other.start.max(0)..other.end).any(|k| !other_seq.element(k).is_whitespace());
        if range.is_empty() && !other.is_empty() {
            let drop = other_has_nl && other_has_content;
            let p = range.start.clamp(0, len);
            let line = seq.line_of(p.min(len.max(1) - 1));
            if let Some(&count) = line_char_counts.get(line) {
                let raw_at = (p - seq.line_start[line] as i64 + seq.trimmed_ws[line] as i64)
                    .clamp(0, count as i64) as usize;
                bars[line].push((raw_at, drop));
            }
            continue;
        }
        let start = range.start.max(0);
        let end = range.end.min(len);
        let mut at = start;
        while at < end {
            let line = seq.line_of(at);
            if at as usize == seq.line_trimmed_end[line] {
                // the `\n` separator element marks no character
                at += 1;
                continue;
            }
            let raw_at = at - seq.line_start[line] as i64 + seq.trimmed_ws[line] as i64;
            let count = line_char_counts[line];
            let ch = seq.element(at);
            // Keep a whitespace mark when it is a pure-indentation pair line, a
            // newline boundary (other side has a '\n'), or a pure insertion/deletion
            // (other side empty -> the whitespace is real added/removed content).
            // Otherwise (space<->space reflow) it is noise and is dropped.
            let keep = if ch.is_whitespace() {
                ws_pair.get(line).copied().unwrap_or(false) || other_has_nl || other.is_empty()
            } else {
                true
            };
            if keep && raw_at >= 0 && (raw_at as usize) < count {
                marks[line][raw_at as usize] = true;
            }
            at += 1;
        }
    }
    let mark_kind = if left { Mark::Del } else { Mark::Ins };
    // A bar marks the OTHER side's change, so it uses the opposite emphasis:
    // a bar on the minus side means the plus side inserted content there -> green;
    // a bar on the plus side means the minus side deleted content there -> red.
    let bar_kind = if left { Mark::Ins } else { Mark::Del };
    lines
        .iter()
        .enumerate()
        .map(|(line_idx, &line)| {
            let mask = &marks[line_idx];
            let mut line_bars = bars[line_idx].clone();
            line_bars.sort_unstable();
            let bytes = &seq.char_bytes[line_idx];
            let n = bytes.len() - 1; // char count of `line`
            let push_run =
                |spans: &mut Vec<DiffSpan<'a>>, a: usize, b: usize, marked: bool| {
                    if b > a {
                        let text = CharSeq::slice(line, bytes, a, b);
                        spans.push(DiffSpan {
                            text,
                            mark: if marked { Some(mark_kind) } else { None },
                            spans_line: false,
                        });
                    }
                };
            let mut spans: Vec<DiffSpan<'a>> = Vec::new();
            let mut run_start = 0usize;
            let mut run_marked = mask.first().copied().unwrap_or(false);
            let mut bi = 0usize;
            let mut i = 0usize;
            while i <= n {
                while bi < line_bars.len() && line_bars[bi].0 == i {
                    push_run(&mut spans, run_start, i, run_marked);
                    run_start = i;
                    spans.push(DiffSpan {
                        text: "",
                        mark: Some(bar_kind),
                        spans_line: line_bars[bi].1,
                    });
                    bi += 1;
                }
                if i == n {
                    break;
                }
                let is_marked = mask[i];
                if is_marked != run_marked {
                    push_run(&mut spans, run_start, i, run_marked);
                    run_start = i;
                    run_marked = is_marked;
                }
                i += 1;
            }
            push_run(&mut spans, run_start, n, run_marked);
            spans
        })
        .collect()
}

// ───────────────────────────────── widening ─────────────────────────────────

/// A line widened with one-cell placeholders for intra-line empty-change bars.
/// `ranges` maps each rendered piece to `(is_emphasis, byte_start, byte_end)`
/// in `text`; the caller maps `is_emphasis` to the emph/base style and slices
/// `text`. Intra-line bars become a single placeholder space; whole-line
/// counterpart bars are dropped (their line already has its own row).
pub struct WidenedLine {
    pub text: String,
    pub ranges: Vec<(Option<Mark>, usize, usize)>,
}

/// Render one line's spans into a widened line + ranges (see [`WidenedLine`]).
///
/// A zero-width bar is rendered as a one-cell placeholder only when it is an
/// intra-line change (`spans_line == false`); whole-line counterpart bars are
/// dropped because that line already has its own row in side-by-side.
pub fn widen_line(spans: &[DiffSpan]) -> WidenedLine {
    let mut text = String::new();
    let mut ranges: Vec<(Option<Mark>, usize, usize)> = Vec::new();
    for s in spans {
        if s.text.is_empty() {
            if let Some(m) = s.mark {
                if !s.spans_line {
                    let start = text.len();
                    // Braille blank: renders as an empty cell but is NOT Unicode
                    // whitespace, so delta's trailing-whitespace-error styling does
                    // not hijack a placeholder that sits at a line's end.
                    text.push('\u{2800}');
                    ranges.push((Some(m), start, text.len()));
                }
            }
            continue;
        }
        let start = text.len();
        text.push_str(s.text);
        ranges.push((s.mark, start, text.len()));
    }
    WidenedLine { text, ranges }
}

// ───────────────────────────────── tests ─────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Concatenate spans: `#` fences a changed range, `│` is the cursor-wide bar.
    fn render(spans: &[DiffSpan]) -> String {
        spans
            .iter()
            .map(|s| match s.mark {
                None => s.text.to_string(),
                Some(_) if s.text.is_empty() => "\u{2502}".to_string(),
                Some(_) => format!("#{}#", s.text),
            })
            .collect()
    }

    #[test]
    fn marks_only_the_changed_chars() {
        let (del, ins) = block_diff_runs(&["hello world"], &["hello brave world"]);
        assert_eq!(render(&del[0]), "hello\u{2502} world");
        assert_eq!(render(&ins[0]), "hello# brave# world");
    }

    #[test]
    fn aligns_a_moved_line_as_common() {
        let (del, ins) = block_diff_runs(&["keep", "gone"], &["fresh", "keep"]);
        assert_eq!(render(&del[0]), "\u{2502}keep");
        assert_eq!(render(&ins[0]), "#fresh#");
        assert_eq!(render(&ins[1]), "keep\u{2502}");
    }

    #[test]
    fn collapses_a_wholly_rewritten_pair() {
        let (_del, ins) = block_diff_runs(&["gone one", "gone two"], &["kept"]);
        assert_eq!(render(&ins[0]), "#kept#");
    }

    #[test]
    fn pairs_rewritten_line_with_true_counterpart() {
        let (del, ins) = block_diff_runs(
            &["import { StyleSheet, TouchableOpacity } from"],
            &["import { WenShu } from", "import { StyleSheet } from"],
        );
        assert_eq!(
            render(&del[0]),
            "\u{2502}import { StyleSheet#, TouchableOpacity# } from"
        );
        assert_eq!(render(&ins[0]), "#import { WenShu } from#");
        assert_eq!(render(&ins[1]), "import { StyleSheet\u{2502} } from");
    }

    #[test]
    fn degrades_past_block_cap() {
        let del: Vec<String> = (0..60).map(|i| format!("line {} {}", i, "x".repeat(700))).collect();
        let add: Vec<String> = (0..60).map(|i| format!("line {} {}", i, "y".repeat(700))).collect();
        let del_refs: Vec<&str> = del.iter().map(|s| s.as_str()).collect();
        let add_refs: Vec<&str> = add.iter().map(|s| s.as_str()).collect();
        let (del_spans, ins_spans) = block_diff_runs(&del_refs, &add_refs);
        assert_eq!(del_spans.len(), 60);
        assert_eq!(ins_spans.len(), 60);
        assert!(render(&ins_spans[0]).contains('#'));
    }
}
