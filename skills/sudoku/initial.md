# Sudoku Basic Agent Skill

## Overview
Complete the CURRENT 9x9 Sudoku using visible row, column and 3x3-box constraints. Rows and columns are numbered 1..9. A dot is empty. Unmarked digits are immutable clues; a digit marked * is your own editable entry. This skill is shared by single-action and chunk agents.

## Subtasks
| Situation | Next objective |
|---|---|
| A row, column or box has one empty cell | Find its missing digit and verify the other two units |
| An empty cell has one legal candidate | Fill that forced cell |
| A missing digit has one possible cell in a unit | Fill that hidden single |
| A previous editable choice causes contradiction | Undo that choice, then reconsider |

## General Principles
1. Use the current board, not an earlier board. Count column positions carefully; the row label and * marker are not extra cells.
2. For an empty cell (r,c), start with digits 1..9 and remove every digit already in its row, column and containing 3x3 box. The box uses row bands 1-3/4-6/7-9 and the same column bands. The remainder is its candidate set.
3. Prefer a forced move: a cell with one remaining candidate, or a digit with only one allowed position in a row/column/box. For nearly completed boards, inspect the units with one missing digit first. Verify ALL three constraints before filling.
4. Use `set(r,c,v)` only on an empty cell. Fill one cell, observe the updated board, and recompute before the next single-action call. A locally accepted entry is not proof of a globally correct solution.
5. Do not guess when a forced move exists. If no forced move exists, any exploratory choice is tentative; record it through the executed-action history and reconsider it if contradictions appear. Never change an original clue.
6. To replace your own wrong entry, first issue `clear(r,c)` on a * cell. This uses one action. A later call may set a replacement. Do not attempt both commands in one single-action output.
7. Completion requires every cell filled consistently, not merely one row finished. If unfinished, do not return an empty action list.

## Common Mistakes
- Checking only the row while ignoring a column or box conflict.
- Swapping row and column, counting * as a cell, or writing into an occupied cell.
- Repeating a rejected set action without changing its coordinates or value.

## Output
In single-action mode return exactly `<actions>["set(r,c,v)"]</actions>` with concrete integer coordinates and value, or one legal clear command. Output only the action array, no reasoning, explanation, code or completed grid. Do not copy placeholders literally.
