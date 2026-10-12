# Rush Hour Basic Agent Skill

## Overview
Solve the current 6x6 board by sliding the horizontal target A in row 3 to columns 5 and 6. Read rows/columns as 1-based coordinates. This is a basic task skill shared by single-action and chunk agents, not a learned interrupt policy.

## Subtasks
| Situation | Next objective |
|---|---|
| A has a clear path to the right edge | Slide A directly to columns 5-6 |
| A's exit corridor is blocked | Move a blocking vehicle out of row 3 |
| That blocker cannot move far enough | Clear the cells it needs using another vehicle |
| A move failed | Re-read the unchanged board and choose a different legal move |

## General Principles
1. Locate every cell of the vehicle you intend to move. Same-row cells imply horizontal movement; same-column cells imply vertical movement. A vehicle's length and direction never change.
2. For a horizontal vehicle use + for right and - for left; for a vertical vehicle use + for down and - for up. The number is DISTANCE, not a destination coordinate. All newly occupied and traversed cells must be empty and inside the board. Its own currently occupied cells are not obstacles.
3. Check A's corridor FIRST: every cell in row 3 strictly to the right of A must be empty. If clear, let c be A's leftmost column. The winning slide is A+(5-c), a single action even when the distance exceeds one. Do not stop one column before the exit.
4. If blocked, identify vehicles intersecting that corridor. A vertical blocker must move far enough up or down that NONE of its cells remain in row 3. Check both directions and the complete path before choosing a move.
5. If a blocker is trapped, identify which vehicle occupies its required destination/path cells and move that vehicle first. Moving another vehicle away from A can be necessary. Do not move A or a blocker back and forth without clearing a dependency.
6. After each action, use the NEW board as authoritative. Recompute coordinates, available distances and blocking relations; never reuse an old distance blindly. An illegal move does not advance the plan.

## Common Mistakes
- Moving a horizontal vehicle vertically, jumping occupied cells, or leaving the board.
- Confusing the exit with column 5 for A's rightmost cell: A must occupy BOTH columns 5 and 6.
- Repeating an invalid action on an unchanged board or mistaking a partial slide for completion.

## Output
Follow the caller's action count. In single-action mode return exactly `<actions>["ID+N"]</actions>` (or ID-N) with one concrete legal command. No reasoning, prose, code, solved board or empty array. Do not copy the placeholder literally.
